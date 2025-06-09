#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_architecture.py

Contains all model component classes for the GNN-Coupled MoE model.
- ExpertGraphConv
- ExpertBlock
- GNNExpertCoupler
- GNNMoELayer
- GNNMoEModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional # Used in GNNExpertCoupler forward
import itertools # For generating pairs/triplets for hyperedges

# Attempt to import PyTorch Geometric, fail gracefully if not available for GNN mode
try:
    from torch_geometric.nn import HypergraphConv
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    HypergraphConv = None # Placeholder if PyG not installed
    Data = None
    Batch = None

# Assuming GNNMoEConfig will be imported from gnn_moe_config.py in the main script

class ExpertGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.message_weight = nn.Linear(in_dim * 2, 1)
        self.adjacency_logits = nn.Parameter(torch.randn(num_experts, num_experts))

    def forward(self, expert_features):
        batch_size, seq_len, num_experts, embed_dim = expert_features.shape
        adjacency = torch.sigmoid(self.adjacency_logits)
        flat_features = expert_features.view(-1, num_experts, embed_dim) # (B*L, E, D)
        
        updated_features_list = []
        for expert_idx in range(num_experts):
            current_expert = flat_features[:, expert_idx, :] # (B*L, D)
            messages = []
            for other_idx in range(num_experts):
                if other_idx != expert_idx:
                    other_expert = flat_features[:, other_idx, :] # (B*L, D)
                    concat_features = torch.cat([current_expert, other_expert], dim=1) # (B*L, 2D)
                    content_weight = torch.sigmoid(self.message_weight(concat_features).squeeze(-1)) # (B*L)
                    message_strength = adjacency[expert_idx, other_idx] * content_weight # (B*L)
                    weighted_message = other_expert * message_strength.unsqueeze(1) # (B*L, D)
                    messages.append(weighted_message)
            
            if messages:
                neighbor_msg = torch.stack(messages, dim=1).sum(dim=1) # Sum over neighbors: (B*L, D)
                neighbor_out = self.neighbor_transform(neighbor_msg)
            else: # Should not happen if num_experts > 1
                neighbor_out = torch.zeros_like(self.neighbor_transform(current_expert))
            
            self_out = self.self_transform(current_expert)
            updated_expert = F.gelu(neighbor_out + self_out) # (B*L, D)
            updated_features_list.append(updated_expert)
            
        updated_stack = torch.stack(updated_features_list, dim=1) # (B*L, E, D)
        return updated_stack.view(batch_size, seq_len, num_experts, embed_dim)

    def get_adjacency_matrix(self):
        return torch.sigmoid(self.adjacency_logits).detach()

class PyGHypergraphConvWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1, concat=True, dropout=0.0, pyg_conv_type="HypergraphConv"):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is not available. Cannot use PyGHypergraphConvWrapper.")
        
        if pyg_conv_type == "HypergraphConv":
            self.conv = HypergraphConv(in_channels, out_channels, use_attention=use_attention, heads=heads, concat=concat, dropout=dropout)
        else:
            # Placeholder for other PyG hypergraph layers if we want to experiment
            raise NotImplementedError(f"PyG convolution type '{pyg_conv_type}' not yet supported in this wrapper.")
        
        self.concat = concat
        self.heads = heads
        self.out_channels_final = out_channels * heads if concat and use_attention else out_channels

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        # x: node features [num_nodes, in_channels]
        # hyperedge_index: [2, num_total_hyperedge_connections]
        # hyperedge_weight: [num_hyperedges] - optional weights for hyperedges
        
        # Validate inputs to prevent CUDA assertion errors
        if hyperedge_index.numel() > 0:
            max_node_idx = hyperedge_index[0].max().item()
            
            # Critical validation: node indices must be within bounds
            if max_node_idx >= x.shape[0]:
                raise ValueError(f"Node index out of bounds: {max_node_idx} >= {x.shape[0]}")
            
            if hyperedge_weight is not None:
                max_hyperedge_id = hyperedge_index[1].max().item()
                if max_hyperedge_id >= hyperedge_weight.shape[0]:
                    raise ValueError(f"Hyperedge ID out of bounds: {max_hyperedge_id} >= {hyperedge_weight.shape[0]}")
        
        return self.conv(x, hyperedge_index, hyperedge_weight=hyperedge_weight)


class HGNNExpertCoupler(nn.Module):
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.config = config
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for HGNNExpertCoupler but not found.")

        self.hgnn_layers_module = nn.ModuleList()
        current_dim = config.embed_dim
        for _ in range(config.gnn_layers): # Using gnn_layers for number of HGNN layers too
            # TODO: Add config options for use_attention, heads for HypergraphConv if desired
            pyg_wrapper = PyGHypergraphConvWrapper(
                current_dim, 
                config.embed_dim, # Output dim of HGNN layer
                pyg_conv_type=config.hgnn_conv_type
                # Pass other HypergraphConv specific params from config if added
            )
            self.hgnn_layers_module.append(pyg_wrapper)
            current_dim = pyg_wrapper.out_channels_final # Update current_dim for next layer if using attention with concat

        self.combiner = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim), # Assumes HGNN output is embed_dim
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )

        self._hyperedge_index = None
        self._num_hyperedges = 0
        self.generate_static_hyperedges()

        if config.hgnn_learnable_edge_weights and self._num_hyperedges > 0:
            self.hyperedge_weights = nn.Parameter(torch.randn(self._num_hyperedges))
        else:
            self.hyperedge_weights = None
            if config.hgnn_learnable_edge_weights and self._num_hyperedges == 0:
                print("Warning: hgnn_learnable_edge_weights is True, but no hyperedges were generated.")


    def generate_static_hyperedges(self):
        num_experts = self.config.num_experts
        strategy = self.config.static_hyperedge_strategy
        
        node_indices = []
        hyperedge_ids = []
        current_hyperedge_id = 0

        if strategy == "all_pairs":
            if num_experts < 2:
                self._hyperedge_index = torch.empty((2,0), dtype=torch.long)
                self._num_hyperedges = 0
                return
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    node_indices.extend([i, j])
                    hyperedge_ids.extend([current_hyperedge_id, current_hyperedge_id])
                    current_hyperedge_id += 1
        elif strategy == "all_triplets":
            if num_experts < 3:
                self._hyperedge_index = torch.empty((2,0), dtype=torch.long)
                self._num_hyperedges = 0
                return
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    for k in range(j + 1, num_experts):
                        node_indices.extend([i, j, k])
                        hyperedge_ids.extend([current_hyperedge_id, current_hyperedge_id, current_hyperedge_id])
                        current_hyperedge_id += 1
        else:
            raise ValueError(f"Unknown static_hyperedge_strategy: {strategy}")

        if not node_indices: # Handles cases like num_experts=1 for "all_pairs"
             self._hyperedge_index = torch.empty((2,0), dtype=torch.long)
        else:
            self._hyperedge_index = torch.tensor([node_indices, hyperedge_ids], dtype=torch.long)
        self._num_hyperedges = current_hyperedge_id
        # print(f"Generated hyperedge_index for {strategy} with {num_experts} experts: {self._hyperedge_index.shape}, Num Hyperedges: {self._num_hyperedges}")


    def forward(self, expert_outputs: torch.Tensor): # expert_outputs: (B, L, E, D)
        B, L, E, D_embed = expert_outputs.shape
        
        # Device for hyperedge_index and weights
        device = expert_outputs.device
        if self._hyperedge_index is not None:
            hyperedge_index_dev = self._hyperedge_index.to(device)
        else: # Should not happen if generate_static_hyperedges was successful
            return self.combiner(expert_outputs.mean(dim=2))
        
        # If no hyperedges, just return averaged expert features
        if self._num_hyperedges == 0:
            return self.combiner(expert_outputs.mean(dim=2))

        # OPTIMIZED: Use PyG Batch for efficient parallel processing
        # Fixed the hyperedge weight indexing issue properly
        
        current_features_flat = expert_outputs.reshape(B * L, E, D_embed) # (B*L, E, D)
        
        # Create batched PyG data structure
        # All B*L graphs have identical hyperedge structure but different node features
        data_list = []
        for i in range(B * L):
            data = Data(
                x=current_features_flat[i],  # (E, D_embed) - node features for this graph
                edge_index=hyperedge_index_dev  # Same hyperedge structure for all graphs
            )
            data_list.append(data)
        
        # Batch all graphs into single large graph for parallel processing
        batched_data = Batch.from_data_list(data_list)
        
        # Prepare edge weights for batched operation
        edge_weights_for_conv = None
        if self.config.hgnn_learnable_edge_weights and self.hyperedge_weights is not None:
            # PyG batching creates sparse hyperedge IDs - we need weights covering the full ID range
            max_hyperedge_id = batched_data.edge_index[1].max().item()
            weight_array_size = max_hyperedge_id + 1  # Cover full range [0, max_id]
            
            # Create weight array by cycling through original weight parameters
            try:
                all_indices = torch.arange(weight_array_size, device=batched_data.edge_index.device)
                weight_indices = all_indices % self._num_hyperedges
                edge_weights_for_conv = self.hyperedge_weights[weight_indices]
                
                # Validate the mapping worked correctly
                if edge_weights_for_conv.shape[0] != weight_array_size:
                    print(f"Warning: Hyperedge weight mapping failed. Expected {weight_array_size}, got {edge_weights_for_conv.shape[0]}")
                    edge_weights_for_conv = None
                    
            except Exception as e:
                print(f"Warning: Hyperedge weight mapping failed: {e}")
                edge_weights_for_conv = None
        
        # Pass through HGNN layers with batched processing
        h_batched = batched_data.x  # (B*L*E, D_embed)
        for hgnn_layer_instance in self.hgnn_layers_module:
            h_batched = hgnn_layer_instance(
                h_batched, 
                batched_data.edge_index, 
                hyperedge_weight=edge_weights_for_conv
            )
        
        # Reshape output back to (B*L, E, D_embed)
        coordinated_expert_features = h_batched.view(B * L, E, D_embed)
        coordinated_expert_features = coordinated_expert_features.view(B, L, E, D_embed)

        # Combine expert features (averaging across experts)
        coordinated_output = coordinated_expert_features.mean(dim=2) # (B, L, D_embed)
        output = self.combiner(coordinated_output) # (B, L, D_embed)
        return output

    def get_expert_communication_matrices(self):
        # For HGNN, "communication matrix" is less direct than GNN's adjacency.
        # We could return the learnable hyperedge_weights if they exist.
        if self.config.hgnn_learnable_edge_weights and self.hyperedge_weights is not None:
            return [self.hyperedge_weights.detach().cpu().clone()] # Return as a list for consistency
        return []


class ExpertBlock(nn.Module):
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embed_dim, config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout_rate)
        )
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class GNNExpertCoupler(nn.Module): # Original GNN Coupler
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.config = config
        self.gnn_layers_module = nn.ModuleList([
            ExpertGraphConv(config.embed_dim, config.embed_dim, config.num_experts)
            for _ in range(config.gnn_layers)
        ])
        self.combiner = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
    def forward(self, expert_outputs: List[torch.Tensor]): # List of (B, L, D) or stacked (B,L,E,D)
        if isinstance(expert_outputs, list):
            stacked_experts = torch.stack(expert_outputs, dim=2) # (B, L, E, D)
        else: # Assuming already stacked
            stacked_experts = expert_outputs

        expert_features = stacked_experts
        for gnn_layer_instance in self.gnn_layers_module:
            new_features = gnn_layer_instance(expert_features) # Expects (B,L,E,D)
            expert_features = new_features + expert_features # Residual connection
        
        coordinated_output = expert_features.mean(dim=2) # (B, L, D)
        output = self.combiner(coordinated_output) # (B, L, D)
        return output

    def get_expert_communication_matrices(self):
        matrices = []
        for gnn_layer_instance in self.gnn_layers_module:
            matrices.append(gnn_layer_instance.get_adjacency_matrix())
        return matrices

class GNNMoELayer(nn.Module): # This layer can now use GNN or HGNN based on config
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([ExpertBlock(config) for _ in range(config.num_experts)])
        
        if config.coupler_type == "HGNN":
            if not PYG_AVAILABLE:
                raise ImportError("PyTorch Geometric is required for HGNN coupler_type but not found.")
            self.coupler = HGNNExpertCoupler(config)
            print(f"Using HGNNExpertCoupler for MoE layer with {config.static_hyperedge_strategy} strategy.")
        elif config.coupler_type == "GNN":
            self.coupler = GNNExpertCoupler(config)
            print("Using GNNExpertCoupler for MoE layer.")
        else:
            raise ValueError(f"Unknown coupler_type: {config.coupler_type}")

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        expert_outputs_tensors = [expert(x, causal_mask, key_padding_mask) for expert in self.experts]
        # Stack expert outputs: (B, L, E, D)
        stacked_expert_outputs = torch.stack(expert_outputs_tensors, dim=2)
        
        coordinated = self.coupler(stacked_expert_outputs) # Coupler expects (B,L,E,D)
        return x + coordinated # Additive skip connection over the whole MoE block

    def get_communication_data(self): # Generic method
        if hasattr(self.coupler, 'get_expert_communication_matrices'):
            return self.coupler.get_expert_communication_matrices()
        return []


class GNNMoEModel(nn.Module):
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate) 
        self.model_layers = nn.ModuleList([GNNMoELayer(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, input_ids, attention_mask=None, return_loss=True, labels=None):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        causal_mask = self.create_causal_mask(L, input_ids.device)
        # PyTorch MultiheadAttention expects key_padding_mask where True means "ignore"
        # Our attention_mask is 1 for valid, 0 for pad. So invert it.
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        
        for layer_instance in self.model_layers:
            x = layer_instance(x, causal_mask, key_padding_mask)
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if return_loss and labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            # Use ignore_index=0 because prepare_batch in training script sets padding to 0
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0) 
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

    def analyze_expert_communication(self): # Renamed from get_communication_data for consistency
        comm_data = {}
        for i, layer_instance in enumerate(self.model_layers):
            if hasattr(layer_instance, 'get_communication_data'): 
                 data = layer_instance.get_communication_data()
                 if data: # Only add if data is not empty
                    comm_data[f'layer_{i}'] = data
        return comm_data

if __name__ == '__main__':
    from dataclasses import dataclass
    from gnn_moe_config import GNNMoEConfig
    # Example of creating a model instance (requires GNNMoEConfig to be defined/imported)
    
    @dataclass
    class DummyConfigForArchTest(GNNMoEConfig): # Inherit to get new fields
        # Override for testing if needed
        coupler_type: str = "GNN" # Test GNN first
        # coupler_type: str = "HGNN" # Then test HGNN
        # static_hyperedge_strategy: str = "all_pairs"
        # hgnn_learnable_edge_weights: bool = True

    test_cfg_gnn = DummyConfigForArchTest(num_experts=2, embed_dim=32, num_layers=1, gnn_layers=1, coupler_type="GNN")
    print("\nTesting GNN-MoE model instantiation with dummy config:", test_cfg_gnn)
    model_gnn = GNNMoEModel(test_cfg_gnn)
    print("GNN Model instance created successfully.")
    
    bs, sl = 2, 16 # Smaller sequence length for faster test
    dummy_input_ids = torch.randint(0, test_cfg_gnn.vocab_size, (bs, sl))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    
    print(f"\nTesting GNN forward pass with input shape: {dummy_input_ids.shape}")
    try:
        output_gnn = model_gnn(dummy_input_ids, attention_mask=dummy_attention_mask, return_loss=False)
        print(f"GNN Forward pass successful. Output logits shape: {output_gnn['logits'].shape}")
        output_gnn_loss = model_gnn(dummy_input_ids, attention_mask=dummy_attention_mask, return_loss=True, labels=dummy_input_ids)
        print(f"GNN Forward pass with loss successful. Loss: {output_gnn_loss['loss'].item()}")
        comm_data_gnn = model_gnn.analyze_expert_communication()
        print("GNN Expert communication analysis ran. Data:", comm_data_gnn)
    except Exception as e:
        print(f"Error during GNN model test: {e}")

    if PYG_AVAILABLE:
        test_cfg_hgnn = DummyConfigForArchTest(num_experts=2, embed_dim=32, num_layers=1, gnn_layers=1, 
                                               coupler_type="HGNN", static_hyperedge_strategy="all_pairs",
                                               hgnn_learnable_edge_weights=True)
        print("\nTesting HGNN-MoE model instantiation with dummy config:", test_cfg_hgnn)
        try:
            model_hgnn = GNNMoEModel(test_cfg_hgnn)
            print("HGNN Model instance created successfully.")
            print(f"Testing HGNN forward pass with input shape: {dummy_input_ids.shape}")
            output_hgnn = model_hgnn(dummy_input_ids, attention_mask=dummy_attention_mask, return_loss=False)
            print(f"HGNN Forward pass successful. Output logits shape: {output_hgnn['logits'].shape}")
            output_hgnn_loss = model_hgnn(dummy_input_ids, attention_mask=dummy_attention_mask, return_loss=True, labels=dummy_input_ids)
            print(f"HGNN Forward pass with loss successful. Loss: {output_hgnn_loss['loss'].item()}")
            comm_data_hgnn = model_hgnn.analyze_expert_communication()
            print("HGNN Expert communication analysis ran. Data:", comm_data_hgnn)
        except Exception as e:
            print(f"Error during HGNN model test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping HGNN model test as PyTorch Geometric is not available.")
