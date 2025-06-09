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
from typing import List # Used in GNNExpertCoupler forward, though not strictly needed for type hint only

# Assuming GNNMoEConfig will be imported from gnn_moe_config.py in the main script
# For standalone use or direct import, you might need:
# from gnn_moe_config import GNNMoEConfig 
# However, to keep this module self-contained for extraction,
# we'll assume GNNMoEConfig is passed appropriately or defined elsewhere when this is used.
# For direct use of this file, GNNMoEConfig would need to be defined or imported.
# We will import it in the main script that uses these classes.

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
        flat_features = expert_features.view(-1, num_experts, embed_dim)
        updated_features_list = []
        for expert_idx in range(num_experts):
            current_expert = flat_features[:, expert_idx, :]
            messages = []
            for other_idx in range(num_experts):
                if other_idx != expert_idx:
                    other_expert = flat_features[:, other_idx, :]
                    concat_features = torch.cat([current_expert, other_expert], dim=1)
                    content_weight = torch.sigmoid(self.message_weight(concat_features).squeeze(-1))
                    message_strength = adjacency[expert_idx, other_idx] * content_weight
                    weighted_message = other_expert * message_strength.unsqueeze(1)
                    messages.append(weighted_message)
            if messages:
                neighbor_msg = torch.stack(messages, dim=0).sum(dim=0)
                neighbor_out = self.neighbor_transform(neighbor_msg)
            else:
                neighbor_out = torch.zeros_like(self.neighbor_transform(current_expert))
            self_out = self.self_transform(current_expert)
            updated_expert = F.gelu(neighbor_out + self_out)
            updated_features_list.append(updated_expert)
        updated_stack = torch.stack(updated_features_list, dim=1)
        return updated_stack.view(batch_size, seq_len, num_experts, embed_dim)

    def get_adjacency_matrix(self):
        return torch.sigmoid(self.adjacency_logits).detach()

class ExpertBlock(nn.Module):
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embed_dim, config.num_heads,
            dropout=config.dropout_rate, # Assuming GNNMoEConfig uses dropout_rate
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate), # Assuming GNNMoEConfig uses dropout_rate
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout_rate) # Assuming GNNMoEConfig uses dropout_rate
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

class GNNExpertCoupler(nn.Module):
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
    def forward(self, expert_outputs: List[torch.Tensor]): # Explicit type hint
        stacked_experts = torch.stack(expert_outputs, dim=2)
        expert_features = stacked_experts
        for gnn_layer_instance in self.gnn_layers_module:
            new_features = gnn_layer_instance(expert_features)
            expert_features = new_features + expert_features # Residual connection
        coordinated_output = expert_features.mean(dim=2)
        output = self.combiner(coordinated_output)
        return output
    def get_expert_communication_matrices(self):
        matrices = []
        for gnn_layer_instance in self.gnn_layers_module:
            matrices.append(gnn_layer_instance.get_adjacency_matrix())
        return matrices

class GNNMoELayer(nn.Module):
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([ExpertBlock(config) for _ in range(config.num_experts)])
        self.gnn_coupler = GNNExpertCoupler(config)
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        expert_outputs = [expert(x, causal_mask, key_padding_mask) for expert in self.experts]
        coordinated = self.gnn_coupler(expert_outputs)
        return x + coordinated

class GNNMoEModel(nn.Module):
    def __init__(self, config): # config will be GNNMoEConfig instance
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        # Ensure GNNMoEConfig uses 'dropout_rate' if 'dropout' was specific to old script's config
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
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        for layer_instance in self.model_layers:
            x = layer_instance(x, causal_mask, key_padding_mask)
        x = self.output_norm(x)
        logits = self.lm_head(x)
        if return_loss and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

    def analyze_expert_communication(self):
        comm_data = {}
        for i, layer_instance in enumerate(self.model_layers):
            if hasattr(layer_instance, 'gnn_coupler'): # Check if it's a GNNMoELayer
                 matrices = layer_instance.gnn_coupler.get_expert_communication_matrices()
                 comm_data[f'layer_{i}'] = matrices
        return comm_data

if __name__ == '__main__':
    # Example of creating a model instance (requires GNNMoEConfig to be defined/imported)
    # This part is for testing the module itself if run directly.
    # from gnn_moe_config import GNNMoEConfig # Would be needed here
    
    # Dummy config for testing
    @dataclass
    class DummyConfig:
        vocab_size: int = 50257; max_seq_length: int = 32; embed_dim: int = 64
        num_layers: int = 2; num_heads: int = 2; dropout_rate: float = 0.1
        num_experts: int = 2; gnn_layers: int = 1
    
    test_cfg = DummyConfig()
    print("Testing model instantiation with dummy config:", test_cfg)
    model = GNNMoEModel(test_cfg)
    print("Model instance created successfully.")
    
    # Test forward pass
    bs, sl = 2, test_cfg.max_seq_length
    dummy_input_ids = torch.randint(0, test_cfg.vocab_size, (bs, sl))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    print(f"Testing forward pass with input shape: {dummy_input_ids.shape}")
    
    try:
        output = model(dummy_input_ids, attention_mask=dummy_attention_mask, return_loss=False)
        print(f"Forward pass successful. Output logits shape: {output['logits'].shape}")
        
        # Test with loss
        output_with_loss = model(dummy_input_ids, attention_mask=dummy_attention_mask, return_loss=True, labels=dummy_input_ids)
        print(f"Forward pass with loss successful. Loss: {output_with_loss['loss'].item()}")

        # Test communication analysis
        comm_data = model.analyze_expert_communication()
        print("Expert communication analysis ran. Data:", comm_data)

    except Exception as e:
        print(f"Error during model test: {e}")
