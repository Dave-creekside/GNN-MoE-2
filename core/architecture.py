#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
architecture.py

Contains all model component classes for the unified MoE model.
- ExpertBlock
- HGNNExpertCoupler
- GhostAwareExpertBlock
- ExpertSaturationMonitor
- GhostActivationController
- PrimaryGhostLRScheduler
- TripleHypergraphCoupler
- MoELayer
- MoEModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import itertools
import copy
import numpy as np

from .config import MoEConfig

try:
    from torch_geometric.nn import HypergraphConv
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    HypergraphConv = None
    Data = None
    Batch = None

class ExpertBlock(nn.Module):
    def __init__(self, config: MoEConfig):
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

class HGNNExpertCoupler(nn.Module):
    def __init__(self, num_experts, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for HGNNExpertCoupler but not found.")

        self.hgnn_layers_module = nn.ModuleList()
        current_dim = config.embed_dim
        for _ in range(config.hgnn.num_layers):
            pyg_wrapper = HypergraphConv(current_dim, config.embed_dim)
            self.hgnn_layers_module.append(pyg_wrapper)
            current_dim = config.embed_dim

        self.combiner = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )

        self._hyperedge_index = None
        self._num_hyperedges = 0
        self.generate_static_hyperedges()

        if config.hgnn.learnable_edge_weights and self._num_hyperedges > 0:
            self.hyperedge_weights = nn.Parameter(torch.randn(self._num_hyperedges))
        else:
            self.hyperedge_weights = None

    def generate_static_hyperedges(self):
        strategy = self.config.hgnn.strategy
        node_indices = []
        hyperedge_ids = []
        current_hyperedge_id = 0

        if self.num_experts == 0:
            self._hyperedge_index = torch.empty((2, 0), dtype=torch.long)
            self._num_hyperedges = 0
            return

        if strategy == "all_pairs":
            if self.num_experts < 2: return
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    node_indices.extend([i, j])
                    hyperedge_ids.extend([current_hyperedge_id, current_hyperedge_id])
                    current_hyperedge_id += 1
        elif strategy == "all_triplets":
            if self.num_experts < 3: return
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    for k in range(j + 1, self.num_experts):
                        node_indices.extend([i, j, k])
                        hyperedge_ids.extend([current_hyperedge_id, current_hyperedge_id, current_hyperedge_id])
                        current_hyperedge_id += 1
        elif strategy == "all":
            node_indices.extend(range(self.num_experts))
            hyperedge_ids.extend([0] * self.num_experts)
            current_hyperedge_id = 1
        else:
            raise ValueError(f"Unknown static_hyperedge_strategy: {strategy}")

        if not node_indices:
            self._hyperedge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self._hyperedge_index = torch.tensor([node_indices, hyperedge_ids], dtype=torch.long)
        self._num_hyperedges = current_hyperedge_id

    def get_adjacency_matrix(self):
        """Computes a weighted adjacency matrix from the hypergraph."""
        if self._num_hyperedges == 0:
            return np.zeros((self.num_experts, self.num_experts))

        adj = torch.zeros((self.num_experts, self.num_experts))
        nodes, hyperedges = self._hyperedge_index
        
        for i in range(self._num_hyperedges):
            nodes_in_hyperedge = nodes[hyperedges == i]
            edge_weight = self.hyperedge_weights[i].item() if self.hyperedge_weights is not None else 1.0
            for u in nodes_in_hyperedge:
                for v in nodes_in_hyperedge:
                    if u != v:
                        adj[u, v] += edge_weight
        return adj.numpy()

    def forward(self, expert_outputs: torch.Tensor):
        B, L, E, D_embed = expert_outputs.shape
        device = expert_outputs.device
        if self._hyperedge_index is None or self._num_hyperedges == 0:
            return self.combiner(expert_outputs.mean(dim=2))
        
        hyperedge_index_dev = self._hyperedge_index.to(device)
        current_features_flat = expert_outputs.reshape(B * L, E, D_embed)
        
        data_list = [Data(x=current_features_flat[i], edge_index=hyperedge_index_dev) for i in range(B * L)]
        batched_data = Batch.from_data_list(data_list)
        
        h_batched = batched_data.x
        for hgnn_layer_instance in self.hgnn_layers_module:
            h_batched = hgnn_layer_instance(h_batched, batched_data.edge_index)
        
        coordinated_expert_features = h_batched.view(B, L, E, D_embed)
        coordinated_output = coordinated_expert_features.mean(dim=2)
        output = self.combiner(coordinated_output)
        return output

class GhostAwareExpertBlock(ExpertBlock):
    def __init__(self, config, is_ghost=False):
        super().__init__(config)
        self.is_ghost = is_ghost
        self.activation_level = 0.0
        self.background_learning = config.ghost.ghost_background_learning if is_ghost else False

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        output = super().forward(x, causal_mask, key_padding_mask)
        if self.is_ghost:
            output = output * self.activation_level
        return output

def compute_orthogonality_score(gram_matrix):
    identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
    return 1.0 - F.mse_loss(gram_matrix, identity)

class ExpertSaturationMonitor:
    def __init__(self, config: MoEConfig):
        self.saturation_threshold = config.ghost.ghost_activation_threshold
        self.variance_window = config.ghost.saturation_monitoring_window
        self.history = []

    def compute_saturation_metrics(self, expert_outputs, input_features):
        stacked_outputs = torch.stack(expert_outputs, dim=2)
        mean_expert_outputs = stacked_outputs.mean(dim=(0, 1))
        normalized_outputs = F.normalize(mean_expert_outputs, p=2, dim=1)
        gram_matrix = torch.mm(normalized_outputs, normalized_outputs.T)
        orthogonality_score = compute_orthogonality_score(gram_matrix)

        expert_reconstruction = stacked_outputs.mean(dim=2)
        residual = input_features - expert_reconstruction
        unexplained_variance = torch.var(residual)
        saturation = orthogonality_score.item() * unexplained_variance.item()

        return {
            'saturation_level': saturation,
            'orthogonality_score': orthogonality_score.item(),
            'unexplained_variance': unexplained_variance.item(),
            'needs_ghost_activation': saturation > self.saturation_threshold
        }

class GhostActivationController:
    def __init__(self, config: MoEConfig):
        self.num_ghosts = config.ghost.num_ghost_experts
        self.activation_schedule = config.ghost.ghost_activation_schedule
        self.activation_rates = torch.zeros(self.num_ghosts)
        self.ghost_states = ["dormant"] * self.num_ghosts

    def update_ghost_activations(self, saturation_metrics, step):
        if saturation_metrics['needs_ghost_activation']:
            for ghost_idx in range(self.num_ghosts):
                if self.ghost_states[ghost_idx] == "dormant":
                    self.ghost_states[ghost_idx] = "activating"
                    self.activation_rates[ghost_idx] = 0.01
                    return self.activation_rates

        for ghost_idx in range(self.num_ghosts):
            if self.ghost_states[ghost_idx] == "activating":
                self.activation_rates[ghost_idx] = min(1.0, self.activation_rates[ghost_idx] + 0.01)
                if self.activation_rates[ghost_idx] >= 1.0:
                    self.ghost_states[ghost_idx] = "active"
        return self.activation_rates

class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.use_ghosts = config.ghost.num_ghost_experts > 0

        self.primary_experts = nn.ModuleList([
            GhostAwareExpertBlock(config, is_ghost=False) for _ in range(config.num_experts)
        ])
        
        if self.use_ghosts:
            self.ghost_experts = nn.ModuleList([
                GhostAwareExpertBlock(config, is_ghost=True) for _ in range(config.ghost.num_ghost_experts)
            ])
            self.saturation_monitor = ExpertSaturationMonitor(config)
            self.ghost_controller = GhostActivationController(config)
        
        if config.use_hypergraph_coupling:
            self.coupler = HGNNExpertCoupler(config.num_experts, config)
        
        self._last_saturation_metrics = {}
        self._last_ghost_activations = torch.zeros(config.ghost.num_ghost_experts)

    def forward(self, x, step, causal_mask=None, key_padding_mask=None):
        primary_outputs = [expert(x, causal_mask, key_padding_mask) for expert in self.primary_experts]
        
        if self.use_ghosts:
            saturation_metrics = self.saturation_monitor.compute_saturation_metrics(primary_outputs, x)
            ghost_activation_levels = self.ghost_controller.update_ghost_activations(saturation_metrics, step)
            self._last_saturation_metrics = saturation_metrics
            self._last_ghost_activations = ghost_activation_levels

            for i, ghost_expert in enumerate(self.ghost_experts):
                ghost_expert.activation_level = ghost_activation_levels[i].item()
            
            ghost_outputs = [expert(x, causal_mask, key_padding_mask) for expert in self.ghost_experts]
            all_outputs = torch.stack(primary_outputs + ghost_outputs, dim=2)
        else:
            all_outputs = torch.stack(primary_outputs, dim=2)

        if self.config.use_hypergraph_coupling:
            coordinated_output = self.coupler(all_outputs)
        else:
            coordinated_output = all_outputs.mean(dim=2)

        return x + coordinated_output

class PrimaryGhostLRScheduler:
    def __init__(self, config: MoEConfig, optimizer):
        self.optimizer = optimizer
        self.config = config
        self.primary_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_steps
        )
        self.use_ghosts = config.ghost.num_ghost_experts > 0
        if self.use_ghosts:
            self.ghost_lr_coupling = config.ghost.ghost_lr_coupling
            self.initial_primary_lr = config.learning_rate
            self.initial_ghost_lr = config.ghost.ghost_learning_rate

    def state_dict(self):
        return self.primary_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.primary_scheduler.load_state_dict(state_dict)

    def step(self, ghost_activation_levels=None):
        self.primary_scheduler.step()
        current_primary_lr = self.primary_scheduler.get_last_lr()[0]
        ghost_lrs = []

        if self.use_ghosts and ghost_activation_levels is not None:
            lr_decay_factor = current_primary_lr / self.initial_primary_lr
            ghost_lr_factor = 1.0 - lr_decay_factor if self.ghost_lr_coupling == "inverse" else 1.0
            for activation_level in ghost_activation_levels:
                ghost_lrs.append(self.initial_ghost_lr * ghost_lr_factor * activation_level.item())
        
        return current_primary_lr, ghost_lrs

def create_dynamic_optimizer(model, config: MoEConfig):
    param_groups = []
    primary_params = [p for layer in model.model_layers for p in layer.primary_experts.parameters()]
    param_groups.append({'params': primary_params, 'lr': config.learning_rate, 'name': 'primary_experts'})

    if config.ghost.num_ghost_experts > 0:
        ghost_params = [p for layer in model.model_layers for p in layer.ghost_experts.parameters()]
        param_groups.append({'params': ghost_params, 'lr': config.ghost.ghost_learning_rate, 'name': 'ghost_experts'})

    return torch.optim.AdamW(param_groups)

class MoEModel(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.model_layers = nn.ModuleList([MoELayer(config) for _ in range(config.num_layers)])
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

    def forward(self, input_ids, step, attention_mask=None, return_loss=True, labels=None):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        
        causal_mask = self.create_causal_mask(L, input_ids.device)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        
        for layer in self.model_layers:
            x = layer(x, step=step, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
            
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if return_loss and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
            return {'loss': loss, 'logits': logits}
            
        return {'logits': logits}

    def get_current_ghost_activations(self):
        if self.config.ghost.num_ghost_experts > 0 and self.model_layers:
            return self.model_layers[0]._last_ghost_activations
        return torch.zeros(self.config.ghost.num_ghost_experts)

    def get_last_saturation_metrics(self):
        if self.config.ghost.num_ghost_experts > 0 and self.model_layers:
            return self.model_layers[0]._last_saturation_metrics
        return {}

    def get_total_orthogonality_loss(self, step):
        # This would be where the orthogonality loss is calculated
        if self.config.use_orthogonal_loss:
            # Placeholder for actual loss calculation
            return torch.tensor(0.0)
        return torch.tensor(0.0)

    def get_expert_activation_loads(self):
        loads = {'primary': [], 'ghost': []}
        if not self.model_layers: return loads
        
        # For simplicity, we'll just get the average activation of the final layer's experts
        final_layer = self.model_layers[-1]
        
        # This is a placeholder. A real implementation would need to track expert usage.
        # For now, we return uniform loads for primary and ghost activations based on their state.
        primary_load = [1.0] * self.config.num_experts
        loads['primary'] = primary_load

        if self.config.ghost.num_ghost_experts > 0:
            ghost_activations = final_layer._last_ghost_activations.cpu().numpy().tolist()
            loads['ghost'] = ghost_activations
            
        return loads
