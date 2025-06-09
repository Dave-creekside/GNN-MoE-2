#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_config.py

Configuration dataclass for GNN-Coupled MoE models, including Ghost Expert features.
"""
from dataclasses import dataclass, field
from typing import Optional

# Assuming GNNMoEConfig is in a file that can be imported
# For now, I will redefine it here based on the provided file.
# In a real project, this would be `from orthogon.adaptive-orthogonal.gnn_moe_config import GNNMoEConfig`
@dataclass
class GNNMoEConfig:
    # Model Architecture
    vocab_size: int = 50257
    max_seq_length: int = 128
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8 # Often embed_dim // 64
    dropout_rate: float = 0.1 # Renamed from 'dropout' to avoid conflict with nn.Dropout
    num_experts: int = 4
    gnn_layers: int = 2 # GNN layers in the coupler

    # Training Hyperparameters
    batch_size: int = 32
    learning_rate: float = 5e-4
    epochs: int = 8
    max_batches_per_epoch: int = -1 # -1 means full epoch (calculated based on dataset size)
    eval_every: int = 200 # Steps

    # Dataset
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-v1" # Default to wikitext-2-v1
    num_train_samples: int = -1 # -1 for all available from the chosen dataset split
    num_eval_samples: int = -1   # -1 for all available from the chosen dataset split

    # Checkpointing & Output
    checkpoint_dir: str = "checkpoints" # Base directory, run-specific subdir will be created if run_name is provided
    resume_checkpoint: str = None
    run_name: str = None # For naming output files and creating subdir in checkpoint_dir

    # Technical
    seed: int = 42
    num_workers_dataloader: int = 2 # Default for A100, can be overridden

    # HGNN Specific Configuration (defaults for GNN compatibility)
    coupler_type: str = "GNN"  # Can be "GNN" or "HGNN"
    hgnn_conv_type: str = "HypergraphConv" # PyG layer name, e.g., "HypergraphConv"
    static_hyperedge_strategy: str = "all_pairs" # e.g., "all_pairs", "all_triplets"
    hgnn_learnable_edge_weights: bool = True # If True, HGNNExpertCoupler will have learnable weights for hyperedges

    # Orthogonal Expert Training Configuration
    apply_orthogonality_loss: bool = True # Enable/disable orthogonality constraints
    orthogonality_loss_weight: float = 0.1 # λ weight for orthogonality penalty in total loss
    orthogonality_aggregation: str = "mean" # How to aggregate across batch/seq: "mean" or "pool"
    orthogonality_loss_type: str = "gram_identity" # Type of orthogonality loss: "gram_identity", "cosine_similarity"
    orthogonality_warmup_steps: int = 1000 # Steps before orthogonality loss reaches full weight
    track_expert_specialization: bool = True # Enable expert specialization monitoring

    # Weight Matrix Orthogonality (Phase 2.1)
    apply_weight_orthogonality_loss: bool = False # Enable weight matrix orthogonality constraints
    weight_orthogonality_loss_weight: float = 0.05 # λ_w weight for weight matrix orthogonality penalty
    weight_orthogonality_target_layer: str = "ffn_input" # Which layer to constrain: "ffn_input", "ffn_output", "attention", "combined"
    weight_orthogonality_normalization: str = "frobenius" # Norm type: "frobenius" or "spectral"
    combine_weight_output_orthogonality: bool = False # Use both weight and output orthogonality constraints

    # Adaptive Weight Orthogonality (Phase 2.2)
    adaptive_weight_orthogonality: bool = False # Enable adaptive weight orthogonality system
    initial_weight_orthogonality_strength: float = 0.1 # Initial strength for adaptive system
    minimum_weight_orthogonality_strength: float = 0.001 # Minimum adaptive strength
    maximum_weight_orthogonality_strength: float = 0.3 # Maximum adaptive strength
    adaptive_decay_schedule: str = "cosine" # Adaptation schedule: "cosine", "exponential", "linear", "step"
    adaptation_frequency: int = 500 # Steps between adaptive adjustments
    target_specialization_score: float = 0.95 # Target orthogonality level (95%)
    specialization_tolerance: float = 0.02 # ±2% tolerance around target
    layer_specific_adaptation: bool = True # Enable layer-specific adaptive strengths
    deeper_layer_scaling: float = 0.8 # Scaling factor for deeper layers
    performance_aware_adaptation: bool = True # Enable performance-based adaptation
    performance_monitor_window: int = 100 # Steps to monitor for performance changes
    collapse_detection_threshold: float = 0.1 # Expert collapse detection sensitivity
    emergency_constraint_boost: bool = True # Enable emergency intervention
    emergency_boost_multiplier: float = 2.0 # Emergency boost multiplier
    emergency_detection_window: int = 50 # Steps for emergency detection

    def __post_init__(self):
        # Auto-calculate num_heads if embed_dim is a multiple of 64 and num_heads is at its default
        if self.embed_dim % 64 == 0:
             expected_heads = self.embed_dim // 64
             # Check if num_heads is still the default value for the class
             if self.num_heads == GNNMoEConfig.__dataclass_fields__['num_heads'].default:
                print(f"Adjusting num_heads from {self.num_heads} to {expected_heads} based on embed_dim {self.embed_dim}")
                self.num_heads = expected_heads

        if self.embed_dim % self.num_heads != 0:
            # This is a critical issue for MultiheadAttention, so make it a strong warning or even an error
            print(f"CRITICAL WARNING: embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}).")
            # Consider raising ValueError here if strictness is desired:
            # raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}).")


@dataclass
class GhostMoEConfig(GNNMoEConfig):
    # Ghost expert parameters
    num_ghost_experts: int = 4
    ghost_activation_threshold: float = 0.7
    ghost_background_learning: bool = True
    ghost_learning_rate: float = 1e-4

    # Activation dynamics
    ghost_activation_schedule: str = "gradual"  # "gradual", "binary", "selective"
    saturation_monitoring_window: int = 100

    # Learning rate coupling
    ghost_lr_coupling: str = "inverse"  # "inverse", "complementary"

    # Hypergraph configuration
    ghost_hypergraph_strategy: str = "all"  # "primary_only", "ghost_only", "all"
    mixed_coupling_weight: float = 0.33


if __name__ == '__main__':
    # Example of creating a config instance
    ghost_cfg = GhostMoEConfig()
    print("Ghost MoE Config:", ghost_cfg)
