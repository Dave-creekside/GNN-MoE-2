#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Unified configuration for all MoE model architectures.
"""

from dataclasses import dataclass, field, fields
from typing import Optional, List

@dataclass
class HGNNParams:
    """Parameters specific to Hypergraph-based coupling."""
    num_layers: int = 2
    strategy: str = "all_pairs"  # "all_pairs", "all_triplets", "all"
    learnable_edge_weights: bool = True

@dataclass
class GhostParams:
    """Parameters specific to the Ghost Expert architecture."""
    num_ghost_experts: int = 0  # Defaulting to 0 disables the Ghost mechanism
    ghost_activation_threshold: float = 0.01
    ghost_learning_rate: float = 1e-4
    ghost_activation_schedule: str = "gradual"  # "gradual", "binary", "selective"
    saturation_monitoring_window: int = 100
    ghost_lr_coupling: str = "inverse"  # "inverse", "complementary"
    ghost_background_learning: bool = False

@dataclass
class GeometricTrainingConfig:
    """Parameters specific to Geometric Constrained Learning."""
    # Core geometric parameters
    enabled: bool = False
    geometric_learning_rate: float = 1e-3
    expert_learning_rate: float = 1e-4  # Lower than geometric LR
    rotation_dimensions: int = 4
    
    # Loss weights
    orthogonality_weight: float = 0.5
    rotation_efficiency_weight: float = 0.2
    specialization_weight: float = 0.3
    
    # Ghost geometric integration
    ghost_geometric_threshold: float = 0.7
    ghost_rotation_dimensions: int = 4
    
    # Lambda calculus specific
    lambda_cognitive_rotations: bool = True
    lambda_rotation_scheduling: str = "curriculum"  # "curriculum", "adaptive", "fixed"
    
    # Advanced geometric parameters
    rotation_matrix_type: str = "givens"  # "givens", "householder", "learned"
    rotation_convergence_threshold: float = 1e-4
    max_rotation_magnitude: float = 2.0  # Maximum rotation angle in radians

@dataclass
class MoEConfig:
    """Unified configuration for all MoE models."""
    # --- Run Management ---
    run_name: str = "moe_run"
    checkpoint_dir: str = "checkpoints"
    resume_checkpoint: Optional[str] = None
    seed: int = 42

    # --- Architecture Selection ---
    # Controls which features are active.
    # 'gnn': Basic GNN coupling.
    # 'hgnn': Hypergraph coupling.
    # 'orthogonal': HGNN with orthogonality constraints.
    # 'ghost': All features, including ghost experts.
    architecture_mode: str = "ghost"

    # --- Core Model Parameters ---
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_length: int = 512
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_experts: int = 4
    dropout_rate: float = 0.1
    num_parameters: Optional[int] = None # To be calculated at runtime

    # --- Feature Flags for Backwards Compatibility ---
    use_hypergraph_coupling: bool = True
    use_orthogonal_loss: bool = True

    # --- Nested Configurations for Specific Features ---
    hgnn: HGNNParams = field(default_factory=HGNNParams)
    ghost: GhostParams = field(default_factory=GhostParams)
    geometric: GeometricTrainingConfig = field(default_factory=GeometricTrainingConfig)

    # --- Training Parameters ---
    training_mode: str = "standard"  # "standard" or "geometric"
    training_loop: str = "standard"
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_batches_per_epoch: int = -1  # -1 for all batches
    eval_every: int = 100
    log_every: int = 25  # New parameter for frequent logging
    max_steps: Optional[int] = None # Calculated at runtime if not set

    # --- Dataset Parameters ---
    dataset_source: str = "huggingface" # or "local_file"
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-v1"
    num_train_samples: int = 2000
    num_eval_samples: int = 400
    num_workers_dataloader: int = 4

    # --- Memory Optimization ---
    use_mixed_precision: bool = True  # Enable mixed precision for geometric rotations

    def __post_init__(self):
        """Set feature flags based on architecture_mode for easy configuration."""
        if self.architecture_mode == 'gnn':
            self.use_hypergraph_coupling = False
            self.use_orthogonal_loss = False
            self.ghost.num_ghost_experts = 0
        elif self.architecture_mode == 'hgnn':
            self.use_hypergraph_coupling = True
            self.use_orthogonal_loss = False
            self.ghost.num_ghost_experts = 0
        elif self.architecture_mode == 'orthogonal':
            self.use_hypergraph_coupling = True
            self.use_orthogonal_loss = True
            self.ghost.num_ghost_experts = 0
        elif self.architecture_mode == 'ghost':
            self.use_hypergraph_coupling = True
            self.use_orthogonal_loss = True
            # num_ghost_experts is configured by the user
        
        # Adjust num_heads if embed_dim is not divisible
        if self.embed_dim % self.num_heads != 0:
            # Find the largest factor of embed_dim <= original num_heads
            for i in range(self.num_heads, 0, -1):
                if self.embed_dim % i == 0:
                    print(f"Adjusting num_heads from {self.num_heads} to {i} based on embed_dim {self.embed_dim}")
                    self.num_heads = i
                    break

    def to_dict(self):
        """Converts the dataclass to a serializable dict."""
        d = field_to_dict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        """Creates a config object from a dictionary, handling nested dataclasses."""
        hgnn_params_dict = d.pop('hgnn', {})
        ghost_params_dict = d.pop('ghost', {})
        geometric_params_dict = d.pop('geometric', {})

        # Create the main config, ignoring nested dataclass fields for now
        main_config_fields = {f.name for f in fields(cls)}
        main_config_args = {k: v for k, v in d.items() if k in main_config_fields}
        
        config = cls(**main_config_args)
        
        # Create and assign the nested dataclasses
        config.hgnn = HGNNParams(**hgnn_params_dict)
        config.ghost = GhostParams(**ghost_params_dict)
        config.geometric = GeometricTrainingConfig(**geometric_params_dict)
        
        return config

def field_to_dict(obj):
    """Recursively convert dataclass to dict."""
    if hasattr(obj, "__dict__"):
        return {k: field_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [field_to_dict(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(field_to_dict(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: field_to_dict(v) for k, v in obj.items()}
    else:
        return obj
