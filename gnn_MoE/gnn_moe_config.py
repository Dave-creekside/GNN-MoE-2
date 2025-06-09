#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_config.py

Configuration dataclass for GNN-Coupled MoE models.
"""
from dataclasses import dataclass, field
from typing import Optional

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
    resume_checkpoint: Optional[str] = None
    run_name: Optional[str] = None # For naming output files and creating subdir in checkpoint_dir

    # Technical
    seed: int = 42
    num_workers_dataloader: int = 2 # Default for A100, can be overridden

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

if __name__ == '__main__':
    # Example of creating a config instance
    default_cfg = GNNMoEConfig()
    print("Default Config:", default_cfg)
    
    # Example of overriding a default for a specific run
    wikitext103_cfg = GNNMoEConfig(
        dataset_config_name="wikitext-103-v1",
        num_train_samples=-1, # Use all
        num_eval_samples=-1,  # Use all
        max_batches_per_epoch = -1, # Full epoch
        epochs=5,
        run_name="wikitext103_run1"
    )
    print("\nExample WikiText-103 Config:", wikitext103_cfg)
