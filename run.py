#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py

Unified entry point for training and evaluating all MoE architectures.
"""

import torch
import os
import argparse
import numpy as np
import random
import json
import warnings
from dataclasses import fields

# --- Suppress Warnings ---
# Suppress NotOpenSSLWarning
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.config import MoEConfig, HGNNParams, GhostParams, GeometricTrainingConfig
from core.architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from core.training import load_checkpoint, train_model
from core.data import load_data
from core.analysis import run_analysis

_VERBOSE = True
def verbose_print(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

def setup_environment(config: MoEConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    return device

def get_args():
    parser = argparse.ArgumentParser(description="Unified MoE Training Script")
    parser.add_argument('--quiet', action='store_true', help="Suppress most print statements for cleaner sweep logs.")
    
    # --- MoEConfig Base Arguments ---
    parser.add_argument('--run_name', type=str, help="Name of the run for checkpointing.")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, help="Path to checkpoint to resume from.")
    parser.add_argument('--seed', type=int, help="Random seed.")
    parser.add_argument('--architecture_mode', type=str, choices=['gnn', 'hgnn', 'orthogonal', 'ghost'], help="Selects the model architecture.")
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--num_experts', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--use_hypergraph_coupling', action='store_true')
    parser.add_argument('--use_orthogonal_loss', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--max_batches_per_epoch', type=int)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_config_name', type=str)
    parser.add_argument('--num_train_samples', type=int)
    parser.add_argument('--num_eval_samples', type=int)
    parser.add_argument('--num_workers_dataloader', type=int)
    parser.add_argument('--dataset_source', type=str, choices=['huggingface', 'local_file'], help="Source type for dataset.")
    parser.add_argument('--training_mode', type=str, choices=['standard', 'geometric'], help="Training paradigm to use.")

    # --- Geometric Training Arguments ---
    parser.add_argument('--geometric_enabled', action='store_true', help="Enable geometric training.")
    parser.add_argument('--geometric_learning_rate', type=float, help="Learning rate for rotation parameters.")
    parser.add_argument('--geometric_expert_learning_rate', type=float, help="Learning rate for expert parameters.")
    parser.add_argument('--geometric_rotation_dimensions', type=int, help="Number of rotation dimensions.")
    parser.add_argument('--geometric_orthogonality_weight', type=float, help="Weight for orthogonality loss.")
    parser.add_argument('--geometric_rotation_efficiency_weight', type=float, help="Weight for rotation efficiency loss.")
    parser.add_argument('--geometric_specialization_weight', type=float, help="Weight for specialization loss.")
    parser.add_argument('--geometric_lambda_cognitive_rotations', action='store_true', help="Enable lambda calculus cognitive rotations.")

    # --- HGNNParams Arguments ---
    parser.add_argument('--hgnn_num_layers', type=int)
    parser.add_argument('--hgnn_strategy', type=str)
    parser.add_argument('--hgnn_learnable_edge_weights', action='store_true')

    # --- GhostParams Arguments ---
    parser.add_argument('--ghost_num_ghost_experts', type=int)
    parser.add_argument('--ghost_ghost_activation_threshold', type=float)
    parser.add_argument('--ghost_ghost_learning_rate', type=float)
    parser.add_argument('--ghost_ghost_activation_schedule', type=str)
    parser.add_argument('--ghost_saturation_monitoring_window', type=int)
    parser.add_argument('--ghost_ghost_lr_coupling', type=str)
    parser.add_argument('--ghost_ghost_background_learning', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Sanitize dataset name - strip quotes and whitespace for robust handling
    if args.dataset_name:
        args.dataset_name = args.dataset_name.strip().strip("'\"")

    if args.quiet:
        _VERBOSE = False

    # Create config from args, filtering out None values so defaults are used
    config_args = {k: v for k, v in vars(args).items() if v is not None and k != 'quiet'}
    
    # Separate nested config args
    hgnn_args = {k.replace('hgnn_', '', 1): v for k, v in config_args.items() if k.startswith('hgnn_')}
    ghost_args = {k.replace('ghost_', '', 1): v for k, v in config_args.items() if k.startswith('ghost_')}
    # Map command-line arguments to GeometricTrainingConfig parameter names
    geometric_arg_mapping = {
        'geometric_enabled': 'enabled',
        'geometric_learning_rate': 'geometric_learning_rate',
        'geometric_expert_learning_rate': 'expert_learning_rate',
        'geometric_rotation_dimensions': 'rotation_dimensions',
        'geometric_orthogonality_weight': 'orthogonality_weight',
        'geometric_rotation_efficiency_weight': 'rotation_efficiency_weight',
        'geometric_specialization_weight': 'specialization_weight',
        'geometric_lambda_cognitive_rotations': 'lambda_cognitive_rotations'
    }
    
    geometric_args = {}
    for cli_arg, config_param in geometric_arg_mapping.items():
        if cli_arg in config_args:
            geometric_args[config_param] = config_args[cli_arg]
    
    # Remove nested args from main config dict
    for k in list(config_args.keys()):
        if k.startswith('hgnn_') or k.startswith('ghost_') or k.startswith('geometric_'):
            config_args.pop(k)

    # Create the final config object
    cfg = MoEConfig(**config_args)
    if hgnn_args:
        cfg.hgnn = HGNNParams(**hgnn_args)
    if ghost_args:
        cfg.ghost = GhostParams(**ghost_args)
    if geometric_args:
        cfg.geometric = GeometricTrainingConfig(**geometric_args)

    # --- Setup ---
    if cfg.run_name:
        cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name)
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    verbose_print(f"📁 Checkpoint directory: {cfg.checkpoint_dir}")

    selected_device = setup_environment(cfg)
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg)
    
    # --- Model Initialization ---
    model = MoEModel(cfg).to(selected_device)
    
    # Log model size and save to config
    num_params = model.get_total_params()
    cfg.num_parameters = num_params
    verbose_print(f"✅ Model initialized with {num_params/1e6:.2f}M parameters.")

    actual_batches_per_epoch = len(train_loader) if cfg.max_batches_per_epoch == -1 else min(len(train_loader), cfg.max_batches_per_epoch)
    if cfg.max_steps is None:
        cfg.max_steps = cfg.epochs * actual_batches_per_epoch
    
    optimizer = create_dynamic_optimizer(model, cfg)
    scheduler = PrimaryGhostLRScheduler(cfg, optimizer)

    # --- Checkpoint Loading ---
    start_epoch, current_step, best_eval_loss_resumed = 0, 0, float('inf')
    if cfg.resume_checkpoint and os.path.isfile(cfg.resume_checkpoint):
        start_epoch, current_step, best_eval_loss_resumed = load_checkpoint(
            cfg.resume_checkpoint, model, optimizer, scheduler
        )
        model.to(selected_device)
        verbose_print(f"✅ Resumed from checkpoint: {cfg.resume_checkpoint}")
        verbose_print(f"   - Resuming from Epoch {start_epoch}, Step {current_step}")

    # --- Training ---
    training_stats, final_best_loss = train_model(
        model, optimizer, scheduler, train_loader, eval_loader, selected_device, cfg, 
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    # --- Analysis ---
    stats_file_path = os.path.join(cfg.checkpoint_dir, "training_log.json")
    if training_stats:
        verbose_print(f"📝 Detailed training log saved to {stats_file_path}")
        run_analysis(stats_file_path)
    else:
        verbose_print("⚠️ No training stats generated, skipping log save and analysis.")
