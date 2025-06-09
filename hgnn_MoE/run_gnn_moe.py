#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gnn_moe.py

Main executable script for GNN-Coupled MoE model training and experimentation.
"""

import torch
import torch.optim as optim
import os
import argparse
from dataclasses import fields
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json

# --- Conditional Print Function ---
_VERBOSE = True
def verbose_print(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

# Import from modules
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel
from gnn_moe_data import load_data
from gnn_moe_training import load_checkpoint, train_gnn_moe
from gnn_moe_analysis import plot_training_results, analyze_expert_communication, plot_expert_connectivity, analyze_model_efficiency

def setup_environment(config: GNNMoEConfig):
    plt.style.use('default')
    sns.set_palette("husl")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        verbose_print(f"🚀 Device: CUDA (Available: {torch.cuda.device_count()})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        verbose_print("🚀 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        verbose_print("🚀 Device: CPU")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists("plots"):
        os.makedirs("plots")
        verbose_print("📁 Created 'plots' directory for output visualizations.")
    
    verbose_print(f"✅ Environment ready. Seed: {config.seed}, Device: {device}")
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-MoE Hyperparameter Training Script")
    
    # Use a temporary config instance to get defaults for help messages
    temp_cfg = GNNMoEConfig()

    # Model Architecture Args
    parser.add_argument('--embed_dim', type=int, default=temp_cfg.embed_dim, help=f"Embedding dimension (default: {temp_cfg.embed_dim})")
    parser.add_argument('--num_layers', type=int, default=temp_cfg.num_layers, help=f"Number of GNNMoELayers (default: {temp_cfg.num_layers})")
    parser.add_argument('--num_heads', type=int, default=temp_cfg.num_heads, help=f"Number of attention heads (default: {temp_cfg.num_heads})")
    parser.add_argument('--dropout_rate', type=float, default=temp_cfg.dropout_rate, help=f"Dropout rate (default: {temp_cfg.dropout_rate})")
    parser.add_argument('--num_experts', type=int, default=temp_cfg.num_experts, help=f"Number of experts per layer (default: {temp_cfg.num_experts})")
    parser.add_argument('--gnn_layers', type=int, default=temp_cfg.gnn_layers, help=f"Number of GNN layers in coupler (default: {temp_cfg.gnn_layers})")

    # HGNN Specific Args
    parser.add_argument('--coupler_type', type=str, default=temp_cfg.coupler_type, choices=["GNN", "HGNN"], help=f"Coupler type (default: {temp_cfg.coupler_type})")
    parser.add_argument('--hgnn_conv_type', type=str, default=temp_cfg.hgnn_conv_type, help=f"PyG hypergraph layer type (default: {temp_cfg.hgnn_conv_type})")
    parser.add_argument('--static_hyperedge_strategy', type=str, default=temp_cfg.static_hyperedge_strategy, choices=["all_pairs", "all_triplets"], help=f"Hyperedge strategy (default: {temp_cfg.static_hyperedge_strategy})")
    
    hew_group = parser.add_mutually_exclusive_group()
    hew_group.add_argument('--hgnn_learnable_edge_weights', action='store_true', dest='hgnn_learnable_cli_true', help="Enable learnable hyperedge weights (sets to True).")
    hew_group.add_argument('--no_hgnn_learnable_edge_weights', action='store_false', dest='hgnn_learnable_cli_false', help="Disable learnable hyperedge weights (sets to False).")
    
    # Training Hyperparameters Args
    parser.add_argument('--batch_size', type=int, default=temp_cfg.batch_size, help=f"Batch size (default: {temp_cfg.batch_size})")
    parser.add_argument('--learning_rate', type=float, default=temp_cfg.learning_rate, help=f"Learning rate (default: {temp_cfg.learning_rate})")
    parser.add_argument('--epochs', type=int, default=temp_cfg.epochs, help=f"Number of epochs (default: {temp_cfg.epochs})")
    parser.add_argument('--max_batches_per_epoch', type=int, default=temp_cfg.max_batches_per_epoch, help=f"Max batches per epoch (default: {temp_cfg.max_batches_per_epoch})")
    parser.add_argument('--eval_every', type=int, default=temp_cfg.eval_every, help=f"Evaluate every N steps (default: {temp_cfg.eval_every})")

    # Dataset Args
    parser.add_argument('--dataset_name', type=str, default=temp_cfg.dataset_name, help=f"Hugging Face dataset name (default: {temp_cfg.dataset_name})")
    parser.add_argument('--dataset_config_name', type=str, default=temp_cfg.dataset_config_name, help=f"Hugging Face dataset config name (default: {temp_cfg.dataset_config_name})")
    parser.add_argument('--num_train_samples', type=int, default=temp_cfg.num_train_samples, help=f"Number of training samples (default: {temp_cfg.num_train_samples})")
    parser.add_argument('--num_eval_samples', type=int, default=temp_cfg.num_eval_samples, help=f"Number of evaluation samples (default: {temp_cfg.num_eval_samples})")

    # Checkpointing & Output Args
    parser.add_argument('--checkpoint_dir', type=str, default=temp_cfg.checkpoint_dir, help=f"Base directory for checkpoints (default: {temp_cfg.checkpoint_dir}).")
    parser.add_argument('--resume_checkpoint', type=str, default=temp_cfg.resume_checkpoint, help="Path to checkpoint to resume training from.")
    parser.add_argument('--run_name', type=str, default=temp_cfg.run_name, help="Optional run name for outputs subdir.")
    
    # Technical Args
    parser.add_argument('--seed', type=int, default=temp_cfg.seed, help=f"Random seed (default: {temp_cfg.seed})")
    parser.add_argument('--num_workers_dataloader', type=int, default=temp_cfg.num_workers_dataloader, help=f"Num workers for DataLoader (default: {temp_cfg.num_workers_dataloader})")
    parser.add_argument('--quiet', action='store_true', help="Suppress most print statements for cleaner sweep logs.")

    args = parser.parse_args()

    # Create a config instance
    cfg = GNNMoEConfig()

    # Override config with CLI arguments
    # Handle hgnn_learnable_edge_weights separately due to mutually exclusive group
    hgnn_learnable_value_from_cli = None
    if args.hgnn_learnable_cli_true: # This implies --hgnn_learnable_edge_weights was passed
        hgnn_learnable_value_from_cli = True
    elif hasattr(args, 'hgnn_learnable_cli_false') and not args.hgnn_learnable_cli_false: # This implies --no_hgnn_learnable_edge_weights was passed
        hgnn_learnable_value_from_cli = False
        
    for arg_name, arg_val in vars(args).items():
        if arg_name in ['hgnn_learnable_cli_true', 'hgnn_learnable_cli_false']:
            continue # Skip these temp argparse dest names

        if hasattr(cfg, arg_name):
            # Get the default value for comparison to avoid unnecessary "Overriding" prints
            cfg_default_val = getattr(temp_cfg, arg_name)
            if arg_val is not None and arg_val != cfg_default_val:
                if arg_name not in ['run_name', 'resume_checkpoint', 'checkpoint_dir', 'quiet']: # these are handled specially or control verbosity
                     verbose_print(f"Overriding config.{arg_name} with CLI arg: {arg_val}")
                setattr(cfg, arg_name, arg_val)
            elif arg_val is not None and arg_name in ['quiet']: # for flags like quiet, always set if present
                setattr(cfg, arg_name, arg_val)


    if hgnn_learnable_value_from_cli is not None:
        if cfg.hgnn_learnable_edge_weights != hgnn_learnable_value_from_cli:
             verbose_print(f"Overriding config.hgnn_learnable_edge_weights from {cfg.hgnn_learnable_edge_weights} to CLI arg: {hgnn_learnable_value_from_cli}")
        cfg.hgnn_learnable_edge_weights = hgnn_learnable_value_from_cli
    
    # Set global verbose flag
    if args.quiet:
        _VERBOSE = False

    base_checkpoint_dir_from_arg = args.checkpoint_dir
    if cfg.run_name:
        cfg.checkpoint_dir = os.path.join(base_checkpoint_dir_from_arg, cfg.run_name)
    else:
        cfg.checkpoint_dir = base_checkpoint_dir_from_arg
    
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)
        verbose_print(f"📁 Created checkpoint directory: {cfg.checkpoint_dir}")

    verbose_print("===== GNN-MoE Hyperparameter Script Execution Started =====")
    if cfg.run_name: verbose_print(f"Run Name: {cfg.run_name}")
    
    # Call __post_init__ logic after all CLI overrides
    cfg.__post_init__() # This will print adjustments e.g. num_heads
    verbose_print(f"Effective Config: {cfg}")


    selected_device = setup_environment(cfg)
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg)
    
    verbose_print(f"\n🏗️ Creating GNN-MoE Model with effective vocab_size: {cfg.vocab_size}")
    model = GNNMoEModel(cfg).to(selected_device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    
    actual_batches_per_epoch_main = len(train_loader) if cfg.max_batches_per_epoch == -1 else min(len(train_loader), cfg.max_batches_per_epoch)
    if actual_batches_per_epoch_main == 0 and cfg.num_train_samples != 0 :
        print(f"ERROR: Train loader has 0 batches. Check dataset path and num_train_samples ({cfg.num_train_samples}).")
        exit(1)
    total_steps_main = cfg.epochs * actual_batches_per_epoch_main
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_main, eta_min=1e-6) if total_steps_main > 0 else None

    start_epoch = 0
    current_step = 0
    best_eval_loss_resumed = float('inf')

    if cfg.resume_checkpoint:
        if os.path.isfile(cfg.resume_checkpoint):
            verbose_print(f"Attempting to resume from: {cfg.resume_checkpoint}")
            resume_data = load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler)
            if resume_data:
                start_epoch, current_step, best_eval_loss_resumed = resume_data
            model.to(selected_device) 
        else:
            verbose_print(f"⚠️ Resume checkpoint not found: '{cfg.resume_checkpoint}'. Starting fresh.")

    training_stats, final_best_loss = train_gnn_moe(
        model, train_loader, eval_loader, selected_device, cfg, 
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    if training_stats: 
        plot_training_results(training_stats, cfg)
        
        best_model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth.tar")
        if os.path.exists(best_model_path):
            verbose_print(f"🔄 Loading best model from {best_model_path} for final analysis...")
            final_analysis_model = GNNMoEModel(cfg).to(selected_device)
            load_checkpoint(best_model_path, final_analysis_model)
            
            communication_data = analyze_expert_communication(final_analysis_model, cfg, detailed=False)
            if communication_data:
                plot_expert_connectivity(communication_data, cfg)
            analyze_model_efficiency(final_analysis_model, cfg)
        else: 
            verbose_print(f"⚠️ best_model.pth.tar not found in {cfg.checkpoint_dir}, analyzing current model state.")
            communication_data = analyze_expert_communication(model, cfg, detailed=False)
            if communication_data:
                plot_expert_connectivity(communication_data, cfg)
            analyze_model_efficiency(model, cfg)

    summary_data = {
        "run_name": cfg.run_name if cfg.run_name else 'default_run',
        "data_mode": data_mode,
        "best_eval_loss": float(f"{final_best_loss:.4f}") if final_best_loss != float('inf') else None,
        "best_eval_perplexity": None
    }
    if training_stats and 'eval_perplexity' in training_stats and training_stats['eval_perplexity']:
        try:
            if final_best_loss in training_stats['eval_loss']:
                 best_loss_idx = training_stats['eval_loss'].index(final_best_loss)
                 summary_data["best_eval_perplexity"] = float(f"{training_stats['eval_perplexity'][best_loss_idx]:.2f}")
            elif training_stats['eval_perplexity']: # Check if list is not empty
                 summary_data["best_eval_perplexity"] = float(f"{min(training_stats['eval_perplexity']):.2f}")
        except (ValueError, IndexError, TypeError):
             if training_stats['eval_perplexity']:
                 try:
                    summary_data["best_eval_perplexity"] = float(f"{min(training_stats['eval_perplexity']):.2f}")
                 except TypeError:
                    pass

    print("\n🎉 GNN-MoE Hyperparameter Script Execution Finished Successfully!")
    print(f"   Run Name: {summary_data['run_name']}")
    print(f"   Data Mode: {summary_data['data_mode']}")
    if summary_data['best_eval_loss'] is not None:
        print(f"   Best Eval Loss from run: {summary_data['best_eval_loss']:.4f}")
    if summary_data['best_eval_perplexity'] is not None:
        print(f"   Best Eval Perplexity from run: {summary_data['best_eval_perplexity']:.2f}")
    print("==============================================")

    summary_file_path = os.path.join(cfg.checkpoint_dir, "run_summary.json")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        verbose_print(f"📝 Run summary saved to {summary_file_path}")
    except Exception as e:
        verbose_print(f"⚠️ Error saving run summary to JSON: {e}")

else:
    verbose_print("GNN-MoE Hyperparameter script (run_gnn_moe.py) imported as a module.")
