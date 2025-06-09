#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gnn_moe.py

Main executable script for GNN-Coupled MoE model training and experimentation.
Uses modularized components for config, architecture, data, training, and analysis.
Allows setting model architecture, training params, and dataset via CLI.
"""

import torch
import torch.optim as optim
import os
import argparse
from dataclasses import fields # To iterate over dataclass fields

import numpy as np # Added
import matplotlib.pyplot as plt # Added
import seaborn as sns # Added
import random # Added
# os is already imported
import json # For saving summary

# --- Conditional Print Function ---
# This will be set based on args.quiet later
_VERBOSE = True
def verbose_print(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

# Import from our new modules
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel
from gnn_moe_data import load_data
# setup_environment will be defined locally in this script
from gnn_moe_training import load_checkpoint, train_gnn_moe 
from gnn_moe_analysis import plot_training_results, analyze_expert_communication, plot_expert_connectivity, analyze_model_efficiency

# --- Initial Setup Function (moved here) ---
def setup_environment(config: GNNMoEConfig): # Takes GNNMoEConfig
    # Matplotlib and Seaborn are imported globally, so they are available here
    plt.style.use('default')
    sns.set_palette("husl")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Device: CUDA (Available: {torch.cuda.device_count()})")
    elif torch.backends.mps.is_available(): # Check for MPS
        device = torch.device("mps")
        print("🚀 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("🚀 Device: CPU")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Create base plots directory if it doesn't exist
    # Note: config.checkpoint_dir (run-specific or base) is created in main block
    if not os.path.exists("plots"): 
        os.makedirs("plots")
        print("📁 Created 'plots' directory for output visualizations.")
    
    # The specific run's checkpoint directory (cfg.checkpoint_dir) is created in the main block
    # after args are parsed. The base 'checkpoints' dir (if different and if run_name is used)
    # might also be created if not os.path.exists(base_checkpoint_dir_from_arg) was added.
    # For simplicity, setup_environment only ensures ./plots.
    # The default "checkpoints" dir (if no run_name and no --checkpoint_dir override)
    # would be implicitly handled by the logic in __main__ that sets up cfg.checkpoint_dir.

    print(f"✅ Environment ready. Seed: {config.seed}, Device: {device}")
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-MoE Hyperparameter Training Script")
    
    # Dynamically add arguments from GNNMoEConfig defaults
    # This makes it easy to update GNNMoEConfig and have CLI args reflect changes.
    config_fields = {f.name: f for f in fields(GNNMoEConfig)}

    # Model Architecture Args
    parser.add_argument('--embed_dim', type=int, default=config_fields['embed_dim'].default, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=config_fields['num_layers'].default, help="Number of GNNMoELayers")
    parser.add_argument('--num_heads', type=int, default=config_fields['num_heads'].default, help="Number of attention heads")
    parser.add_argument('--dropout_rate', type=float, default=config_fields['dropout_rate'].default, help="Dropout rate")
    parser.add_argument('--num_experts', type=int, default=config_fields['num_experts'].default, help="Number of experts per layer")
    parser.add_argument('--gnn_layers', type=int, default=config_fields['gnn_layers'].default, help="Number of GNN layers in coupler")

    # Training Hyperparameters Args
    parser.add_argument('--batch_size', type=int, default=config_fields['batch_size'].default, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=config_fields['learning_rate'].default, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=config_fields['epochs'].default, help="Number of epochs")
    parser.add_argument('--max_batches_per_epoch', type=int, default=config_fields['max_batches_per_epoch'].default, help="Max batches per epoch (-1 for full epoch)")
    parser.add_argument('--eval_every', type=int, default=config_fields['eval_every'].default, help="Evaluate every N steps")

    # Dataset Args
    parser.add_argument('--dataset_name', type=str, default=config_fields['dataset_name'].default, help="Hugging Face dataset name")
    parser.add_argument('--dataset_config_name', type=str, default=config_fields['dataset_config_name'].default, help="Hugging Face dataset config name")
    parser.add_argument('--num_train_samples', type=int, default=config_fields['num_train_samples'].default, help="Number of training samples (-1 for all)")
    parser.add_argument('--num_eval_samples', type=int, default=config_fields['num_eval_samples'].default, help="Number of evaluation samples (-1 for all)")

    # Checkpointing & Output Args
    parser.add_argument('--checkpoint_dir', type=str, default=config_fields['checkpoint_dir'].default, help="Base directory for checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, default=config_fields['resume_checkpoint'].default, help="Path to checkpoint to resume training from.")
    parser.add_argument('--run_name', type=str, default=config_fields['run_name'].default, help="Optional run name for outputs subdir")
    
    # Technical Args
    parser.add_argument('--seed', type=int, default=config_fields['seed'].default, help="Random seed")
    parser.add_argument('--num_workers_dataloader', type=int, default=config_fields['num_workers_dataloader'].default, help="Num workers for DataLoader")
    parser.add_argument('--quiet', action='store_true', help="Suppress most print statements for cleaner sweep logs.")

    args = parser.parse_args()

    # Create a config instance and override with CLI arguments
    cfg = GNNMoEConfig()
    for arg_name, arg_val in vars(args).items():
        if hasattr(cfg, arg_name) and arg_val is not None: # Check if arg is a field in GNNMoEConfig
            # Special handling for checkpoint_dir which is base, run_name makes it specific
            if arg_name not in ['run_name', 'resume_checkpoint', 'checkpoint_dir']: # these are handled separately or are not direct cfg overrides in the same way
                 print(f"Overriding config.{arg_name} with CLI arg: {arg_val}")
            setattr(cfg, arg_name, arg_val)
    
    
    # Set global _VERBOSE based on --quiet flag
    # global _VERBOSE # Not needed here as we are in module scope
    if args.quiet:
        _VERBOSE = False # verbose_print will use this updated module-level global

    # Handle run_name and checkpoint_dir logic
    # cfg.checkpoint_dir is initialized with the default "checkpoints" or the CLI override for the base.
    # If run_name is given, create a subdirectory.
    base_checkpoint_dir_from_arg = args.checkpoint_dir # This is what user specified for base, or its default
    if cfg.run_name:
        cfg.checkpoint_dir = os.path.join(base_checkpoint_dir_from_arg, cfg.run_name)
    else: # No run_name, use the checkpoint_dir as is (could be default "checkpoints" or user-specified base)
        cfg.checkpoint_dir = base_checkpoint_dir_from_arg
    
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)
        verbose_print(f"📁 Created checkpoint directory: {cfg.checkpoint_dir}")

    verbose_print("===== GNN-MoE Hyperparameter Script Execution Started =====")
    if cfg.run_name: verbose_print(f"Run Name: {cfg.run_name}")
    verbose_print(f"Effective Config: {cfg}")

    selected_device = setup_environment(cfg) # setup_environment uses cfg.seed and its own prints
    
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg) # load_data uses its own prints
    
    # vocab_size in cfg might be updated by load_data if tokenizer's vocab is different
    verbose_print(f"\n🏗️ Creating GNN-MoE Model with effective vocab_size: {cfg.vocab_size}")
    model = GNNMoEModel(cfg).to(selected_device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    
    # Calculate total_steps for scheduler
    actual_batches_per_epoch_main = len(train_loader) if cfg.max_batches_per_epoch == -1 else min(len(train_loader), cfg.max_batches_per_epoch)
    if actual_batches_per_epoch_main == 0 and cfg.num_train_samples != 0 : # If loader is empty but samples were requested
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
            # load_checkpoint expects model, optimizer, scheduler to be defined
            resume_data = load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler) # load_checkpoint has its own prints
            if resume_data:
                start_epoch, current_step, best_eval_loss_resumed = resume_data
            model.to(selected_device) 
        else:
            verbose_print(f"⚠️ Resume checkpoint not found: '{cfg.resume_checkpoint}'. Starting fresh.")

    # cfg.checkpoint_dir is already set to the specific run's directory if run_name was provided
    training_stats, final_best_loss = train_gnn_moe(
        model, train_loader, eval_loader, selected_device, cfg, 
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    if training_stats: 
        plot_training_results(training_stats, cfg) # Uses cfg.run_name for plot filename
        
        best_model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth.tar")
        if os.path.exists(best_model_path):
            verbose_print(f"🔄 Loading best model from {best_model_path} for final analysis...")
            # Create a new model instance with the same config to load best weights
            final_analysis_model = GNNMoEModel(cfg).to(selected_device)
            load_checkpoint(best_model_path, final_analysis_model) # Loads model_state_dict, has own prints
            
            communication_data = analyze_expert_communication(final_analysis_model, cfg, detailed=False) # Has own prints
            if communication_data:
                plot_expert_connectivity(communication_data, cfg) # Has own prints
            analyze_model_efficiency(final_analysis_model, cfg) # Has own prints
        else: 
            verbose_print(f"⚠️ best_model.pth.tar not found in {cfg.checkpoint_dir}, analyzing current model state.")
            communication_data = analyze_expert_communication(model, cfg, detailed=False) # Has own prints
            if communication_data:
                plot_expert_connectivity(communication_data, cfg) # Has own prints
            analyze_model_efficiency(model, cfg) # Has own prints

    # Always print this summary, regardless of quiet flag
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
            elif training_stats['eval_perplexity']:
                 summary_data["best_eval_perplexity"] = float(f"{min(training_stats['eval_perplexity']):.2f}")
        except (ValueError, IndexError, TypeError):
             if training_stats['eval_perplexity']:
                 try:
                    summary_data["best_eval_perplexity"] = float(f"{min(training_stats['eval_perplexity']):.2f}")
                 except TypeError: # Handle if min() result is not float convertible
                    pass


    print("\n🎉 GNN-MoE Hyperparameter Script Execution Finished Successfully!")
    print(f"   Run Name: {summary_data['run_name']}")
    print(f"   Data Mode: {summary_data['data_mode']}")
    if summary_data['best_eval_loss'] is not None:
        print(f"   Best Eval Loss from run: {summary_data['best_eval_loss']:.4f}")
    if summary_data['best_eval_perplexity'] is not None:
        print(f"   Best Eval Perplexity from run: {summary_data['best_eval_perplexity']:.2f}")
    print("==============================================")

    # Save summary to JSON file
    summary_file_path = os.path.join(cfg.checkpoint_dir, "run_summary.json")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"📝 Run summary saved to {summary_file_path}")
    except Exception as e:
        print(f"⚠️ Error saving run summary to JSON: {e}")

else:
    verbose_print("GNN-MoE Hyperparameter script (run_gnn_moe.py) imported as a module.")
