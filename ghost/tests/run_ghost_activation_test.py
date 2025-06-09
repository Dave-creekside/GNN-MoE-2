#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ghost_activation_test.py

Focused test to demonstrate Ghost Expert activation with optimal parameters.
Based on user discoveries:
- Low threshold (0.01) for fast activation
- Higher embed_dim (512) for better saturation
- Frequent evaluation to capture activation dynamics
"""

import os
import sys
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import from ghost module using proper package structure
from ghost.gnn_moe_config import GhostMoEConfig
from ghost.gnn_moe_architecture import GhostMoEModel
from ghost.gnn_moe_data import load_data
from ghost.gnn_moe_training import train_ghost_moe_model

def run_ghost_activation_test():
    """
    Run focused test to demonstrate ghost activation dynamics.
    """
    # Create timestamped test directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join('ghost', 'tests', 'test-runs', f'ghost_activation_test_{timestamp}')
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"🔬 Ghost Activation Test")
    print(f"📁 Test directory: {test_dir}")
    
    # Optimal configuration based on user discoveries
    config = GhostMoEConfig(
        # Architecture - larger embedding for better saturation
        embed_dim=512,
        num_layers=4,
        num_experts=4,
        num_ghost_experts=4,
        
        # Ghost configuration - low threshold for fast activation
        ghost_activation_threshold=0.01,  # Key insight: much lower threshold
        ghost_learning_rate=1e-4,
        ghost_activation_schedule="gradual",
        
        # Training - enough steps to see activation around 150
        epochs=2,
        max_batches_per_epoch=150,  # Should give us ~300 steps total
        eval_every=25,  # Frequent evaluation to capture dynamics
        
        # Dataset - small subset for laptop compatibility
        num_train_samples=1500,
        num_eval_samples=300,
        
        # Output
        checkpoint_dir=test_dir,
        run_name="ghost_activation_demo"
    )
    
    print(f"📊 Configuration:")
    print(f"   Embed dim: {config.embed_dim}")
    print(f"   Ghost threshold: {config.ghost_activation_threshold}")
    print(f"   Max steps: ~{config.epochs * config.max_batches_per_epoch}")
    print(f"   Eval every: {config.eval_every} steps")
    
    # Save configuration
    config_path = os.path.join(test_dir, 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=4)
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print(f"📚 Loading WikiText-2 data...")
    train_loader, eval_loader, tokenizer, data_mode = load_data(config)
    print(f"   Data mode: {data_mode}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Eval batches: {len(eval_loader)}")
    
    # Initialize model
    print(f"🤖 Initializing Ghost MoE model...")
    model = GhostMoEModel(config, tokenizer.vocab_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Train model
    print(f"\n🚀 Starting training (expecting ghost activation around step 150)...")
    training_log, best_loss = train_ghost_moe_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        config=config
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Best eval loss: {best_loss:.4f}")
    print(f"   Training log entries: {len(training_log)}")
    
    # Analyze and plot results
    print(f"\n📈 Generating analysis plots...")
    analyze_ghost_activation(training_log, test_dir)
    
    # Print activation summary
    print_activation_summary(training_log)
    
    print(f"\n🎯 Test completed! Results saved to: {test_dir}")
    return test_dir

def analyze_ghost_activation(training_log, output_dir):
    """Generate detailed plots showing ghost activation dynamics."""
    df = pd.DataFrame(training_log)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ghost Expert Activation Dynamics Test', fontsize=16)
    
    # 1. Loss curves
    axes[0,0].plot(df['step'], df['train_loss'], label='Train Loss', alpha=0.7)
    axes[0,0].plot(df['step'], df['eval_loss'], label='Eval Loss', marker='o', markersize=4)
    axes[0,0].set_title('Loss Curves')
    axes[0,0].set_xlabel('Step')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Ghost activations over time
    ghost_activations = pd.DataFrame(df['ghost_activations'].tolist(), index=df['step'])
    ghost_activations.columns = [f'Ghost {i}' for i in range(len(ghost_activations.columns))]
    
    for i, col in enumerate(ghost_activations.columns):
        axes[0,1].plot(df['step'], ghost_activations[col], 
                      label=col, marker='o', markersize=3, alpha=0.8)
    
    axes[0,1].set_title('Ghost Expert Activation Levels')
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('Activation Level')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(-0.05, 1.05)
    
    # 3. Saturation dynamics
    axes[1,0].plot(df['step'], df['saturation_level'], 
                   label='Saturation Level', color='red', marker='s', markersize=3)
    ax_twin = axes[1,0].twinx()
    ax_twin.plot(df['step'], df['orthogonality_score'], 
                 label='Orthogonality Score', color='blue', marker='^', markersize=3)
    
    axes[1,0].set_title('Expert Saturation vs Orthogonality')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Saturation Level', color='red')
    ax_twin.set_ylabel('Orthogonality Score', color='blue')
    axes[1,0].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Learning rates
    axes[1,1].plot(df['step'], df['primary_lr'], label='Primary LR', marker='o', markersize=3)
    
    # Plot ghost learning rates
    if 'ghost_lrs' in df.columns:
        ghost_lrs = pd.DataFrame(df['ghost_lrs'].tolist(), index=df['step'])
        for i, col in enumerate(ghost_lrs.columns):
            axes[1,1].plot(df['step'], ghost_lrs[col], 
                          label=f'Ghost {i} LR', alpha=0.7, marker='s', markersize=2)
    
    axes[1,1].set_title('Learning Rate Dynamics')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].legend()
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'ghost_activation_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Analysis plot saved: {plot_path}")

def print_activation_summary(training_log):
    """Print summary of ghost activation behavior."""
    df = pd.DataFrame(training_log)
    
    print(f"\n📋 Ghost Activation Summary:")
    
    # Find when each ghost first activates (>0.1)
    ghost_activations = pd.DataFrame(df['ghost_activations'].tolist(), index=df['step'])
    
    for i, col in enumerate(ghost_activations.columns):
        activation_steps = df[ghost_activations[col] > 0.1]['step']
        if len(activation_steps) > 0:
            first_activation = activation_steps.iloc[0]
            final_activation = ghost_activations[col].iloc[-1]
            print(f"   Ghost {i}: First activated at step {first_activation}, final level: {final_activation:.3f}")
        else:
            print(f"   Ghost {i}: Never activated (threshold 0.1)")
    
    # Saturation summary
    initial_saturation = df['saturation_level'].iloc[0]
    final_saturation = df['saturation_level'].iloc[-1]
    max_saturation = df['saturation_level'].max()
    
    print(f"\n📊 Saturation Evolution:")
    print(f"   Initial: {initial_saturation:.4f}")
    print(f"   Maximum: {max_saturation:.4f}")
    print(f"   Final: {final_saturation:.4f}")
    
    # Loss improvement
    initial_loss = df['eval_loss'].iloc[0]
    final_loss = df['eval_loss'].iloc[-1]
    best_loss = df['eval_loss'].min()
    
    print(f"\n📉 Loss Evolution:")
    print(f"   Initial eval loss: {initial_loss:.4f}")
    print(f"   Best eval loss: {best_loss:.4f}")
    print(f"   Final eval loss: {final_loss:.4f}")
    print(f"   Total improvement: {initial_loss - final_loss:.4f}")

if __name__ == "__main__":
    test_dir = run_ghost_activation_test()
