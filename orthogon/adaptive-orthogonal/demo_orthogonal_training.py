#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_orthogonal_training.py

Simple demonstration of orthogonal expert training with HGNN-MoE.
This script shows how to set up and run a small training experiment
with orthogonality constraints.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel
from gnn_moe_training import train_gnn_moe, prepare_batch
from orthogonal_analysis import (
    plot_orthogonality_training_curves,
    compute_expert_similarity_matrix,
    plot_expert_similarity_heatmap,
    generate_orthogonality_report
)

def create_synthetic_dataset(vocab_size=1000, num_samples=1000, seq_len=32):
    """Create a simple synthetic dataset for demonstration."""
    print(f"📚 Creating synthetic dataset: {num_samples} samples, seq_len={seq_len}")
    
    # Generate random sequences
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    # Create attention masks (all ones for simplicity)
    attention_mask = torch.ones_like(input_ids)
    
    # Create dataset - using a custom dataset class for dictionary format
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            
        def __len__(self):
            return len(self.input_ids)
            
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx]
            }
    
    dataset = DictDataset(input_ids, attention_mask)
    return dataset

def run_orthogonal_training_demo():
    """Run a complete orthogonal training demonstration."""
    print("🚀 Starting Orthogonal Expert Training Demo\n")
    
    # Configuration for small-scale demo
    config = GNNMoEConfig(
        # Model architecture
        num_experts=4,
        embed_dim=128,
        num_layers=2,
        num_heads=8,
        vocab_size=1000,
        max_seq_length=32,
        
        # Training setup
        batch_size=16,
        learning_rate=1e-3,
        epochs=3,
        max_batches_per_epoch=50,  # Limit for demo
        eval_every=10,
        
        # Coupler configuration
        coupler_type="HGNN",  # Use HGNN for advanced coupling
        static_hyperedge_strategy="all_triplets",
        hgnn_learnable_edge_weights=True,
        
        # Orthogonal expert training
        apply_orthogonality_loss=True,
        orthogonality_loss_weight=0.1,
        orthogonality_loss_type="gram_identity",
        orthogonality_aggregation="mean",
        orthogonality_warmup_steps=20,
        track_expert_specialization=True,
        
        # Output
        checkpoint_dir="demo_checkpoints",
        run_name="orthogonal_demo"
    )
    
    print("⚙️ Configuration:")
    print(f"   Model: {config.coupler_type}-MoE with {config.num_experts} experts")
    print(f"   Orthogonality: {config.orthogonality_loss_type} loss (λ={config.orthogonality_loss_weight})")
    print(f"   Training: {config.epochs} epochs, {config.max_batches_per_epoch} batches/epoch")
    print()
    
    # Create synthetic dataset
    train_dataset = create_synthetic_dataset(
        vocab_size=config.vocab_size,
        num_samples=800,
        seq_len=config.max_seq_length
    )
    
    eval_dataset = create_synthetic_dataset(
        vocab_size=config.vocab_size, 
        num_samples=200,
        seq_len=config.max_seq_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Single-threaded for demo
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    model = GNNMoEModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ Model created: {total_params:,} parameters")
    
    # Analyze initial expert similarity
    print("\n🔍 Analyzing initial expert similarity...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        input_ids = sample_batch['input_ids'].to(device)
        attention_mask = sample_batch['attention_mask'].to(device)
        
        # Forward pass to initialize expert outputs
        model(input_ids, attention_mask)
        
        # Get initial orthogonality metrics
        initial_metrics = model.get_expert_specialization_metrics()
        print(f"Initial orthogonality loss: {initial_metrics.get('total_orthogonality_loss', 0):.6f}")
    
    # Train the model
    print("\n🎯 Starting training...")
    
    try:
        stats, best_loss = train_gnn_moe(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            config=config
        )
        
        print(f"\n✅ Training completed! Best eval loss: {best_loss:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        stats = {}
        best_loss = float('inf')
    
    # Analyze final expert similarity
    print("\n🔍 Analyzing final expert similarity...")
    model.eval()
    with torch.no_grad():
        # Forward pass to get final expert outputs  
        model(input_ids, attention_mask)
        
        # Get final orthogonality metrics
        final_metrics = model.get_expert_specialization_metrics()
        print(f"Final orthogonality loss: {final_metrics.get('total_orthogonality_loss', 0):.6f}")
    
    # Generate analysis plots if we have training data
    if stats and any(stats.values()):
        print("\n📈 Generating analysis plots...")
        
        try:
            # Plot training curves
            fig = plot_orthogonality_training_curves(stats)
            plt.savefig("demo_training_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Training curves saved as 'demo_training_curves.png'")
            
            # Generate comprehensive report
            report_path = generate_orthogonality_report(
                model=model,
                stats=stats,
                output_dir="demo_analysis",
                config=config,
                sample_input=input_ids
            )
            print(f"✅ Analysis report generated: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Plotting failed (might be due to display): {e}")
    
    # Summary
    print("\n🎉 Demo completed!")
    print("\nKey takeaways:")
    print("• Orthogonal expert training encourages expert specialization")
    print("• HGNN coupling enables rich multi-expert communication")
    print("• Warmup prevents early constraint interference")
    print("• Analysis tools help monitor expert differentiation")
    
    if stats.get('orthogonality_loss'):
        ortho_losses = stats['orthogonality_loss']
        if len(ortho_losses) > 10:
            initial_avg = np.mean(ortho_losses[:10])
            final_avg = np.mean(ortho_losses[-10:])
            reduction = (initial_avg - final_avg) / initial_avg * 100
            print(f"• Orthogonality loss reduced by {reduction:.1f}% during training")
    
    return model, stats, config

if __name__ == '__main__':
    # Run the demonstration
    model, stats, config = run_orthogonal_training_demo()
    
    print(f"\n💡 To run a full experiment, modify the config in {__file__}")
    print("   and increase epochs, batch size, and sequence length.")
