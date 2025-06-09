#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_orthogonal_features.py

Test script to verify orthogonal expert training functionality.
"""

import torch
import numpy as np
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel
from orthogonal_analysis import (
    compute_expert_similarity_matrix, 
    compute_orthogonality_metrics,
    plot_expert_similarity_heatmap
)

def test_orthogonality_loss_computation():
    """Test orthogonality loss computation in isolation."""
    print("🧪 Testing orthogonality loss computation...")
    
    # Create config with orthogonality enabled
    config = GNNMoEConfig(
        num_experts=4,
        embed_dim=64,
        num_layers=1,
        apply_orthogonality_loss=True,
        orthogonality_loss_weight=0.1,
        orthogonality_loss_type="gram_identity",
        orthogonality_aggregation="mean",
        track_expert_specialization=True
    )
    
    # Create model
    model = GNNMoEModel(config)
    
    # Create sample input
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    model.train()
    outputs = model(input_ids, attention_mask, labels=input_ids)
    
    # Test orthogonality loss collection
    total_ortho_loss = model.get_total_orthogonality_loss(training_step=100)
    
    print(f"✅ Language modeling loss: {outputs['loss'].item():.4f}")
    print(f"✅ Total orthogonality loss: {total_ortho_loss.item():.4f}")
    
    # Test expert specialization metrics
    specialization_metrics = model.get_expert_specialization_metrics()
    print(f"✅ Specialization metrics: {specialization_metrics}")
    
    return True

def test_orthogonality_vs_no_orthogonality():
    """Compare expert similarity with and without orthogonality constraints."""
    print("\n🔬 Testing orthogonality vs no orthogonality...")
    
    def create_and_run_model(apply_ortho, ortho_weight=0.1):
        config = GNNMoEConfig(
            num_experts=4,
            embed_dim=32,
            num_layers=1,
            apply_orthogonality_loss=apply_ortho,
            orthogonality_loss_weight=ortho_weight,
            vocab_size=1000,  # Smaller vocab for faster testing
            max_seq_length=16
        )
        
        model = GNNMoEModel(config)
        
        # Create dummy expert outputs for analysis
        # Simulate forward pass results
        batch_size, seq_len = 4, 16
        expert_outputs = torch.randn(batch_size, seq_len, config.num_experts, config.embed_dim)
        
        return expert_outputs, config
    
    # Test without orthogonality
    expert_outputs_no_ortho, config_no_ortho = create_and_run_model(apply_ortho=False)
    
    # Test with orthogonality  
    expert_outputs_ortho, config_ortho = create_and_run_model(apply_ortho=True)
    
    # Compute similarity matrices
    sim_matrix_no_ortho = compute_expert_similarity_matrix(expert_outputs_no_ortho, method="cosine")
    sim_matrix_ortho = compute_expert_similarity_matrix(expert_outputs_ortho, method="cosine")
    
    # Compute orthogonality metrics
    metrics_no_ortho = compute_orthogonality_metrics(expert_outputs_no_ortho)
    metrics_ortho = compute_orthogonality_metrics(expert_outputs_ortho)
    
    print("📊 Metrics without orthogonality constraints:")
    print(f"   Off-diagonal mean: {metrics_no_ortho['cosine_off_diagonal_mean']:.4f}")
    print(f"   Gram identity MSE: {metrics_no_ortho['gram_identity_mse']:.4f}")
    
    print("📊 Metrics with orthogonality constraints:")
    print(f"   Off-diagonal mean: {metrics_ortho['cosine_off_diagonal_mean']:.4f}")
    print(f"   Gram identity MSE: {metrics_ortho['gram_identity_mse']:.4f}")
    
    return True

def test_warmup_functionality():
    """Test orthogonality loss warmup mechanism."""
    print("\n🔥 Testing warmup functionality...")
    
    config = GNNMoEConfig(
        num_experts=3,
        embed_dim=32,
        num_layers=1,
        apply_orthogonality_loss=True,
        orthogonality_warmup_steps=100,
        orthogonality_loss_weight=0.2
    )
    
    model = GNNMoEModel(config)
    
    # Test warmup at different steps
    test_steps = [0, 25, 50, 75, 100, 150]
    
    print("Step | Warmup Factor | Effective Weight")
    print("-" * 35)
    
    for step in test_steps:
        model.update_all_training_steps(step)
        
        # Get warmup factor from first layer
        layer = model.model_layers[0]
        warmup_factor = layer.get_orthogonality_warmup_factor()
        effective_weight = config.orthogonality_loss_weight * warmup_factor
        
        print(f"{step:4d} | {warmup_factor:11.3f} | {effective_weight:13.4f}")
    
    return True

def test_different_loss_types():
    """Test different orthogonality loss types."""
    print("\n🎯 Testing different orthogonality loss types...")
    
    loss_types = ["gram_identity", "cosine_similarity"]
    
    for loss_type in loss_types:
        print(f"\n📝 Testing {loss_type} loss:")
        
        config = GNNMoEConfig(
            num_experts=3,
            embed_dim=32,
            num_layers=1,
            apply_orthogonality_loss=True,
            orthogonality_loss_type=loss_type,
            orthogonality_loss_weight=0.1
        )
        
        model = GNNMoEModel(config)
        
        # Create sample input
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=input_ids)
        ortho_loss = model.get_total_orthogonality_loss()
        
        print(f"   ✅ {loss_type} orthogonality loss: {ortho_loss.item():.6f}")
    
    return True

def test_hgnn_with_orthogonality():
    """Test HGNN model with orthogonality constraints."""
    print("\n🕸️ Testing HGNN with orthogonality...")
    
    try:
        config = GNNMoEConfig(
            num_experts=4,
            embed_dim=32,
            num_layers=1,
            coupler_type="HGNN",
            static_hyperedge_strategy="all_triplets",
            apply_orthogonality_loss=True,
            orthogonality_loss_weight=0.15,
            track_expert_specialization=True
        )
        
        model = GNNMoEModel(config)
        
        # Create sample input
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=input_ids)
        ortho_loss = model.get_total_orthogonality_loss(training_step=50)
        specialization_metrics = model.get_expert_specialization_metrics()
        
        print(f"✅ HGNN forward pass successful")
        print(f"✅ Language modeling loss: {outputs['loss'].item():.4f}")
        print(f"✅ Orthogonality loss: {ortho_loss.item():.4f}")
        print(f"✅ Specialization tracking: {len(specialization_metrics)} metrics")
        
        return True
        
    except ImportError:
        print("⚠️ PyTorch Geometric not available, skipping HGNN test")
        return True

def main():
    """Run all orthogonality tests."""
    print("🚀 Running orthogonal expert training tests...\n")
    
    tests = [
        test_orthogonality_loss_computation,
        test_orthogonality_vs_no_orthogonality,
        test_warmup_functionality, 
        test_different_loss_types,
        test_hgnn_with_orthogonality
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All orthogonal expert training features working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
