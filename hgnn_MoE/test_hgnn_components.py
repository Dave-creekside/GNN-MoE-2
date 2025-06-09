#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_hgnn_components.py

Comprehensive tests for HGNN components following the testing plan.
"""

import torch
import torch.nn as nn
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import HGNNExpertCoupler, PyGHypergraphConvWrapper, GNNMoEModel

def test_hyperedge_generation():
    """Test Stage 1: Hyperedge Generation Logic"""
    print("=== Testing Hyperedge Generation ===")
    
    # Test all_pairs strategy with different num_experts
    test_cases = [
        (0, "all_pairs", 0, (2, 0)),  # Edge case: 0 experts
        (1, "all_pairs", 0, (2, 0)),  # Edge case: 1 expert
        (2, "all_pairs", 1, (2, 2)),  # 2 experts: 1 pair (0,1)
        (3, "all_pairs", 3, (2, 6)),  # 3 experts: 3 pairs (0,1), (0,2), (1,2)
        (4, "all_pairs", 6, (2, 12)), # 4 experts: 6 pairs
    ]
    
    for num_experts, strategy, expected_hyperedges, expected_shape in test_cases:
        config = GNNMoEConfig(
            num_experts=num_experts,
            coupler_type="HGNN",
            static_hyperedge_strategy=strategy,
            hgnn_learnable_edge_weights=False,
            embed_dim=32
        )
        
        if num_experts == 0:
            print(f"Skipping num_experts=0 (would cause issues in model)")
            continue
            
        try:
            coupler = HGNNExpertCoupler(config)
            print(f"✅ {strategy} with {num_experts} experts:")
            print(f"   Expected: {expected_hyperedges} hyperedges, shape {expected_shape}")
            print(f"   Actual: {coupler._num_hyperedges} hyperedges, shape {coupler._hyperedge_index.shape}")
            
            if coupler._num_hyperedges == expected_hyperedges and coupler._hyperedge_index.shape == expected_shape:
                print(f"   ✅ PASS")
                
                # Print the actual hyperedge structure for small cases
                if num_experts <= 4 and coupler._num_hyperedges > 0:
                    print(f"   Hyperedge index: {coupler._hyperedge_index}")
            else:
                print(f"   ❌ FAIL")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        print()
    
    # Test all_triplets strategy
    print("--- Testing all_triplets strategy ---")
    triplet_test_cases = [
        (2, "all_triplets", 0, (2, 0)),  # 2 experts: no triplets possible
        (3, "all_triplets", 1, (2, 3)),  # 3 experts: 1 triplet (0,1,2)  
        (4, "all_triplets", 4, (2, 12)), # 4 experts: 4 triplets
    ]
    
    for num_experts, strategy, expected_hyperedges, expected_shape in triplet_test_cases:
        config = GNNMoEConfig(
            num_experts=num_experts,
            coupler_type="HGNN", 
            static_hyperedge_strategy=strategy,
            hgnn_learnable_edge_weights=False,
            embed_dim=32
        )
        
        try:
            coupler = HGNNExpertCoupler(config)
            print(f"✅ {strategy} with {num_experts} experts:")
            print(f"   Expected: {expected_hyperedges} hyperedges, shape {expected_shape}")
            print(f"   Actual: {coupler._num_hyperedges} hyperedges, shape {coupler._hyperedge_index.shape}")
            
            if coupler._num_hyperedges == expected_hyperedges and coupler._hyperedge_index.shape == expected_shape:
                print(f"   ✅ PASS")
                
                # Print the actual hyperedge structure
                if coupler._num_hyperedges > 0:
                    print(f"   Hyperedge index: {coupler._hyperedge_index}")
            else:
                print(f"   ❌ FAIL")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        print()

def test_learnable_weights():
    """Test learnable hyperedge weights"""
    print("=== Testing Learnable Hyperedge Weights ===")
    
    # Test with learnable weights enabled
    config_with_weights = GNNMoEConfig(
        num_experts=3,
        coupler_type="HGNN",
        static_hyperedge_strategy="all_pairs",
        hgnn_learnable_edge_weights=True,
        embed_dim=32
    )
    
    coupler_with_weights = HGNNExpertCoupler(config_with_weights)
    print(f"✅ With learnable weights:")
    print(f"   Num hyperedges: {coupler_with_weights._num_hyperedges}")
    print(f"   Has hyperedge_weights: {coupler_with_weights.hyperedge_weights is not None}")
    if coupler_with_weights.hyperedge_weights is not None:
        print(f"   Weights shape: {coupler_with_weights.hyperedge_weights.shape}")
        print(f"   Weights (initial): {coupler_with_weights.hyperedge_weights.data}")
    
    # Test with learnable weights disabled
    config_no_weights = GNNMoEConfig(
        num_experts=3,
        coupler_type="HGNN",
        static_hyperedge_strategy="all_pairs", 
        hgnn_learnable_edge_weights=False,
        embed_dim=32
    )
    
    coupler_no_weights = HGNNExpertCoupler(config_no_weights)
    print(f"✅ Without learnable weights:")
    print(f"   Num hyperedges: {coupler_no_weights._num_hyperedges}")
    print(f"   Has hyperedge_weights: {coupler_no_weights.hyperedge_weights is not None}")
    print()

def test_forward_pass():
    """Test Stage 2: Forward pass with different configurations"""
    print("=== Testing Forward Pass ===")
    
    # Test data
    B, L, E, D = 2, 8, 3, 32
    expert_outputs = torch.randn(B, L, E, D)
    
    # Test HGNN forward pass
    config = GNNMoEConfig(
        num_experts=E,
        coupler_type="HGNN",
        static_hyperedge_strategy="all_pairs",
        hgnn_learnable_edge_weights=True,
        embed_dim=D,
        gnn_layers=1
    )
    
    try:
        coupler = HGNNExpertCoupler(config)
        print(f"✅ Forward pass test with shape {expert_outputs.shape}")
        
        output = coupler(expert_outputs)
        expected_output_shape = (B, L, D)
        
        print(f"   Input shape: {expert_outputs.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape: {expected_output_shape}")
        
        if output.shape == expected_output_shape:
            print(f"   ✅ PASS: Output shape correct")
        else:
            print(f"   ❌ FAIL: Output shape incorrect")
            
        # Test backward pass
        dummy_loss = output.sum()
        dummy_loss.backward()
        
        if coupler.hyperedge_weights.grad is not None:
            print(f"   ✅ PASS: Gradients computed for hyperedge_weights")
            print(f"   Gradient shape: {coupler.hyperedge_weights.grad.shape}")
        else:
            print(f"   ❌ FAIL: No gradients for hyperedge_weights")
            
    except Exception as e:
        print(f"   ❌ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
    print()

def test_full_model_integration():
    """Test Stage 2: Full model integration"""
    print("=== Testing Full Model Integration ===")
    
    # Small model for testing
    config = GNNMoEConfig(
        vocab_size=1000,  # Smaller vocab for testing
        max_seq_length=32,
        embed_dim=64, 
        num_layers=2,
        num_experts=3,
        gnn_layers=1,
        coupler_type="HGNN",
        static_hyperedge_strategy="all_pairs",
        hgnn_learnable_edge_weights=True
    )
    
    try:
        model = GNNMoEModel(config)
        print(f"✅ Model instantiated successfully")
        
        # Test data
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass without loss
        output = model(input_ids, attention_mask=attention_mask, return_loss=False)
        expected_logits_shape = (batch_size, seq_len, config.vocab_size)
        
        print(f"   Forward pass (no loss): Output logits shape {output['logits'].shape}")
        if output['logits'].shape == expected_logits_shape:
            print(f"   ✅ PASS: Forward pass shape correct")
        else:
            print(f"   ❌ FAIL: Forward pass shape incorrect")
        
        # Forward pass with loss
        output_with_loss = model(input_ids, attention_mask=attention_mask, return_loss=True, labels=input_ids)
        print(f"   Forward pass (with loss): Loss value {output_with_loss['loss'].item():.4f}")
        
        # Backward pass
        output_with_loss['loss'].backward()
        print(f"   ✅ PASS: Backward pass completed")
        
        # Check expert communication data
        comm_data = model.analyze_expert_communication()
        print(f"   Expert communication data: {list(comm_data.keys())}")
        
        if comm_data:
            for layer_name, data in comm_data.items():
                print(f"     {layer_name}: {len(data)} matrices/weights")
                for i, item in enumerate(data):
                    print(f"       Item {i} shape: {item.shape}")
        
    except Exception as e:
        print(f"   ❌ ERROR during full model test: {e}")
        import traceback
        traceback.print_exc()
    print()

if __name__ == '__main__':
    print("Running HGNN Component Tests")
    print("=" * 50)
    
    test_hyperedge_generation()
    test_learnable_weights()
    test_forward_pass()
    test_full_model_integration()
    
    print("=" * 50)
    print("Test suite completed!")
