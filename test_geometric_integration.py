#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_geometric_integration.py

Test script to validate the modular geometric training architecture integration.
"""

import torch
from core.config import MoEConfig, GeometricTrainingConfig
from core.architecture import MoEModel
from core.training_controllers import create_training_controller, StandardTrainingController, GeometricTrainingController

def test_config_system():
    """Test that configuration system works with geometric training options."""
    print("🧪 Testing Configuration System...")
    
    # Test 1: Default config should have geometric disabled
    config = MoEConfig()
    assert config.training_mode == "standard"
    assert config.geometric.enabled == False
    print("✅ Default config has geometric training disabled")
    
    # Test 2: Enable geometric training
    config.training_mode = "geometric"
    config.geometric.enabled = True
    config.geometric.geometric_learning_rate = 1e-3
    config.geometric.rotation_dimensions = 8
    print("✅ Geometric training configuration can be enabled")
    
    # Test 3: Serialization round-trip
    config_dict = config.to_dict()
    assert 'geometric' in config_dict
    assert 'training_mode' in config_dict
    
    config_restored = MoEConfig.from_dict(config_dict)
    assert config_restored.training_mode == "geometric"
    assert config_restored.geometric.enabled == True
    assert config_restored.geometric.geometric_learning_rate == 1e-3
    assert config_restored.geometric.rotation_dimensions == 8
    print("✅ Configuration serialization works correctly")
    
    print("🎉 Configuration system tests passed!\n")


def test_training_controller_factory():
    """Test that training controller factory creates the correct controllers."""
    print("🧪 Testing Training Controller Factory...")
    
    # Create a small test model with simple GNN architecture
    config = MoEConfig()
    config.embed_dim = 64
    config.num_experts = 2
    config.num_layers = 1
    config.architecture_mode = "gnn"  # Use simple GNN to avoid PyTorch Geometric dependency
    config.use_hypergraph_coupling = False  # Explicitly disable hypergraph coupling
    model = MoEModel(config)
    
    # Test 1: Standard training controller
    config.training_mode = "standard"
    controller = create_training_controller(model, config)
    assert isinstance(controller, StandardTrainingController)
    print("✅ Standard training controller created correctly")
    
    # Test 2: Geometric training controller (should fall back to standard for now)
    config.training_mode = "geometric"
    config.geometric.enabled = True
    controller = create_training_controller(model, config)
    assert isinstance(controller, GeometricTrainingController)
    print("✅ Geometric training controller created correctly")
    
    # Test 3: Geometric training disabled should warn and use standard
    config.training_mode = "geometric"
    config.geometric.enabled = False
    controller = create_training_controller(model, config)
    assert isinstance(controller, StandardTrainingController)
    print("✅ Geometric disabled correctly falls back to standard")
    
    # Test 4: Invalid training mode should raise error
    config.training_mode = "invalid"
    try:
        controller = create_training_controller(model, config)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid training mode correctly raises error")
    
    print("🎉 Training controller factory tests passed!\n")


def test_training_controller_interface():
    """Test that training controllers implement the required interface."""
    print("🧪 Testing Training Controller Interface...")
    
    config = MoEConfig()
    config.embed_dim = 64
    config.num_experts = 2
    config.num_layers = 1
    config.architecture_mode = "gnn"  # Use simple GNN to avoid PyTorch Geometric dependency
    config.use_hypergraph_coupling = False  # Explicitly disable hypergraph coupling
    model = MoEModel(config)
    
    # Test standard controller interface
    controller = create_training_controller(model, config)
    
    # Test required methods exist
    assert hasattr(controller, 'training_step')
    assert hasattr(controller, 'get_optimizers')
    assert hasattr(controller, 'get_schedulers')
    assert hasattr(controller, 'get_current_metrics')
    print("✅ Standard controller has required interface")
    
    # Test that optimizers and schedulers return lists
    optimizers = controller.get_optimizers()
    schedulers = controller.get_schedulers()
    metrics = controller.get_current_metrics()
    
    assert isinstance(optimizers, list)
    assert isinstance(schedulers, list)
    assert isinstance(metrics, dict)
    assert len(optimizers) > 0
    assert len(schedulers) > 0
    print("✅ Controller methods return expected types")
    
    print("🎉 Training controller interface tests passed!\n")


def test_geometric_config_options():
    """Test that geometric configuration has all expected options."""
    print("🧪 Testing Geometric Configuration Options...")
    
    config = GeometricTrainingConfig()
    
    # Test all expected attributes exist
    expected_attrs = [
        'enabled', 'geometric_learning_rate', 'expert_learning_rate', 'rotation_dimensions',
        'orthogonality_weight', 'rotation_efficiency_weight', 'specialization_weight',
        'ghost_geometric_threshold', 'ghost_rotation_dimensions',
        'lambda_cognitive_rotations', 'lambda_rotation_scheduling',
        'rotation_matrix_type', 'rotation_convergence_threshold', 'max_rotation_magnitude'
    ]
    
    for attr in expected_attrs:
        assert hasattr(config, attr), f"Missing attribute: {attr}"
    print("✅ All expected geometric config attributes present")
    
    # Test default values are reasonable
    assert config.enabled == False
    assert config.geometric_learning_rate > 0
    assert config.expert_learning_rate > 0
    assert config.rotation_dimensions > 0
    assert config.lambda_rotation_scheduling in ['curriculum', 'adaptive', 'fixed']
    print("✅ Default geometric config values are reasonable")
    
    print("🎉 Geometric configuration tests passed!\n")


def run_all_tests():
    """Run all integration tests."""
    print("🚀 Running Geometric Training Integration Tests\n")
    
    try:
        test_config_system()
        test_training_controller_factory()
        test_training_controller_interface()
        test_geometric_config_options()
        
        print("🎉🎉🎉 ALL TESTS PASSED! 🎉🎉🎉")
        print("\nThe modular geometric training architecture is ready!")
        print("Users can now:")
        print("  • Configure geometric training via advanced settings")
        print("  • Switch between standard and geometric training modes")
        print("  • Access full geometric parameter customization")
        print("  • Use the training controller factory pattern")
        print("\nNext steps: Implement the actual geometric training logic!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
