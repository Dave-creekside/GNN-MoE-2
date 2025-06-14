#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_architecture_training_combinations.py

Comprehensive test suite to verify all 8 combinations of:
- 4 Architectures: GNN, HGNN, Orthogonal, Ghost  
- 2 Training Modes: Standard, Geometric

This ensures the architecture/training mode separation is working correctly.
"""

import torch
import pytest
from core.config import MoEConfig, GhostParams, GeometricTrainingConfig
from core.architecture import MoEModel
from core.training_controllers import create_training_controller, StandardTrainingController, GeometricTrainingController

def create_test_config(architecture_mode: str, training_mode: str) -> MoEConfig:
    """Create a minimal test configuration."""
    # Create proper dataclass objects
    ghost_params = GhostParams(
        num_ghost_experts=1 if architecture_mode == 'ghost' else 0
    )
    
    geometric_params = GeometricTrainingConfig(
        enabled=training_mode == 'geometric',
        geometric_learning_rate=1e-3,
        expert_learning_rate=1e-4,
        rotation_dimensions=2
    )
    
    config = MoEConfig(
        run_name=f"test_{architecture_mode}_{training_mode}",
        architecture_mode=architecture_mode,
        training_mode=training_mode,
        
        # Minimal settings for fast testing
        vocab_size=1000,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        num_experts=2,
        max_seq_length=32,
        batch_size=2,
        epochs=1,
        learning_rate=1e-4,
        
        # Proper dataclass objects
        ghost=ghost_params,
        geometric=geometric_params
    )
    return config

def create_test_batch(config: MoEConfig):
    """Create a minimal test batch."""
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_length)),
        'attention_mask': torch.ones(config.batch_size, config.max_seq_length)
    }
    return batch

class TestArchitectureTrainingCombinations:
    """Test all 8 combinations of architecture × training mode."""
    
    @pytest.mark.parametrize("architecture_mode", ["gnn", "hgnn", "orthogonal", "ghost"])
    @pytest.mark.parametrize("training_mode", ["standard", "geometric"])
    def test_config_creation(self, architecture_mode, training_mode):
        """Test that all configuration combinations are valid."""
        config = create_test_config(architecture_mode, training_mode)
        
        # Verify config creation succeeds
        assert config.architecture_mode == architecture_mode
        assert config.training_mode == training_mode
        
        # Verify __post_init__ sets correct flags
        if architecture_mode == 'gnn':
            assert not config.use_hypergraph_coupling
            assert not config.use_orthogonal_loss
            assert config.ghost.num_ghost_experts == 0
        elif architecture_mode == 'hgnn':
            assert config.use_hypergraph_coupling
            assert not config.use_orthogonal_loss
            assert config.ghost.num_ghost_experts == 0
        elif architecture_mode == 'orthogonal':
            assert config.use_hypergraph_coupling
            assert config.use_orthogonal_loss
            assert config.ghost.num_ghost_experts == 0
        elif architecture_mode == 'ghost':
            assert config.use_hypergraph_coupling
            assert config.use_orthogonal_loss
            assert config.ghost.num_ghost_experts > 0
        
        # Verify geometric training enabled correctly
        if training_mode == 'geometric':
            assert config.geometric.enabled
        
        print(f"✅ Config: {architecture_mode} + {training_mode}")

    @pytest.mark.parametrize("architecture_mode", ["gnn", "hgnn", "orthogonal", "ghost"])
    @pytest.mark.parametrize("training_mode", ["standard", "geometric"])
    def test_model_creation(self, architecture_mode, training_mode):
        """Test that models can be created for all combinations."""
        config = create_test_config(architecture_mode, training_mode)
        
        # Create model
        model = MoEModel(config)
        
        # Verify model creation succeeds
        assert model is not None
        assert hasattr(model, 'model_layers')
        assert len(model.model_layers) == config.num_layers
        
        # Verify architecture-specific features
        layer = model.model_layers[0]
        
        if config.use_hypergraph_coupling:
            assert hasattr(layer, 'coupler')
            assert layer.coupler is not None
        
        if config.ghost.num_ghost_experts > 0:
            assert hasattr(layer, 'ghost_experts')
            assert layer.ghost_experts is not None
            assert len(layer.ghost_experts) == config.ghost.num_ghost_experts
        
        print(f"✅ Model: {architecture_mode} + {training_mode}")

    @pytest.mark.parametrize("architecture_mode", ["gnn", "hgnn", "orthogonal", "ghost"])
    @pytest.mark.parametrize("training_mode", ["standard", "geometric"])
    def test_training_controller_creation(self, architecture_mode, training_mode):
        """Test that training controllers can be created for all combinations."""
        config = create_test_config(architecture_mode, training_mode)
        model = MoEModel(config)
        
        # Create training controller
        controller = create_training_controller(model, config)
        
        # Verify correct controller type
        if training_mode == "standard":
            assert isinstance(controller, StandardTrainingController)
        elif training_mode == "geometric":
            assert isinstance(controller, GeometricTrainingController)
        
        # Verify controller has required methods
        assert hasattr(controller, 'training_step')
        assert hasattr(controller, 'get_optimizers')
        assert hasattr(controller, 'get_schedulers')
        assert hasattr(controller, 'get_current_metrics')
        
        print(f"✅ Controller: {architecture_mode} + {training_mode}")

    @pytest.mark.parametrize("architecture_mode", ["gnn", "hgnn", "orthogonal", "ghost"])
    @pytest.mark.parametrize("training_mode", ["standard", "geometric"])
    def test_training_step_execution(self, architecture_mode, training_mode):
        """Test that training steps can execute for all combinations."""
        config = create_test_config(architecture_mode, training_mode)
        model = MoEModel(config)
        controller = create_training_controller(model, config)
        batch = create_test_batch(config)
        
        # Execute training step
        try:
            loss = controller.training_step(batch, step=0)
            
            # Verify training step succeeds
            assert loss is not None
            assert torch.is_tensor(loss)
            assert loss.numel() == 1  # Scalar loss
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            
            print(f"✅ Training Step: {architecture_mode} + {training_mode} - Loss: {loss.item():.4f}")
            
        except Exception as e:
            pytest.fail(f"Training step failed for {architecture_mode} + {training_mode}: {str(e)}")

    @pytest.mark.parametrize("architecture_mode", ["gnn", "hgnn", "orthogonal", "ghost"])
    @pytest.mark.parametrize("training_mode", ["standard", "geometric"])
    def test_metrics_extraction(self, architecture_mode, training_mode):
        """Test that metrics can be extracted for all combinations."""
        config = create_test_config(architecture_mode, training_mode)
        model = MoEModel(config)
        controller = create_training_controller(model, config)
        batch = create_test_batch(config)
        
        # Execute training step to populate metrics
        controller.training_step(batch, step=0)
        
        # Get metrics
        metrics = controller.get_current_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'learning_rate' in metrics
        
        # Verify architecture-specific metrics
        if config.use_orthogonal_loss:
            # Should have orthogonality metrics (not necessarily in current_metrics but should compute)
            if hasattr(controller, '_compute_expert_orthogonality'):
                ortho_score = controller._compute_expert_orthogonality()
                assert isinstance(ortho_score, float)
                assert 0.0 <= ortho_score <= 1.0
        
        if config.ghost.num_ghost_experts > 0:
            # Should have ghost metrics
            if hasattr(controller, '_count_ghost_activations'):
                ghost_count = controller._count_ghost_activations()
                assert isinstance(ghost_count, int)
                assert 0 <= ghost_count <= config.ghost.num_ghost_experts
        
        print(f"✅ Metrics: {architecture_mode} + {training_mode} - Keys: {list(metrics.keys())}")

def run_comprehensive_test():
    """Run all tests for all combinations."""
    architectures = ["gnn", "hgnn", "orthogonal", "ghost"]
    training_modes = ["standard", "geometric"]
    
    print("🧪 COMPREHENSIVE ARCHITECTURE + TRAINING MODE TEST")
    print("=" * 60)
    
    test_instance = TestArchitectureTrainingCombinations()
    
    for arch in architectures:
        for training in training_modes:
            print(f"\n🔬 Testing: {arch.upper()} + {training.upper()}")
            print("-" * 40)
            
            try:
                # Run all tests for this combination
                test_instance.test_config_creation(arch, training)
                test_instance.test_model_creation(arch, training)
                test_instance.test_training_controller_creation(arch, training)
                test_instance.test_training_step_execution(arch, training)
                test_instance.test_metrics_extraction(arch, training)
                
                print(f"✅ ALL TESTS PASSED: {arch} + {training}")
                
            except Exception as e:
                print(f"❌ TEST FAILED: {arch} + {training} - {str(e)}")
                raise

def test_specific_combination(architecture: str, training: str):
    """Test a specific architecture + training combination."""
    print(f"\n🎯 FOCUSED TEST: {architecture.upper()} + {training.upper()}")
    print("=" * 50)
    
    config = create_test_config(architecture, training)
    print(f"📋 Config created: {config.architecture_mode} + {config.training_mode}")
    
    model = MoEModel(config)
    print(f"🧠 Model created: {model.get_total_params()} parameters")
    
    controller = create_training_controller(model, config)
    print(f"🎮 Controller created: {type(controller).__name__}")
    
    batch = create_test_batch(config)
    loss = controller.training_step(batch, step=0)
    print(f"🔥 Training step: Loss = {loss.item():.4f}")
    
    metrics = controller.get_current_metrics()
    print(f"📊 Metrics extracted: {len(metrics)} metrics")
    
    print("✅ FOCUSED TEST PASSED!")

if __name__ == "__main__":
    # Run comprehensive test of all combinations
    run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("🎉 ALL 8 COMBINATIONS TESTED SUCCESSFULLY!")
    print("   - 4 Architectures: GNN, HGNN, Orthogonal, Ghost")
    print("   - 2 Training Modes: Standard, Geometric")
    print("   - Total: 8 combinations verified")
    print("=" * 60)
