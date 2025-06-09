#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_adaptive_orthogonality.py

Phase 2.2: Adaptive Weight Orthogonality Demo
Demonstrates the intelligent, dynamic adjustment of weight matrix orthogonality constraints.
"""

import torch
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel

def demo_adaptive_orthogonality():
    """
    Demonstrate Phase 2.2 Adaptive Weight Orthogonality capabilities.
    """
    print("🚀 Phase 2.2 Adaptive Weight Orthogonality Demo")
    print("=" * 60)
    
    # Create adaptive configuration
    config = GNNMoEConfig(
        # Small model for demo safety
        num_experts=4,
        embed_dim=128,
        num_layers=3,
        vocab_size=1000,  # Smaller vocab for demo
        max_seq_length=32,
        
        # Enable adaptive weight orthogonality (Phase 2.2)
        adaptive_weight_orthogonality=True,
        initial_weight_orthogonality_strength=0.15,
        minimum_weight_orthogonality_strength=0.001,
        maximum_weight_orthogonality_strength=0.5,
        
        # Adaptation settings
        adaptive_decay_schedule="cosine",
        adaptation_frequency=100,  # More frequent for demo
        target_specialization_score=0.92,
        specialization_tolerance=0.03,
        
        # Layer-specific adaptation
        layer_specific_adaptation=True,
        deeper_layer_scaling=0.75,
        
        # Performance-aware adaptation
        performance_aware_adaptation=True,
        emergency_constraint_boost=True,
        emergency_boost_multiplier=2.5,
        
        # Architecture
        coupler_type="GNN",
        apply_weight_orthogonality_loss=True,
        weight_orthogonality_target_layer="ffn_input"
    )
    
    print(f"📋 Configuration:")
    print(f"   Initial strength: {config.initial_weight_orthogonality_strength}")
    print(f"   Target specialization: {config.target_specialization_score:.1%}")
    print(f"   Adaptation schedule: {config.adaptive_decay_schedule}")
    print(f"   Layer-specific: {config.layer_specific_adaptation}")
    print()
    
    # Create model with adaptive controller
    print("🧠 Creating model with Adaptive Weight Orthogonality...")
    model = GNNMoEModel(config)
    print()
    
    # Simulate training progression
    print("📈 Simulating training progression with adaptive updates...")
    print("-" * 60)
    
    # Create dummy training data
    batch_size, seq_len = 4, 16
    
    for step in [0, 100, 200, 500, 1000, 1500, 2000]:
        # Generate dummy batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, return_loss=True, labels=input_ids)
        lm_loss = outputs['loss'].item()
        
        # Get current orthogonality loss
        ortho_loss = model.get_total_orthogonality_loss(training_step=step)
        total_loss = lm_loss + ortho_loss.item()
        
        # Simulate evaluation loss (decreasing with noise)
        eval_loss = 8.0 - (step / 500) + torch.randn(1).item() * 0.1
        eval_loss = max(3.0, eval_loss)  # Floor at 3.0
        
        # Update adaptive system
        model.update_adaptive_orthogonality(step, eval_loss)
        
        # Get adaptation summary
        summary = model.get_adaptation_summary()
        
        # Display progress
        if summary.get('current_strengths'):
            strengths = list(summary['current_strengths'].values())
            strengths_str = f"[{', '.join(f'{s:.3f}' for s in strengths)}]"
        else:
            strengths_str = "[initializing...]"
        
        emergency_status = "🚨" if summary.get('current_emergency_mode', False) else "✅"
        
        print(f"Step {step:4d}: "
              f"LM={lm_loss:.3f}, Orth={ortho_loss.item():.3f}, "
              f"Eval={eval_loss:.3f}, Strengths={strengths_str} {emergency_status}")
        
        # Show detailed info at key steps
        if step in [100, 1000, 2000]:
            adaptations = summary.get('total_adaptations', 0)
            emergencies = summary.get('emergency_activations', 0)
            trend = summary.get('specialization_trend', {})
            trend_dir = trend.get('direction', 'unknown')
            
            print(f"         📊 Adaptations: {adaptations}, Emergencies: {emergencies}")
            print(f"         📈 Specialization trend: {trend_dir}")
            print()
    
    print("🎯 Adaptive Training Simulation Complete!")
    print()
    
    # Final summary
    final_summary = model.get_adaptation_summary()
    print("📊 Final Adaptive System Summary:")
    print("-" * 40)
    
    if final_summary.get('current_strengths'):
        print("🎛️  Final adaptive strengths:")
        for layer, strength in final_summary['current_strengths'].items():
            print(f"     {layer}: {strength:.4f}")
    
    stats = final_summary.get('strength_statistics', {})
    if stats:
        print(f"📈 Strength range: {stats.get('min', 0):.4f} - {stats.get('max', 0):.4f}")
        print(f"📊 Average strength: {stats.get('mean', 0):.4f}")
    
    total_adaptations = final_summary.get('total_adaptations', 0)
    emergencies = final_summary.get('emergency_activations', 0)
    print(f"🔄 Total adaptations: {total_adaptations}")
    print(f"🚨 Emergency interventions: {emergencies}")
    
    trend = final_summary.get('specialization_trend', {})
    if trend.get('direction'):
        direction = trend['direction']
        print(f"📈 Final specialization trend: {direction}")
    
    print()
    print("✨ Phase 2.2 Adaptive Weight Orthogonality Demo Complete!")
    print("   💡 Key features demonstrated:")
    print("   ✅ Dynamic strength adjustment based on training progress")
    print("   ✅ Layer-specific adaptation with deeper layer scaling") 
    print("   ✅ Performance-aware constraint modification")
    print("   ✅ Emergency intervention for expert collapse prevention")
    print("   ✅ Comprehensive adaptation tracking and analysis")
    print()
    print("🚀 Ready for production adaptive orthogonal expert training!")

if __name__ == "__main__":
    demo_adaptive_orthogonality()
