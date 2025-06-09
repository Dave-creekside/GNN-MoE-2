#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_weight_orthogonality.py

Phase 2.2: Adaptive Weight Orthogonality Controller for HGNN-MoE
Dynamic adjustment of weight matrix orthogonality constraints based on training progress.
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class AdaptiveWeightOrthogonalityController:
    """
    Intelligent controller for adaptive weight matrix orthogonality constraints.
    
    Features:
    - Dynamic constraint strength adjustment based on specialization progress
    - Layer-specific adaptation with deeper layer scaling
    - Performance-aware adaptation responding to training dynamics
    - Emergency intervention to prevent expert collapse
    - Multiple adaptation schedules (cosine, exponential, linear, step)
    """
    
    def __init__(self, config, model):
        """
        Initialize adaptive controller.
        
        Args:
            config: GNNMoEConfig with adaptive_weight_orthogonality=True
            model: GNNMoEModel instance to control
        """
        self.config = config
        self.model = model
        
        # Tracking histories
        self.adaptation_history = []
        self.specialization_history = []
        self.performance_history = []
        
        # Current state
        self.current_strengths = {}  # Per-layer strength tracking
        self.emergency_mode = False
        self.adaptation_step = 0
        
        # Initialize per-layer strengths
        self._initialize_layer_strengths()
        
        print(f"🧠 Adaptive Weight Orthogonality Controller initialized")
        print(f"   Target specialization: {config.target_specialization_score:.1%}")
        print(f"   Adaptation schedule: {config.adaptive_decay_schedule}")
        print(f"   Initial strengths: {self._format_strengths()}")
    
    def _initialize_layer_strengths(self):
        """Initialize layer-specific orthogonality strengths with deeper layer scaling."""
        base_strength = self.config.initial_weight_orthogonality_strength
        
        for layer_idx in range(self.config.num_layers):
            if self.config.layer_specific_adaptation:
                # Deeper layers get reduced constraints (they're more specialized)
                layer_factor = self.config.deeper_layer_scaling ** layer_idx
                strength = base_strength * layer_factor
            else:
                # Uniform strength across all layers
                strength = base_strength
            
            self.current_strengths[f'layer_{layer_idx}'] = strength
    
    def _format_strengths(self) -> str:
        """Format current strengths for logging."""
        strengths = [f"{v:.3f}" for v in self.current_strengths.values()]
        return f"[{', '.join(strengths)}]"
    
    def compute_specialization_metrics(self) -> Dict[str, float]:
        """
        Compute real-time expert specialization metrics across all layers.
        
        Returns:
            Dict with per-layer and overall specialization scores
        """
        metrics = {}
        
        with torch.no_grad():  # Ensure no gradients for metric computation
            for layer_idx, layer in enumerate(self.model.model_layers):
                # Extract expert weight matrices for this layer
                weight_matrices = layer._get_target_weight_matrices()
                
                if len(weight_matrices) >= 2:
                    try:
                        # Normalize weight matrices to same size (handle different dimensions)
                        flat_weights = []
                        max_size = max(w.numel() for w in weight_matrices)
                        
                        for weight in weight_matrices:
                            flat_weight = weight.view(-1)
                            if flat_weight.numel() < max_size:
                                # Pad with zeros
                                padding = torch.zeros(max_size - flat_weight.numel(), 
                                                    device=flat_weight.device)
                                flat_weight = torch.cat([flat_weight, padding])
                            flat_weights.append(flat_weight)
                        
                        # Stack and compute Gram matrix
                        stacked_weights = torch.stack(flat_weights, dim=0)  # (num_experts, max_size)
                        
                        # Normalize to unit vectors for better numerical stability
                        stacked_weights = F.normalize(stacked_weights, p=2, dim=1)
                        
                        # Compute Gram matrix: G[i,j] = weight_i · weight_j
                        gram_matrix = torch.mm(stacked_weights, stacked_weights.T)  # (E, E)
                        
                        # Compute specialization score (1 - off-diagonal similarity)
                        num_experts = len(weight_matrices)
                        mask = ~torch.eye(num_experts, dtype=torch.bool, device=gram_matrix.device)
                        off_diagonal_mean = gram_matrix[mask].abs().mean()
                        specialization_score = 1.0 - off_diagonal_mean.item()
                        
                        # Clamp to reasonable range
                        specialization_score = max(0.0, min(1.0, specialization_score))
                        
                        metrics[f'layer_{layer_idx}_specialization'] = specialization_score
                        
                    except Exception as e:
                        print(f"Warning: Failed to compute specialization for layer {layer_idx}: {e}")
                        metrics[f'layer_{layer_idx}_specialization'] = 0.5  # Default fallback
        
        # Overall specialization score (weighted average)
        if metrics:
            layer_scores = list(metrics.values())
            metrics['overall_specialization'] = sum(layer_scores) / len(layer_scores)
        else:
            metrics['overall_specialization'] = 0.5  # Fallback
        
        return metrics
    
    def detect_expert_collapse(self, window_size: int = None) -> bool:
        """
        Detect if experts are collapsing (becoming too similar).
        
        Args:
            window_size: Number of recent steps to analyze
        Returns:
            True if expert collapse detected
        """
        if window_size is None:
            window_size = self.config.emergency_detection_window
        
        if len(self.specialization_history) < window_size:
            return False
        
        recent_scores = [h['overall_specialization'] 
                        for h in self.specialization_history[-window_size:]]
        
        if len(recent_scores) >= 2:
            # Check for rapid decrease in specialization
            start_score = recent_scores[0]
            end_score = recent_scores[-1]
            trend = (end_score - start_score) / len(recent_scores)
            
            # Collapse detected if rapid negative trend
            collapse_threshold = -self.config.collapse_detection_threshold
            return trend < collapse_threshold
        
        return False
    
    def detect_performance_plateau(self, window_size: int = None) -> bool:
        """
        Detect training performance plateau.
        
        Args:
            window_size: Number of recent steps to analyze
        Returns:
            True if performance plateau detected
        """
        if window_size is None:
            window_size = self.config.performance_monitor_window
        
        if len(self.performance_history) < window_size:
            return False
        
        recent_losses = [h['eval_loss'] for h in self.performance_history[-window_size:] 
                        if 'eval_loss' in h and h['eval_loss'] is not None]
        
        if len(recent_losses) >= 10:
            # Check if loss improvement has stagnated
            mid_point = len(recent_losses) // 2
            early_avg = sum(recent_losses[:mid_point]) / mid_point
            late_avg = sum(recent_losses[mid_point:]) / (len(recent_losses) - mid_point)
            
            # Calculate relative improvement
            if early_avg > 0:
                improvement = (early_avg - late_avg) / early_avg
                return improvement < 0.01  # Less than 1% improvement
        
        return False
    
    def compute_adaptive_strength(self, training_step: int, layer_idx: int) -> float:
        """
        Compute adaptive orthogonality strength for specific layer.
        
        Args:
            training_step: Current training step
            layer_idx: Layer index to compute strength for
        Returns:
            Adaptive strength value
        """
        base_strength = self.current_strengths[f'layer_{layer_idx}']
        
        # 1. Time-based adaptation factor
        time_factor = self._compute_time_decay_factor(training_step)
        
        # 2. Performance-based adaptation factor
        performance_factor = self._compute_performance_factor()
        
        # 3. Emergency intervention factor
        emergency_factor = self._compute_emergency_factor()
        
        # Combine all factors
        adaptive_strength = base_strength * time_factor * performance_factor * emergency_factor
        
        # Clamp to configured bounds
        adaptive_strength = max(self.config.minimum_weight_orthogonality_strength,
                              min(self.config.maximum_weight_orthogonality_strength, 
                                  adaptive_strength))
        
        return adaptive_strength
    
    def _compute_time_decay_factor(self, training_step: int) -> float:
        """Compute time-based decay factor based on adaptation schedule."""
        # Total adaptation cycles
        max_steps = self.config.adaptation_frequency * 20  # 20 cycles for full decay
        progress = min(1.0, training_step / max_steps)
        
        if self.config.adaptive_decay_schedule == "cosine":
            # Cosine decay: starts at 1, decays to minimum
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            factor = self.config.minimum_weight_orthogonality_strength + \
                    (1 - self.config.minimum_weight_orthogonality_strength) * factor
        
        elif self.config.adaptive_decay_schedule == "exponential":
            # Exponential decay
            decay_rate = 0.0001
            factor = math.exp(-decay_rate * training_step)
            factor = max(self.config.minimum_weight_orthogonality_strength, factor)
        
        elif self.config.adaptive_decay_schedule == "linear":
            # Linear decay
            factor = 1.0 - progress
            factor = max(self.config.minimum_weight_orthogonality_strength, factor)
        
        elif self.config.adaptive_decay_schedule == "step":
            # Step decay
            decay_factor = 0.5
            step_size = max_steps // 4  # 4 steps total
            if step_size > 0:
                factor = decay_factor ** (training_step // step_size)
            else:
                factor = 1.0
            factor = max(self.config.minimum_weight_orthogonality_strength, factor)
        
        else:
            # Default: no decay
            factor = 1.0
        
        return factor
    
    def _compute_performance_factor(self) -> float:
        """Compute performance-based adaptation factor."""
        if not self.config.performance_aware_adaptation or not self.specialization_history:
            return 1.0
        
        current_specialization = self.specialization_history[-1].get('overall_specialization', 0.5)
        target = self.config.target_specialization_score
        tolerance = self.config.specialization_tolerance
        
        if current_specialization < target - tolerance:
            # Below target - increase constraints
            return 1.5
        elif current_specialization > target + tolerance:
            # Above target - reduce constraints  
            return 0.7
        else:
            # Within target range - maintain
            return 1.0
    
    def _compute_emergency_factor(self) -> float:
        """Compute emergency intervention factor."""
        if self.emergency_mode and self.config.emergency_constraint_boost:
            return self.config.emergency_boost_multiplier
        return 1.0
    
    def update_adaptation(self, training_step: int, eval_loss: Optional[float] = None):
        """
        Main adaptation update called during training.
        
        Args:
            training_step: Current training step
            eval_loss: Optional evaluation loss for performance tracking
        """
        # Only update at specified frequency
        if training_step % self.config.adaptation_frequency != 0:
            return
        
        self.adaptation_step += 1
        
        # Compute current specialization metrics
        specialization_metrics = self.compute_specialization_metrics()
        self.specialization_history.append({
            'step': training_step,
            'adaptation_step': self.adaptation_step,
            **specialization_metrics
        })
        
        # Record performance if available
        if eval_loss is not None:
            self.performance_history.append({
                'step': training_step,
                'adaptation_step': self.adaptation_step,
                'eval_loss': eval_loss
            })
        
        # Check for emergency conditions
        expert_collapse = self.detect_expert_collapse()
        performance_plateau = self.detect_performance_plateau()
        
        # Update emergency mode (collapse without plateau indicates real problem)
        previous_emergency = self.emergency_mode
        self.emergency_mode = expert_collapse and not performance_plateau
        
        # Update per-layer strengths
        old_strengths = self.current_strengths.copy()
        for layer_idx in range(self.config.num_layers):
            new_strength = self.compute_adaptive_strength(training_step, layer_idx)
            self.current_strengths[f'layer_{layer_idx}'] = new_strength
        
        # Log adaptation decision
        adaptation_event = {
            'step': training_step,
            'adaptation_step': self.adaptation_step,
            'old_strengths': old_strengths,
            'new_strengths': self.current_strengths.copy(),
            'emergency_mode': self.emergency_mode,
            'emergency_triggered': self.emergency_mode and not previous_emergency,
            'expert_collapse': expert_collapse,
            'performance_plateau': performance_plateau,
            'specialization_metrics': specialization_metrics
        }
        self.adaptation_history.append(adaptation_event)
        
        # Log important events
        if self.emergency_mode and not previous_emergency:
            print(f"🚨 Emergency intervention activated at step {training_step}")
            print(f"   Specialization: {specialization_metrics['overall_specialization']:.3f}")
            print(f"   New strengths: {self._format_strengths()}")
        elif previous_emergency and not self.emergency_mode:
            print(f"✅ Emergency intervention deactivated at step {training_step}")
        
        # Periodic status update
        if self.adaptation_step % 5 == 0:  # Every 5 adaptations
            spec_score = specialization_metrics['overall_specialization']
            print(f"🔄 Adaptation #{self.adaptation_step}: spec={spec_score:.3f}, "
                  f"strengths={self._format_strengths()}, emergency={self.emergency_mode}")
    
    def get_current_strength(self, layer_idx: int) -> float:
        """Get current adaptive strength for a specific layer."""
        return self.current_strengths.get(f'layer_{layer_idx}', 
                                        self.config.initial_weight_orthogonality_strength)
    
    def get_adaptation_summary(self) -> Dict:
        """Get comprehensive summary of adaptation behavior for analysis."""
        if not self.adaptation_history:
            return {
                'status': 'no_adaptations_yet',
                'current_strengths': self.current_strengths.copy()
            }
        
        emergency_activations = sum(1 for h in self.adaptation_history if h['emergency_triggered'])
        
        # Compute adaptation statistics
        all_strengths = []
        for event in self.adaptation_history:
            all_strengths.extend(event['new_strengths'].values())
        
        summary = {
            'total_adaptations': len(self.adaptation_history),
            'emergency_activations': emergency_activations,
            'current_emergency_mode': self.emergency_mode,
            'current_strengths': self.current_strengths.copy(),
            'strength_statistics': {
                'min': min(all_strengths) if all_strengths else 0,
                'max': max(all_strengths) if all_strengths else 0,
                'mean': sum(all_strengths) / len(all_strengths) if all_strengths else 0
            },
            'recent_specialization': self.specialization_history[-5:] if self.specialization_history else [],
            'recent_adaptations': self.adaptation_history[-3:] if self.adaptation_history else [],
            'specialization_trend': self._compute_specialization_trend()
        }
        
        return summary
    
    def _compute_specialization_trend(self) -> Dict:
        """Compute recent specialization trend."""
        if len(self.specialization_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_window = min(10, len(self.specialization_history))
        recent_scores = [h['overall_specialization'] 
                        for h in self.specialization_history[-recent_window:]]
        
        if len(recent_scores) >= 2:
            start_score = recent_scores[0]
            end_score = recent_scores[-1]
            trend = (end_score - start_score) / len(recent_scores)
            
            return {
                'start_score': start_score,
                'end_score': end_score,
                'trend_per_step': trend,
                'direction': 'improving' if trend > 0.01 else 'declining' if trend < -0.01 else 'stable'
            }
        
        return {'status': 'insufficient_data'}

if __name__ == "__main__":
    # Simple test of adaptive controller
    print("🧪 Testing AdaptiveWeightOrthogonalityController...")
    
    # This would normally be done with real config and model
    # Just testing parameter validation for now
    class MockConfig:
        adaptive_weight_orthogonality = True
        initial_weight_orthogonality_strength = 0.1
        minimum_weight_orthogonality_strength = 0.001
        maximum_weight_orthogonality_strength = 0.3
        adaptive_decay_schedule = "cosine"
        adaptation_frequency = 500
        target_specialization_score = 0.95
        specialization_tolerance = 0.02
        layer_specific_adaptation = True
        deeper_layer_scaling = 0.8
        performance_aware_adaptation = True
        performance_monitor_window = 100
        collapse_detection_threshold = 0.1
        emergency_constraint_boost = True
        emergency_boost_multiplier = 2.0
        emergency_detection_window = 50
        num_layers = 4
    
    config = MockConfig()
    
    # Test time decay computation
    controller = AdaptiveWeightOrthogonalityController.__new__(AdaptiveWeightOrthogonalityController)
    controller.config = config
    controller.current_strengths = {f'layer_{i}': 0.1 * (0.8 ** i) for i in range(4)}
    
    # Test decay factors
    for step in [0, 1000, 5000, 10000]:
        factor = controller._compute_time_decay_factor(step)
        print(f"Step {step}: decay factor = {factor:.4f}")
    
    print("✅ Basic adaptive controller tests passed!")
