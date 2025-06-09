#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_controllers.py

Training controller architecture for MoE models supporting multiple training paradigms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from .config import MoEConfig
from .architecture import create_dynamic_optimizer, PrimaryGhostLRScheduler


class TrainingController(ABC):
    """Abstract base class for all training controllers."""
    
    def __init__(self, model, config: MoEConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Metrics tracking
        self.step_count = 0
        self.metrics_history = {
            'loss': [],
            'learning_rate': [],
            'orthogonality': []
        }
    
    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """Execute one training step and return loss."""
        pass
    
    @abstractmethod
    def get_optimizers(self) -> List[torch.optim.Optimizer]:
        """Return optimizers for this training mode."""
        pass
    
    @abstractmethod
    def get_schedulers(self) -> List[Any]:
        """Return schedulers for this training mode."""
        pass
    
    @abstractmethod
    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current training metrics."""
        pass
    
    def update_metrics(self, loss: float, additional_metrics: Dict[str, Any] = None):
        """Update metrics tracking."""
        self.metrics_history['loss'].append(loss)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
    
    def reset_optimizers(self):
        """Reset optimizer states (useful for dynamic changes)."""
        for optimizer in self.get_optimizers():
            optimizer.zero_grad()


class StandardTrainingController(TrainingController):
    """Standard training controller using traditional gradient descent."""
    
    def __init__(self, model, config: MoEConfig):
        super().__init__(model, config)
        
        # Create standard optimizer and scheduler
        self.optimizer = create_dynamic_optimizer(model, config)
        self.scheduler = PrimaryGhostLRScheduler(config, self.optimizer)
        
        # Additional metrics for standard training
        self.metrics_history.update({
            'expert_entropy': [],
            'ghost_activations': []
        })
    
    def training_step(self, batch: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """Standard training step with backpropagation."""
        self.step_count = step
        
        # Forward pass
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Create targets (shifted input for language modeling)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        # Model forward pass
        outputs = self.model(inputs, step=step, attention_mask=attention_mask[:, :-1])
        
        # Extract logits from model output
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        current_metrics = {
            'learning_rate': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate,
            'orthogonality': self._compute_expert_orthogonality(),
            'expert_entropy': self._compute_expert_entropy(),
            'ghost_activations': self._count_ghost_activations()
        }
        
        self.update_metrics(loss.item(), current_metrics)
        
        return loss
    
    def get_optimizers(self) -> List[torch.optim.Optimizer]:
        """Return the standard optimizer."""
        return [self.optimizer]
    
    def get_schedulers(self) -> List[Any]:
        """Return the standard scheduler."""
        return [self.scheduler]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current standard training metrics."""
        if not self.metrics_history['loss']:
            return {'loss': 0.0, 'learning_rate': self.config.learning_rate}
        
        return {
            'loss': self.metrics_history['loss'][-1],
            'learning_rate': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else self.config.learning_rate,
            'orthogonality': self.metrics_history['orthogonality'][-1] if self.metrics_history['orthogonality'] else 0.0,
            'expert_entropy': self.metrics_history['expert_entropy'][-1] if self.metrics_history['expert_entropy'] else 0.0,
            'ghost_activations': self.metrics_history['ghost_activations'][-1] if self.metrics_history['ghost_activations'] else 0
        }
    
    def _compute_expert_orthogonality(self) -> float:
        """Compute orthogonality metric for experts."""
        if not hasattr(self.model, 'moe_layers'):
            return 0.0
        
        total_orthogonality = 0.0
        num_layers = 0
        
        for layer in self.model.moe_layers:
            if hasattr(layer, 'experts'):
                # Get expert weight matrices
                expert_weights = []
                for expert in layer.experts:
                    if hasattr(expert, 'linear1'):
                        expert_weights.append(expert.linear1.weight.flatten())
                
                if len(expert_weights) > 1:
                    # Compute pairwise orthogonality
                    expert_matrix = torch.stack(expert_weights)
                    normalized_weights = F.normalize(expert_matrix, dim=1)
                    similarity_matrix = torch.mm(normalized_weights, normalized_weights.t())
                    
                    # Off-diagonal elements should be close to 0 for orthogonality
                    mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
                    off_diagonal_mean = similarity_matrix[mask].abs().mean()
                    
                    # Convert to orthogonality score (1 - similarity)
                    orthogonality = 1.0 - off_diagonal_mean.item()
                    total_orthogonality += orthogonality
                    num_layers += 1
        
        return total_orthogonality / num_layers if num_layers > 0 else 0.0
    
    def _compute_expert_entropy(self) -> float:
        """Compute expert activation entropy (diversity measure)."""
        # This would require tracking expert activations during forward pass
        # For now, return a placeholder
        return 0.0
    
    def _count_ghost_activations(self) -> int:
        """Count number of active ghost experts."""
        if not hasattr(self.model, 'ghost_experts_active'):
            return 0
        
        # This would require tracking ghost expert activations
        # For now, return a placeholder
        return 0


class GeometricTrainingController(TrainingController):
    """Geometric Constrained Learning training controller - the revolutionary paradigm."""
    
    def __init__(self, model, config: MoEConfig):
        super().__init__(model, config)
        
        if not config.geometric.enabled:
            raise ValueError("Geometric training controller requires geometric.enabled=True in config")
        
        # Import geometric components
        from .geometric_training import GeometricDataRotator, GeometricLossComputer, LambdaCalculusGeometricRotator
        
        # Initialize geometric components
        if config.geometric.lambda_cognitive_rotations:
            self.data_rotator = LambdaCalculusGeometricRotator(config)
        else:
            self.data_rotator = GeometricDataRotator(config)
        
        self.loss_computer = GeometricLossComputer(config)
        
        # Separate optimizers for geometric vs expert parameters
        self.rotation_optimizer = torch.optim.Adam(
            self.data_rotator.parameters(),
            lr=config.geometric.geometric_learning_rate
        )
        
        self.expert_optimizer = create_dynamic_optimizer(model, config)
        # Scale down expert learning rate (geometric training uses higher LR for rotations)
        for param_group in self.expert_optimizer.param_groups:
            param_group['lr'] = config.geometric.expert_learning_rate
        
        # Schedulers
        self.rotation_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.rotation_optimizer, T_max=config.max_steps if config.max_steps else 1000
        )
        self.expert_scheduler = PrimaryGhostLRScheduler(config, self.expert_optimizer)
        
        # Additional metrics tracking for geometric training
        self.metrics_history.update({
            'rotation_angles': [],
            'rotation_efficiency_loss': [],
            'expert_specialization': [],
            'geometric_loss_components': []
        })
        
        print("🎯 Geometric Constrained Learning activated!")
        print(f"   Rotation dimensions: {config.geometric.rotation_dimensions}")
        print(f"   Geometric LR: {config.geometric.geometric_learning_rate:.2e}")
        print(f"   Expert LR: {config.geometric.expert_learning_rate:.2e}")
    
    def training_step(self, batch: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """Revolutionary geometric training step - optimize data presentation, not just weights."""
        self.step_count = step
        
        # Forward pass with geometric data rotation
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Create targets (shifted input for language modeling)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        # Get input embeddings from model
        with torch.no_grad():
            input_embeddings = self.model.token_emb(inputs)  # [batch, seq, embed_dim]
        
        # THE REVOLUTIONARY STEP: Rotate data presentations for each expert
        rotated_presentations = self.data_rotator.rotate_data_for_experts(input_embeddings)
        
        # Process each expert with its optimal data presentation
        expert_outputs = []
        for expert_idx, rotated_data in enumerate(rotated_presentations):
            # Forward through the specific expert with rotated data
            expert_output = self._forward_expert(expert_idx, rotated_data, attention_mask[:, :-1])
            expert_outputs.append(expert_output)
        
        # Get rotation angles for loss computation
        rotation_angles = self.data_rotator.get_rotation_angles()
        
        # Compute geometric loss
        geometric_loss, loss_components = self.loss_computer.compute_geometric_loss(
            expert_outputs=expert_outputs,
            rotated_data=rotated_presentations,
            targets=targets,
            rotation_angles=rotation_angles,
            model=self.model
        )
        
        # Backward pass and optimization
        geometric_loss.backward()
        
        # Update rotation parameters (higher learning rate)
        self.rotation_optimizer.step()
        self.rotation_scheduler.step()
        self.rotation_optimizer.zero_grad()
        
        # Update expert parameters (lower learning rate)
        self.expert_optimizer.step()
        self.expert_scheduler.step([0])  # Placeholder for ghost activations
        self.expert_optimizer.zero_grad()
        
        # Update metrics
        current_metrics = {
            'learning_rate': self.rotation_optimizer.param_groups[0]['lr'],
            'expert_learning_rate': self.expert_optimizer.param_groups[0]['lr'],
            'rotation_angles': rotation_angles.detach().cpu().numpy().tolist(),
            'rotation_efficiency_loss': loss_components.get('rotation_efficiency_loss', 0.0),
            'expert_specialization': loss_components.get('specialization_loss', 0.0),
            'geometric_loss_components': loss_components
        }
        
        self.update_metrics(geometric_loss.item(), current_metrics)
        
        return geometric_loss
    
    def _forward_expert(self, expert_idx: int, rotated_data: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through a specific expert with its optimally rotated data."""
        # Apply position embeddings to the rotated data
        batch_size, seq_len, embed_dim = rotated_data.shape
        pos_ids = torch.arange(seq_len, device=rotated_data.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.model.pos_emb(pos_ids)
        
        # Combine rotated token embeddings with position embeddings
        x = rotated_data + pos_emb
        x = self.model.dropout(x)
        
        # Create causal mask
        causal_mask = self.model.create_causal_mask(seq_len, rotated_data.device)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        
        # Forward through model layers with step parameter
        for layer in self.model.model_layers:
            x = layer(x, step=self.step_count, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        # Apply output normalization and head
        x = self.model.output_norm(x)
        logits = self.model.lm_head(x)
        
        return logits
    
    def get_optimizers(self) -> List[torch.optim.Optimizer]:
        """Return both rotation and expert optimizers."""
        return [self.rotation_optimizer, self.expert_optimizer]
    
    def get_schedulers(self) -> List[Any]:
        """Return both rotation and expert schedulers."""
        return [self.rotation_scheduler, self.expert_scheduler]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current geometric training metrics."""
        if not self.metrics_history['loss']:
            return {
                'loss': 0.0, 
                'learning_rate': self.config.geometric.geometric_learning_rate,
                'training_mode': 'geometric'
            }
        
        return {
            'loss': self.metrics_history['loss'][-1],
            'learning_rate': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else self.config.geometric.geometric_learning_rate,
            'expert_learning_rate': self.metrics_history.get('expert_learning_rate', [self.config.geometric.expert_learning_rate])[-1],
            'rotation_angles': self.metrics_history['rotation_angles'][-1] if self.metrics_history['rotation_angles'] else [],
            'rotation_efficiency': self.metrics_history['rotation_efficiency_loss'][-1] if self.metrics_history['rotation_efficiency_loss'] else 0.0,
            'expert_specialization': self.metrics_history['expert_specialization'][-1] if self.metrics_history['expert_specialization'] else 0.0,
            'geometric_components': self.metrics_history['geometric_loss_components'][-1] if self.metrics_history['geometric_loss_components'] else {},
            'training_mode': 'geometric'
        }


def create_training_controller(model, config: MoEConfig) -> TrainingController:
    """Factory function to create appropriate training controller."""
    
    if config.training_mode == "geometric":
        if not config.geometric.enabled:
            print("⚠️  Geometric training mode selected but not enabled in config. Using standard training.")
            return StandardTrainingController(model, config)
        return GeometricTrainingController(model, config)
    
    elif config.training_mode == "standard":
        return StandardTrainingController(model, config)
    
    else:
        raise ValueError(f"Unknown training_mode: '{config.training_mode}'. Must be 'standard' or 'geometric'.")
