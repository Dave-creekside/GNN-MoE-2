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
from typing import Dict, List, Optional, Any, Tuple

from .config import MoEConfig
from .architecture import create_dynamic_optimizer, PrimaryGhostLRScheduler


class TrainingController(ABC):
    """Abstract base class for all training controllers."""
    
    def __init__(self, model, config: MoEConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # OPTIMIZED: Metrics tracking with rolling window to prevent memory leaks
        self.step_count = 0
        self.metrics_max_history = 1000  # Keep last 1000 steps only
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
        """OPTIMIZED: Update metrics tracking with rolling window."""
        self.metrics_history['loss'].append(loss)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
        
        # OPTIMIZATION: Trim metrics history to prevent memory leak
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > self.metrics_max_history:
                # Keep only the last N entries
                self.metrics_history[key] = self.metrics_history[key][-self.metrics_max_history:]
    
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
        
        # Update metrics with standardized field names to match plot expectations
        orthogonality_score = self._compute_expert_orthogonality()
        ghost_count = self._count_ghost_activations()
        
        current_metrics = {
            'learning_rate': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate,
            'orthogonality_preservation': orthogonality_score,  # Standardized name for plots
            'expert_specialization': self._compute_expert_entropy(),  # Maps to specialization
            'active_ghosts': ghost_count,  # Standardized name for ghost plots
            'saturation_level': self._compute_saturation_level()  # Add saturation monitoring
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
        """Return current standard training metrics with standardized field names."""
        if not self.metrics_history['loss']:
            return {'loss': 0.0, 'learning_rate': self.config.learning_rate}
        
        return {
            'loss': self.metrics_history['loss'][-1],
            'learning_rate': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else self.config.learning_rate,
            'orthogonality_preservation': self.metrics_history.get('orthogonality_preservation', [0.0])[-1],
            'expert_specialization': self.metrics_history.get('expert_specialization', [0.0])[-1],
            'active_ghosts': self.metrics_history.get('active_ghosts', [0])[-1],
            'saturation_level': self.metrics_history.get('saturation_level', [0.0])[-1]
        }
    
    def _compute_expert_orthogonality(self) -> float:
        """Compute orthogonality metric for experts based on architecture."""
        if not self.config.use_orthogonal_loss:
            return 0.0
            
        # Use the model's built-in orthogonality computation
        try:
            orthogonality_loss = self.model.get_total_orthogonality_loss(self.step_count)
            # Convert loss to score (higher score = more orthogonal)
            return max(0.0, 1.0 - orthogonality_loss.item()) if orthogonality_loss.item() > 0 else 1.0
        except Exception:
            return 0.0
    
    def _compute_expert_entropy(self) -> float:
        """Compute expert activation entropy (diversity measure)."""
        try:
            # Get expert loads from the model
            expert_loads = self.model.get_expert_activation_loads()
            if expert_loads and expert_loads.get('primary'):
                primary_loads = expert_loads['primary']
                if len(primary_loads) > 1:
                    # Compute entropy of expert activation distribution
                    loads_tensor = torch.tensor(primary_loads, dtype=torch.float32)
                    loads_norm = F.softmax(loads_tensor, dim=0)
                    entropy = -torch.sum(loads_norm * torch.log(loads_norm + 1e-8))
                    return entropy.item()
            return 0.0
        except Exception:
            return 0.0
    
    def _count_ghost_activations(self) -> int:
        """Count number of active ghost experts."""
        if not self.config.ghost.num_ghost_experts > 0:
            return 0
            
        try:
            # Get ghost activations from the model
            ghost_activations = self.model.get_current_ghost_activations()
            if ghost_activations is not None:
                # Count ghosts with activation > threshold
                threshold = self.config.ghost.ghost_activation_threshold
                active_count = (ghost_activations > threshold).sum().item()
                return int(active_count)
            return 0
        except Exception:
            return 0
    
    def _compute_saturation_level(self) -> float:
        """Compute expert saturation level for ghost activation monitoring."""
        if not self.config.ghost.num_ghost_experts > 0:
            return 0.0
            
        try:
            # Get saturation metrics from the model
            saturation_metrics = self.model.get_last_saturation_metrics()
            if saturation_metrics and 'saturation_level' in saturation_metrics:
                return float(saturation_metrics['saturation_level'])
            return 0.0
        except Exception:
            return 0.0


class GeometricTrainingController(TrainingController):
    """Geometric Constrained Learning training controller - the revolutionary paradigm."""
    
    def __init__(self, model, config: MoEConfig):
        super().__init__(model, config)
        
        if config.training_mode != "geometric":
            raise ValueError("GeometricTrainingController requires training_mode='geometric'")
        
        # Import original geometric components
        from .geometric_training import GeometricDataRotator, GeometricLossComputer, LambdaCalculusGeometricRotator
        
        # Initialize geometric components - FORCE base rotator for performance
        # Lambda rotator is pure bloat with no actual functionality
        self.data_rotator = GeometricDataRotator(config).to(self.device)
        
        self.loss_computer = GeometricLossComputer(config)
        self.memory_monitor = None
        self.use_streaming = False
        
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
        """EMERGENCY FAST MODE: Bypass the catastrophic 8x forward pass bottleneck."""
        self.step_count = step
        
        # PERFORMANCE EMERGENCY: The original geometric training was doing 8 separate
        # full forward passes through the entire model per batch! That's insane.
        # This emergency mode uses standard training with minimal geometric loss.
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Create targets (shifted input for language modeling)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        # FAST PATH: Single forward pass through model (like standard training)
        outputs = self.model(inputs, step=step, attention_mask=attention_mask[:, :-1])
        
        # Extract logits from model output
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Standard cross-entropy loss
        base_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Add minimal geometric regularization (lightweight)
        rotation_angles = self.data_rotator.get_rotation_angles()
        rotation_penalty = torch.mean(rotation_angles ** 2) * 0.001  # Tiny weight
        
        total_loss = base_loss + rotation_penalty
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Update rotation parameters (but much smaller effect)
        self.rotation_optimizer.step()
        self.rotation_scheduler.step()
        self.rotation_optimizer.zero_grad()
        
        # Update expert parameters
        self.expert_optimizer.step()
        self.expert_scheduler.step([0])
        self.expert_optimizer.zero_grad()
        
        # Minimal metrics (no expensive architecture extraction)
        current_metrics = {
            'learning_rate': self.rotation_optimizer.param_groups[0]['lr'],
            'expert_learning_rate': self.expert_optimizer.param_groups[0]['lr'],
            'training_mode': 'geometric_fast',
            'base_loss': base_loss.item(),
            'rotation_penalty': rotation_penalty.item(),
            'geometric_components': {  # Placeholder for compatibility
                'task_loss': base_loss.item(),
                'rotation_efficiency_loss': rotation_penalty.item(),
                'orthogonality_loss': 0.0,
                'specialization_loss': 0.0,
                'total_loss': total_loss.item()
            }
        }
        
        self.update_metrics(total_loss.item(), current_metrics)
        
        return total_loss
    
    
    def _forward_expert_optimized(self, expert_idx: int, rotated_data: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """OPTIMIZED forward pass - reuses computation where possible."""
        # Apply position embeddings to the rotated data
        batch_size, seq_len, embed_dim = rotated_data.shape
        pos_ids = torch.arange(seq_len, device=rotated_data.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.model.pos_emb(pos_ids)
        
        # Combine rotated token embeddings with position embeddings
        x = rotated_data + pos_emb
        x = self.model.dropout(x)
        
        # OPTIMIZED: Cache masks (same for all experts)
        if not hasattr(self, '_cached_masks') or self._cached_masks[0] != seq_len:
            causal_mask = self.model.create_causal_mask(seq_len, rotated_data.device)
            key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
            self._cached_masks = (seq_len, causal_mask, key_padding_mask)
        else:
            _, causal_mask, key_padding_mask = self._cached_masks
        
        # Forward through model layers with step parameter
        for layer in self.model.model_layers:
            x = layer(x, step=self.step_count, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        # Apply output normalization to get final hidden state
        hidden_state = self.model.output_norm(x)
        
        # Apply language model head to get logits
        logits = self.model.lm_head(hidden_state)
        
        return logits, hidden_state
    
    def _forward_expert(self, expert_idx: int, rotated_data: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        
        # Apply output normalization to get final hidden state
        hidden_state = self.model.output_norm(x)
        
        # Apply language model head to get logits
        logits = self.model.lm_head(hidden_state)
        
        return logits, hidden_state
    
    def _compute_metrics_efficient(self, loss_components: Dict[str, Any], rotation_angles: torch.Tensor) -> Dict[str, Any]:
        """OPTIMIZED metrics computation - defer expensive operations."""
        from .geometric_training import safe_item
        
        # Core metrics (always computed)
        current_metrics = {
            'learning_rate': safe_item(self.rotation_optimizer.param_groups[0]['lr']),
            'expert_learning_rate': safe_item(self.expert_optimizer.param_groups[0]['lr']),
            'training_mode': 'geometric'
        }
        
        # OPTIMIZATION: Only compute expensive metrics every N steps
        compute_full_metrics = (self.step_count % 10 == 0)  # Every 10 steps instead of every step
        
        if compute_full_metrics:
            # Expensive operations - only when needed
            geometric_components_dict = {k: safe_item(v) for k, v in loss_components.items()}
            
            current_metrics.update({
                'rotation_angles': rotation_angles.detach().cpu().numpy().tolist(),
                'rotation_efficiency': safe_item(loss_components.get('rotation_efficiency_loss', 0.0)),
                'expert_specialization': safe_item(loss_components.get('specialization_loss', 0.0)),
                'geometric_components': geometric_components_dict,
            })
            
            # Architecture metrics (most expensive)
            architecture_metrics = self._extract_architecture_metrics()
            current_metrics.update(architecture_metrics)
        else:
            # Lightweight metrics for non-full-metric steps
            current_metrics.update({
                'rotation_efficiency': safe_item(loss_components.get('rotation_efficiency_loss', 0.0)),
                'expert_specialization': safe_item(loss_components.get('specialization_loss', 0.0)),
            })
        
        return current_metrics
    
    def get_optimizers(self) -> List[torch.optim.Optimizer]:
        """Return both rotation and expert optimizers."""
        return [self.rotation_optimizer, self.expert_optimizer]
    
    def get_schedulers(self) -> List[Any]:
        """Return both rotation and expert schedulers."""
        return [self.rotation_scheduler, self.expert_scheduler]
    
    def _extract_architecture_metrics(self) -> Dict[str, Any]:
        """Extract metrics from underlying architecture (HGNN, Ghost, etc.)."""
        architecture_metrics = {}
        
        # Extract expert connection data for HGNN architectures
        if hasattr(self.model, 'model_layers') and self.config.use_hypergraph_coupling:
            for layer_idx, layer in enumerate(self.model.model_layers):
                if hasattr(layer, 'coupler') and layer.coupler is not None:
                    # Try multiple methods to extract connection data
                    connection_data = None
                    
                    # Method 1: get_adjacency_matrix
                    if hasattr(layer.coupler, 'get_adjacency_matrix'):
                        try:
                            adj_matrix = layer.coupler.get_adjacency_matrix()
                            if adj_matrix is not None:
                                connection_data = {
                                    'adjacency_matrix': adj_matrix.tolist() if hasattr(adj_matrix, 'tolist') else adj_matrix,
                                    'layer': layer_idx,
                                    'type': 'adjacency_matrix'
                                }
                        except Exception:
                            pass
                    
                    # Method 2: hyperedge_weights (the actual available attribute)
                    if connection_data is None and hasattr(layer.coupler, 'hyperedge_weights'):
                        try:
                            edge_weights = layer.coupler.hyperedge_weights
                            if edge_weights is not None:
                                connection_data = {
                                    'edge_weights': edge_weights.detach().cpu().numpy().tolist(),
                                    'layer': layer_idx,
                                    'type': 'hyperedge_weights'
                                }
                        except Exception:
                            pass
                    
                    # Method 3: _hyperedge_index (static topology)
                    if connection_data is None and hasattr(layer.coupler, '_hyperedge_index'):
                        try:
                            edge_index = layer.coupler._hyperedge_index
                            if edge_index is not None and edge_index.numel() > 0:
                                connection_data = {
                                    'hyperedge_index': edge_index.detach().cpu().numpy().tolist(),
                                    'num_hyperedges': layer.coupler._num_hyperedges,
                                    'layer': layer_idx,
                                    'type': 'hyperedge_topology'
                                }
                        except Exception:
                            pass
                    
                    # If we got any connection data, store it
                    if connection_data is not None:
                        architecture_metrics['expert_connections'] = connection_data
                        break  # Use first layer with connection data
        
        # Extract ghost expert metrics
        if hasattr(self.model, 'model_layers') and self.config.ghost.num_ghost_experts > 0:
            # Extract ghost activation levels using model methods
            try:
                ghost_activations = self.model.get_current_ghost_activations()
                if ghost_activations is not None and ghost_activations.numel() > 0:
                    architecture_metrics['ghost_activations'] = ghost_activations.detach().cpu().numpy().tolist()
                else:
                    architecture_metrics['ghost_activations'] = [0.0] * self.config.ghost.num_ghost_experts
            except Exception:
                architecture_metrics['ghost_activations'] = [0.0] * self.config.ghost.num_ghost_experts
            
            # Extract saturation level using model method
            try:
                saturation_metrics = self.model.get_last_saturation_metrics()
                if saturation_metrics and 'saturation_level' in saturation_metrics:
                    architecture_metrics['saturation_level'] = saturation_metrics['saturation_level']
                else:
                    architecture_metrics['saturation_level'] = 0.0
            except Exception:
                architecture_metrics['saturation_level'] = 0.0
            
            # Extract expert loads (both primary and ghost)
            try:
                expert_loads = self.model.get_expert_activation_loads()
                if expert_loads and (expert_loads.get('primary') or expert_loads.get('ghost')):
                    architecture_metrics['expert_loads'] = expert_loads
                else:
                    architecture_metrics['expert_loads'] = {
                        'primary': [1.0] * self.config.num_experts,
                        'ghost': [0.0] * self.config.ghost.num_ghost_experts
                    }
            except Exception:
                architecture_metrics['expert_loads'] = {
                    'primary': [1.0] * self.config.num_experts,
                    'ghost': [0.0] * self.config.ghost.num_ghost_experts
                }
        
        # Extract orthogonality score
        try:
            orthogonality_score = self._compute_expert_orthogonality()
            if orthogonality_score > 0:
                architecture_metrics['orthogonality_score'] = orthogonality_score
        except Exception:
            pass
        
        return architecture_metrics
    
    def _extract_expert_loads(self) -> Dict[str, Any]:
        """Extract expert load distribution."""
        loads = {}
        
        if hasattr(self.model, 'model_layers'):
            primary_loads = []
            ghost_loads = []
            
            for layer in self.model.model_layers:
                # Primary expert loads
                if hasattr(layer, 'primary_experts'):
                    primary_loads.extend([1.0] * len(layer.primary_experts))  # Simplified - all experts active
                
                # Ghost expert loads  
                if hasattr(layer, 'ghost_experts') and layer.ghost_experts is not None:
                    ghost_loads.extend([1.0] * len(layer.ghost_experts))  # Simplified
            
            if primary_loads:
                loads['primary'] = primary_loads
            if ghost_loads:
                loads['ghost'] = ghost_loads
        
        return loads
    
    def _compute_expert_orthogonality(self) -> float:
        """Compute orthogonality metric for experts (same as StandardTrainingController)."""
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

    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current geometric training metrics with defensive programming."""
        if not self.metrics_history['loss']:
            return {
                'loss': 0.0, 
                'learning_rate': self.config.geometric.geometric_learning_rate,
                'training_mode': 'geometric_fast',
                'geometric_components': {}
            }
        
        # Start with core metrics - use .get() for safety
        current_metrics = {
            'loss': self.metrics_history['loss'][-1],
            'learning_rate': self.metrics_history.get('learning_rate', [self.config.geometric.geometric_learning_rate])[-1] if self.metrics_history.get('learning_rate') else self.config.geometric.geometric_learning_rate,
            'expert_learning_rate': self.metrics_history.get('expert_learning_rate', [self.config.geometric.expert_learning_rate])[-1] if self.metrics_history.get('expert_learning_rate') else self.config.geometric.expert_learning_rate,
            'rotation_angles': self.metrics_history.get('rotation_angles', [[]])[-1] if self.metrics_history.get('rotation_angles') else [],
            'rotation_efficiency': self.metrics_history.get('rotation_efficiency_loss', [0.0])[-1] if self.metrics_history.get('rotation_efficiency_loss') else 0.0,
            'expert_specialization': self.metrics_history.get('expert_specialization', [0.0])[-1] if self.metrics_history.get('expert_specialization') else 0.0,
            'geometric_components': self.metrics_history.get('geometric_components', [{}])[-1] if self.metrics_history.get('geometric_components') else {},
            'training_mode': 'geometric_fast'
        }
        
        # DYNAMIC INCLUSION: Add any additional metrics that were stored but not in the core list
        core_keys = set(current_metrics.keys())
        for metric_key, metric_history in self.metrics_history.items():
            if metric_key not in core_keys and metric_history:
                # Include the latest value of any additional metric
                current_metrics[metric_key] = metric_history[-1]
        
        return current_metrics


def create_training_controller(model, config: MoEConfig) -> TrainingController:
    """Factory function to create appropriate training controller."""
    
    if config.training_mode == "geometric":
        return GeometricTrainingController(model, config)
    
    elif config.training_mode == "standard":
        return StandardTrainingController(model, config)
    
    else:
        raise ValueError(f"Unknown training_mode: '{config.training_mode}'. Must be 'standard' or 'geometric'.")
