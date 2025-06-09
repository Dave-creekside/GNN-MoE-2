#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometric_training.py

Core Geometric Constrained Learning implementation for MoE models.
Implements the revolutionary paradigm where model geometry is fixed and 
training optimizes data presentation angles (theta rotations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

from .config import MoEConfig


def safe_item(value):
    """Safely extract scalar value from tensor or return scalar as-is."""
    if torch.is_tensor(value):
        return value.item()
    return value


class GeometricDataRotator(nn.Module):
    """
    Learns optimal theta rotations for presenting data to orthogonal experts.
    
    This is the core of Geometric Constrained Learning - instead of adjusting
    expert weights to fit data, we adjust data presentation to fit the fixed
    orthogonal expert geometry (the "100-sided die").
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.embed_dim = config.embed_dim
        self.rotation_dimensions = config.geometric.rotation_dimensions
        
        # Learnable rotation parameters (theta values) for each expert
        # These represent the angles at which data should be presented to each expert
        self.theta_parameters = nn.Parameter(
            torch.randn(self.num_experts, self.rotation_dimensions) * 0.1
        )
        
        # Rotation transformation matrices - these convert theta parameters into actual rotations
        self.rotation_projectors = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        
        # Initialize rotation projectors to near-identity
        for projector in self.rotation_projectors:
            nn.init.orthogonal_(projector.weight)
            # Scale down to start near identity
            projector.weight.data *= 0.1
            projector.weight.data += torch.eye(self.embed_dim)
    
    def create_rotation_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Convert theta parameters to rotation matrix using Givens rotations.
        
        Givens rotations are ideal for maintaining orthogonality and are
        geometrically interpretable as rotations in specific planes.
        """
        device = theta.device
        dim = self.embed_dim
        
        # Start with identity matrix
        rotation_matrix = torch.eye(dim, device=device, dtype=theta.dtype)
        
        # Apply series of Givens rotations
        rotation_idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                if rotation_idx >= len(theta):
                    break
                    
                # Get rotation angle, constrained to reasonable range
                angle = torch.tanh(theta[rotation_idx]) * self.config.geometric.max_rotation_magnitude
                
                # Create Givens rotation matrix
                givens = torch.eye(dim, device=device, dtype=theta.dtype)
                givens[i, i] = torch.cos(angle)
                givens[i, j] = -torch.sin(angle)
                givens[j, i] = torch.sin(angle)
                givens[j, j] = torch.cos(angle)
                
                # Compose rotations
                rotation_matrix = torch.mm(rotation_matrix, givens)
                rotation_idx += 1
                
                if rotation_idx >= self.rotation_dimensions:
                    break
            
            if rotation_idx >= self.rotation_dimensions:
                break
        
        return rotation_matrix
    
    def compute_rotation_matrices(self) -> List[torch.Tensor]:
        """Convert all theta parameters to rotation matrices for each expert."""
        rotation_matrices = []
        
        for expert_idx in range(self.num_experts):
            theta = self.theta_parameters[expert_idx]
            rotation_matrix = self.create_rotation_matrix(theta)
            rotation_matrices.append(rotation_matrix)
        
        return rotation_matrices
    
    def rotate_data_for_experts(self, input_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Present the same data optimally to each expert via learned theta rotations.
        
        This is the revolutionary core: instead of changing expert weights,
        we change how data is presented to each expert's fixed geometry.
        
        Args:
            input_data: [batch_size, seq_len, embed_dim]
            
        Returns:
            List of rotated data presentations, one optimized for each expert
        """
        rotation_matrices = self.compute_rotation_matrices()
        
        rotated_presentations = []
        for expert_idx, rotation_matrix in enumerate(rotation_matrices):
            # Apply learned rotation to present data optimally for this expert
            batch_size, seq_len, embed_dim = input_data.shape
            
            # Reshape for matrix multiplication
            flat_data = input_data.view(-1, embed_dim)  # [batch_size * seq_len, embed_dim]
            
            # Apply rotation
            rotated_flat = torch.mm(flat_data, rotation_matrix.t())
            
            # Reshape back
            rotated_data = rotated_flat.view(batch_size, seq_len, embed_dim)
            
            # Apply expert-specific projector for additional learned transformation
            rotated_data = self.rotation_projectors[expert_idx](rotated_data)
            
            rotated_presentations.append(rotated_data)
        
        return rotated_presentations
    
    def get_rotation_angles(self) -> torch.Tensor:
        """Get current rotation angles for analysis and visualization."""
        angles = []
        for expert_idx in range(self.num_experts):
            theta = self.theta_parameters[expert_idx]
            # Convert to actual angles used in rotation
            actual_angles = torch.tanh(theta) * self.config.geometric.max_rotation_magnitude
            angles.append(actual_angles)
        return torch.stack(angles)
    
    def compute_rotation_efficiency_loss(self) -> torch.Tensor:
        """
        Compute loss that encourages efficient rotations (not over-rotating).
        
        This prevents the system from learning unnecessarily large rotations
        when smaller ones would suffice.
        """
        angles = self.get_rotation_angles()
        
        # L2 penalty on rotation magnitudes
        efficiency_loss = torch.mean(angles ** 2)
        
        # Additional penalty for very large rotations
        large_rotation_penalty = torch.mean(F.relu(torch.abs(angles) - 1.0) ** 2)
        
        return efficiency_loss + large_rotation_penalty


class LambdaCalculusGeometricRotator(GeometricDataRotator):
    """
    Specialized rotator for lambda calculus cognitive dimensions.
    
    Extends base rotator with domain-specific rotation formatters for:
    - Syntax: structural parsing
    - Reduction: step-by-step β-reduction  
    - Semantic: meaning interpretation
    - Pedagogical: teaching explanation
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        
        # Pre-defined cognitive rotation angles for lambda calculus
        self.cognitive_rotations = {
            'syntax': 0,      # 0° - structural parsing
            'reduction': 90,  # 90° - step-by-step reduction  
            'semantic': 180,  # 180° - meaning interpretation
            'pedagogical': 270  # 270° - teaching explanation
        }
        
        # Learnable mixing weights for cognitive dimensions
        self.cognitive_mixing = nn.Parameter(torch.ones(4))
    
    def create_lambda_specific_rotations(self, lambda_expression: str) -> Dict[str, torch.Tensor]:
        """
        Create rotations specific to lambda calculus cognitive dimensions.
        
        This would integrate with tokenized input to create specialized
        data presentations for different reasoning aspects.
        """
        rotations = {}
        
        # This is a placeholder for lambda-specific formatting
        # In practice, this would analyze the lambda expression structure
        # and create appropriate data presentations
        
        # For now, return the standard rotations
        base_rotations = self.rotate_data_for_experts
        
        return {
            'syntax': base_rotations,
            'reduction': base_rotations, 
            'semantic': base_rotations,
            'pedagogical': base_rotations
        }


class GeometricLossComputer:
    """
    Computes the multi-component geometric loss that rewards optimal data presentation.
    
    Components:
    1. Task performance (standard cross-entropy)
    2. Orthogonality preservation (experts stay orthogonal)
    3. Rotation efficiency (don't over-rotate)
    4. Expert specialization (each expert activates for different rotations)
    """
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.geometric = config.geometric
    
    def compute_geometric_loss(
        self, 
        expert_outputs: List[torch.Tensor],
        rotated_data: List[torch.Tensor], 
        targets: torch.Tensor,
        rotation_angles: torch.Tensor,
        model
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the complete geometric loss with all components.
        
        Returns:
            total_loss: Combined geometric loss
            loss_components: Dictionary of individual loss components
        """
        
        # 1. Task performance loss (standard language modeling)
        task_loss = self._compute_task_loss(expert_outputs, targets)
        
        # 2. Orthogonality preservation loss
        orthogonality_loss = self._compute_orthogonality_preservation_loss(expert_outputs)
        
        # 3. Rotation efficiency loss
        rotation_efficiency_loss = self._compute_rotation_efficiency_loss(rotation_angles)
        
        # 4. Expert specialization loss
        specialization_loss = self._compute_expert_specialization_loss(expert_outputs)
        
        # Combine losses with configured weights
        total_loss = (
            task_loss + 
            self.geometric.orthogonality_weight * orthogonality_loss +
            self.geometric.rotation_efficiency_weight * rotation_efficiency_loss +
            self.geometric.specialization_weight * specialization_loss
        )
        
        loss_components = {
            'task_loss': task_loss.item(),
            'orthogonality_loss': orthogonality_loss.item(),
            'rotation_efficiency_loss': rotation_efficiency_loss.item(),
            'specialization_loss': specialization_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _compute_task_loss(self, expert_outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Standard task performance loss."""
        # Combine expert outputs (simple averaging for now)
        combined_output = torch.stack(expert_outputs).mean(dim=0)
        
        # Standard cross-entropy loss
        return F.cross_entropy(
            combined_output.view(-1, combined_output.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
    
    def _compute_orthogonality_preservation_loss(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Ensure experts maintain orthogonal specialization patterns."""
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)
        
        # Compute pairwise similarities between expert outputs
        similarities = []
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                # Flatten outputs for similarity computation
                out_i = expert_outputs[i].view(-1)
                out_j = expert_outputs[j].view(-1)
                
                # Cosine similarity
                similarity = F.cosine_similarity(out_i.unsqueeze(0), out_j.unsqueeze(0))
                similarities.append(similarity.abs())
        
        # Loss encourages low similarity (high orthogonality)
        if similarities:
            return torch.stack(similarities).mean()
        else:
            return torch.tensor(0.0, device=expert_outputs[0].device)
    
    def _compute_rotation_efficiency_loss(self, rotation_angles: torch.Tensor) -> torch.Tensor:
        """Encourage efficient (not excessive) rotations."""
        # L2 penalty on rotation magnitudes
        return torch.mean(rotation_angles ** 2)
    
    def _compute_expert_specialization_loss(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Encourage each expert to specialize on different input patterns."""
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)
        
        # Compute variance in expert activations
        # Higher variance = better specialization
        output_stack = torch.stack(expert_outputs)  # [num_experts, batch, seq, features]
        
        # Compute activation strengths
        activation_strengths = torch.norm(output_stack, dim=-1)  # [num_experts, batch, seq]
        
        # Variance across experts (higher is better)
        expert_variance = torch.var(activation_strengths, dim=0)  # [batch, seq]
        
        # Loss encourages high variance (negative variance + constant offset)
        specialization_loss = torch.mean(-expert_variance + 1.0)
        
        return F.relu(specialization_loss)  # Only penalize if variance is too low
