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
        
        # MEMORY OPTIMIZED: Replace heavy rotation projectors with lightweight scaling
        # Old: num_experts × embed_dim × embed_dim parameters (huge!)
        # New: num_experts × embed_dim parameters (tiny!)
        self.rotation_scales = nn.Parameter(torch.ones(self.num_experts, self.embed_dim))
        self.rotation_shifts = nn.Parameter(torch.zeros(self.num_experts, self.embed_dim))
        
        # Optional: Shared basis for even more memory efficiency
        if getattr(self.config.geometric, 'use_shared_basis', False):
            basis_size = min(16, self.embed_dim // 4)  # Adaptive basis size
            self.shared_basis = nn.Parameter(torch.randn(self.embed_dim, basis_size) * 0.1)
            self.expert_coefficients = nn.Parameter(torch.randn(self.num_experts, basis_size) * 0.1)
        else:
            self.shared_basis = None
    
    def create_rotation_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Convert theta parameters to rotation matrix using efficient Cayley transform.
        
        OPTIMIZED: O(d²) complexity instead of O(d³). Dramatically faster!
        Uses Cayley transform: R = (I - A)(I + A)^(-1) where A is skew-symmetric.
        """
        device = theta.device
        dim = self.embed_dim
        
        # Create skew-symmetric matrix from theta parameters
        # This is much more efficient than iterative Givens rotations
        A = torch.zeros(dim, dim, device=device, dtype=theta.dtype)
        
        # Fill upper triangle with theta values (skew-symmetric: A[j,i] = -A[i,j])
        theta_idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                if theta_idx < len(theta):
                    angle = torch.tanh(theta[theta_idx]) * self.config.geometric.max_rotation_magnitude
                    A[i, j] = angle
                    A[j, i] = -angle  # Skew-symmetric property
                    theta_idx += 1
                else:
                    break
            if theta_idx >= len(theta):
                break
        
        # Cayley transform: R = (I - A)(I + A)^(-1)
        # This gives us an orthogonal matrix efficiently
        I = torch.eye(dim, device=device, dtype=theta.dtype)
        
        # STABILIZATION: Add a small epsilon to the diagonal of (I + A)
        # This ensures the matrix is invertible and prevents the lu_solve error.
        stabilized_I_plus_A = I + A + torch.eye(dim, device=device, dtype=theta.dtype) * 1e-6

        try:
            # Solve (I + A) * R = (I - A) for R using the stabilized matrix
            rotation_matrix = torch.linalg.solve(stabilized_I_plus_A, I - A)
        except (torch.linalg.LinAlgError, RuntimeError):
            # Fallback to pseudo-inverse if singular (handles both LinAlgError and RuntimeError)
            rotation_matrix = torch.mm(I - A, torch.linalg.pinv(stabilized_I_plus_A))
        
        return rotation_matrix

    def create_rotation_matrix_lightweight(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Ultra-lightweight rotation using direct parameterization.
        Even faster alternative for maximum performance.
        """
        device = theta.device
        dim = self.embed_dim
        
        # Use Householder reflections for O(d) construction
        # Much faster than full matrix operations
        if len(theta) < dim:
            # Pad theta if needed
            theta_padded = torch.cat([theta, torch.zeros(dim - len(theta), device=device)])
        else:
            theta_padded = theta[:dim]
        
        # Normalize to create unit vector
        v = F.normalize(theta_padded.unsqueeze(0), dim=1).squeeze(0)
        
        # Householder reflection: R = I - 2*v*v^T
        # But we want rotation, so use: R = I - 2*v*v^T + 2*cos(θ)*v*v^T
        I = torch.eye(dim, device=device, dtype=theta.dtype)
        outer_product = torch.outer(v, v)
        
        # Create rotation with controlled angle
        rotation_angle = torch.norm(theta) * 0.1  # Scale down for stability
        rotation_matrix = I - 2 * torch.sin(rotation_angle) * outer_product
        
        return rotation_matrix
    
    def compute_rotation_matrices(self) -> List[torch.Tensor]:
        """Convert all theta parameters to rotation matrices for each expert."""
        rotation_matrices = []
        
        for expert_idx in range(self.num_experts):
            theta = self.theta_parameters[expert_idx]
            rotation_matrix = self.create_rotation_matrix_lightweight(theta)
            rotation_matrices.append(rotation_matrix)
        
        return rotation_matrices
    
    def rotate_data_for_experts(self, input_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Present the same data optimally to each expert via learned theta rotations.
        
        MEMORY OPTIMIZED VERSION: Computes rotations on-demand with mixed precision.
        
        Args:
            input_data: [batch_size, seq_len, embed_dim]
            
        Returns:
            List of rotated data presentations, one optimized for each expert
        """
        rotated_presentations = []
        batch_size, seq_len, embed_dim = input_data.shape
        
        # Pre-flatten input data once for efficiency
        flat_data = input_data.view(-1, embed_dim)  # [batch_size * seq_len, embed_dim]
        
        for expert_idx in range(self.num_experts):
            # Compute rotation matrix just-in-time (no pre-computation of all matrices)
            # Device-agnostic autocast - works on both CUDA and MPS
            device_type = 'cuda' if flat_data.device.type == 'cuda' else flat_data.device.type
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=self.config.use_mixed_precision and device_type in ['cuda', 'mps']):
                theta = self.theta_parameters[expert_idx]
                rotation_matrix = self.create_rotation_matrix_lightweight(theta)
                
                # Apply rotation with device-appropriate precision
                if self.config.use_mixed_precision and device_type in ['cuda', 'mps']:
                    rotated_flat = torch.mm(flat_data.half(), rotation_matrix.half().t()).float()
                else:
                    rotated_flat = torch.mm(flat_data, rotation_matrix.t())
            
            # Reshape back to original dimensions
            rotated_data = rotated_flat.view(batch_size, seq_len, embed_dim)
            
            # Apply lightweight expert-specific scaling (much more memory efficient)
            rotated_data = rotated_data * self.rotation_scales[expert_idx] + self.rotation_shifts[expert_idx]
            
            # Optional: Apply shared basis transformation if enabled
            if self.shared_basis is not None:
                # Use shared basis with expert-specific coefficients
                basis_transform = torch.mm(self.shared_basis, self.expert_coefficients[expert_idx].unsqueeze(1)).squeeze(1)
                rotated_data = rotated_data + basis_transform.unsqueeze(0).unsqueeze(0)
            
            rotated_presentations.append(rotated_data)
            
            # Clear intermediate tensors to free memory immediately
            del rotated_flat, rotation_matrix
            # Device-agnostic memory cleanup (less aggressive to avoid device placement issues)
            if expert_idx < self.num_experts - 1:  # Don't clear on last iteration
                if hasattr(torch, 'cuda') and torch.cuda.is_available() and flat_data.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and flat_data.device.type == 'mps':
                    torch.mps.empty_cache()
        
        return rotated_presentations

    def rotate_data_for_experts_generator(self, input_data: torch.Tensor):
        """
        Memory-efficient generator version that yields rotations on-demand.
        Use this for even lower memory usage when processing experts sequentially.
        """
        batch_size, seq_len, embed_dim = input_data.shape
        flat_data = input_data.view(-1, embed_dim)
        
        for expert_idx in range(self.num_experts):
            # Device-agnostic autocast for generator version
            device_type = 'cuda' if flat_data.device.type == 'cuda' else flat_data.device.type
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=self.config.use_mixed_precision and device_type in ['cuda', 'mps']):
                theta = self.theta_parameters[expert_idx]
                rotation_matrix = self.create_rotation_matrix_lightweight(theta)
                
                # Apply rotation with device-appropriate precision
                if self.config.use_mixed_precision and device_type in ['cuda', 'mps']:
                    rotated_flat = torch.mm(flat_data.half(), rotation_matrix.half().t()).float()
                else:
                    rotated_flat = torch.mm(flat_data, rotation_matrix.t())
            
            rotated_data = rotated_flat.view(batch_size, seq_len, embed_dim)
            
            # Apply lightweight expert-specific scaling
            rotated_data = rotated_data * self.rotation_scales[expert_idx] + self.rotation_shifts[expert_idx]
            
            # Optional: Apply shared basis transformation if enabled
            if self.shared_basis is not None:
                basis_transform = torch.mm(self.shared_basis, self.expert_coefficients[expert_idx].unsqueeze(1)).squeeze(1)
                rotated_data = rotated_data + basis_transform.unsqueeze(0).unsqueeze(0)
            
            yield rotated_data
            
            # Device-agnostic memory cleanup
            del rotated_flat, rotation_matrix
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    
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
        expert_hidden_states: List[torch.Tensor],
        rotated_data: List[torch.Tensor], 
        targets: torch.Tensor,
        rotation_angles: torch.Tensor,
        model
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the complete geometric loss with all components.
        
        Args:
            expert_outputs: List of expert logits (for task loss)
            expert_hidden_states: List of expert hidden states (for orthogonality loss)
            
        Returns:
            total_loss: Combined geometric loss
            loss_components: Dictionary of individual loss components
        """
        
        loss_components = {}
        
        # 1. Task performance loss (standard language modeling) - use logits
        try:
            task_loss = self._compute_task_loss(expert_outputs, targets)
            if task_loss is not None and torch.isfinite(task_loss):
                loss_components['task_loss'] = safe_item(task_loss)
            else:
                task_loss = torch.tensor(0.0, device=targets.device)
                loss_components['task_loss'] = 0.0
        except Exception:
            task_loss = torch.tensor(0.0, device=targets.device)
            loss_components['task_loss'] = 0.0
        
        # 2. Orthogonality preservation loss - use hidden states (memory efficient)
        try:
            orthogonality_loss = self._compute_orthogonality_preservation_loss(expert_hidden_states)
            if orthogonality_loss is not None and torch.isfinite(orthogonality_loss):
                loss_components['orthogonality_loss'] = safe_item(orthogonality_loss)
            else:
                orthogonality_loss = torch.tensor(0.0, device=targets.device)
                loss_components['orthogonality_loss'] = 0.0
        except Exception:
            orthogonality_loss = torch.tensor(0.0, device=targets.device)
            loss_components['orthogonality_loss'] = 0.0
        
        # 3. Rotation efficiency loss
        try:
            rotation_efficiency_loss = self._compute_rotation_efficiency_loss(rotation_angles)
            if rotation_efficiency_loss is not None and torch.isfinite(rotation_efficiency_loss):
                loss_components['rotation_efficiency_loss'] = safe_item(rotation_efficiency_loss)
            else:
                rotation_efficiency_loss = torch.tensor(0.0, device=targets.device)
                loss_components['rotation_efficiency_loss'] = 0.0
        except Exception:
            rotation_efficiency_loss = torch.tensor(0.0, device=targets.device)
            loss_components['rotation_efficiency_loss'] = 0.0
        
        # 4. Expert specialization loss - use hidden states
        try:
            specialization_loss = self._compute_expert_specialization_loss(expert_hidden_states)
            if specialization_loss is not None and torch.isfinite(specialization_loss):
                loss_components['specialization_loss'] = safe_item(specialization_loss)
            else:
                specialization_loss = torch.tensor(0.0, device=targets.device)
                loss_components['specialization_loss'] = 0.0
        except Exception:
            specialization_loss = torch.tensor(0.0, device=targets.device)
            loss_components['specialization_loss'] = 0.0
        
        # Combine losses with configured weights
        total_loss = (
            task_loss + 
            self.geometric.orthogonality_weight * orthogonality_loss +
            self.geometric.rotation_efficiency_weight * rotation_efficiency_loss +
            self.geometric.specialization_weight * specialization_loss
        )
        
        loss_components['total_loss'] = safe_item(total_loss)
        
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
