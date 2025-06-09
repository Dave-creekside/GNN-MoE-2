# Step 1: Add Orthogonal Expert Output Loss

## **Objective**
Modify the existing `GNNMoELayer` to add soft orthogonality constraints on expert output representations, starting with a simple loss-based approach.

## **Files to Modify**
- `gnn_moe_architecture.py` (main changes)
- `gnn_moe_config.py` (add new hyperparameters)
- Training script (add loss component)

## **Implementation Details for Cline**

**1. Add to GNNMoEConfig:**
```python
# Add these new fields to your config dataclass
orthogonality_loss_weight: float = 0.1  # λ for orthogonality penalty
apply_orthogonality_loss: bool = True    # Enable/disable feature
orthogonality_aggregation: str = "mean"  # How to aggregate across batch/seq: "mean" or "pool"
```

**2. Add new method to GNNMoELayer class:**
```python
def compute_orthogonality_loss(self, expert_outputs_stack):
    """
    Compute soft orthogonality loss on expert output representations
    expert_outputs_stack: (B, L, E, D) tensor from stacked expert outputs
    Returns: scalar loss encouraging expert outputs to be orthogonal
    """
    if not self.config.apply_orthogonality_loss:
        return torch.tensor(0.0, device=expert_outputs_stack.device)
    
    B, L, E, D = expert_outputs_stack.shape
    
    if self.config.orthogonality_aggregation == "mean":
        # Average across batch and sequence dimensions
        mean_expert_outputs = expert_outputs_stack.mean(dim=(0, 1))  # (E, D)
        # Compute Gram matrix: expert_i · expert_j for all pairs
        gram_matrix = torch.mm(mean_expert_outputs, mean_expert_outputs.T)  # (E, E)
    else:
        # Pool approach: flatten B,L dims and compute gram matrix
        flat_outputs = expert_outputs_stack.view(-1, E, D)  # (B*L, E, D)
        gram_matrices = torch.bmm(flat_outputs, flat_outputs.transpose(1, 2))  # (B*L, E, E)
        gram_matrix = gram_matrices.mean(dim=0)  # (E, E)
    
    # Target: identity matrix (orthogonal experts)
    identity_target = torch.eye(E, device=expert_outputs_stack.device)
    
    # MSE loss between gram matrix and identity
    orthogonality_loss = F.mse_loss(gram_matrix, identity_target)
    
    return orthogonality_loss
```

**3. Modify GNNMoELayer.forward() method:**
```python
def forward(self, x, causal_mask=None, key_padding_mask=None):
    expert_outputs_tensors = [expert(x, causal_mask, key_padding_mask) for expert in self.experts]
    # Stack expert outputs: (B, L, E, D)
    stacked_expert_outputs = torch.stack(expert_outputs_tensors, dim=2)
    
    # NEW: Compute orthogonality loss and store for retrieval
    self._last_orthogonality_loss = self.compute_orthogonality_loss(stacked_expert_outputs)
    
    coordinated = self.coupler(stacked_expert_outputs) # Coupler expects (B,L,E,D)
    return x + coordinated # Additive skip connection over the whole MoE block
```

