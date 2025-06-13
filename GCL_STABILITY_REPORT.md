# Analysis of Training Instability in Geometric Constrained Learning

## 1. Executive Summary

The Geometric Constrained Learning (GCL) training process is currently unstable, leading to a `RuntimeError` during the backward pass of the geometric loss. This document provides a detailed analysis of the root cause and proposes a definitive solution that restores stability without compromising performance.

## 2. The Problem: `RuntimeError` in `lu_solve`

### 2.1 Error Description
The training process fails with the following error:

```
RuntimeError: Pivots given to lu_solve must all be greater or equal to 1. Did you properly pass the result of lu_factor?
```

### 2.2 Error Location
This error originates in the `torch.linalg.solve` function, which is called within the `create_rotation_matrix` function in `core/geometric_training.py`. The full traceback shows:

```
File "/Users/orion/Projects/GNN-MoE-2/core/training_controllers.py", line 297, in training_step
    geometric_loss.backward()
...
RuntimeError: Pivots given to lu_solve must all be greater or equal to 1.
```

### 2.3 Context
This function is a critical component of the GCL paradigm, responsible for creating the rotation matrices that present data optimally to each expert. The failure occurs during the backward pass when gradients are being computed.

## 3. Root Cause Analysis

### 3.1 Technical Analysis
The `create_rotation_matrix` function uses the Cayley transform to efficiently compute rotation matrices:

```python
# Cayley transform: R = (I - A)(I + A)^(-1)
rotation_matrix = torch.linalg.solve(I + A, I - A)
```

This method involves solving a linear system of equations using `torch.linalg.solve`. The `RuntimeError` occurs because the matrix being solved, `(I + A)`, is becoming "singular" or "ill-conditioned" during training. This means the matrix is not invertible, and the `lu_solve` operation (used internally by `torch.linalg.solve`) fails.

### 3.2 Why This Happens
- During training, the theta parameters evolve, causing the skew-symmetric matrix `A` to change
- Under certain conditions, `(I + A)` becomes numerically singular or near-singular
- The `lu_factorization` produces invalid pivot values (< 1), causing the solve operation to fail
- This is a numerical stability issue, not a logical error in the code

### 3.3 Existing Safety Mechanism
The code already contains a `try...except` block to handle this scenario:

```python
try:
    rotation_matrix = torch.linalg.solve(stabilized_I_plus_A, I - A)
except torch.linalg.LinAlgError:
    # Fallback to pseudo-inverse if singular (rare)
    rotation_matrix = torch.mm(I - A, torch.linalg.pinv(stabilized_I_plus_A))
```

However, this only catches `torch.linalg.LinAlgError`. The error being thrown is a more general `RuntimeError`, which is not being caught, causing the program to crash.

## 4. Failed Attempt: Performance Degradation

### 4.1 Initial Fix Attempt
An initial attempt was made to switch to the `create_rotation_matrix_lightweight` function, which uses Householder reflections instead of the Cayley transform. While this eliminated the `RuntimeError`, it introduced severe performance issues:

- Training speed dropped to less than 2 steps in 2 minutes
- Memory usage increased dramatically, nearly crashing the system
- The `torch.outer` operations in the lightweight function proved inefficient on the target hardware

### 4.2 Lesson Learned
Simply avoiding the problematic function is not a viable solution if it destroys performance. The original function must be fixed, not replaced.

## 5. The Definitive Solution: Enhanced Error Handling

### 5.1 Proposed Fix
The most effective solution is to enhance the existing error handling to also catch the `RuntimeError`. This ensures that any numerical instability in `torch.linalg.solve` is properly handled by falling back to the `torch.linalg.pinv` (pseudo-inverse) function.

**Current Code:**
```python
try:
    rotation_matrix = torch.linalg.solve(stabilized_I_plus_A, I - A)
except torch.linalg.LinAlgError:
    rotation_matrix = torch.mm(I - A, torch.linalg.pinv(stabilized_I_plus_A))
```

**Proposed Change:**
```python
try:
    rotation_matrix = torch.linalg.solve(stabilized_I_plus_A, I - A)
except (torch.linalg.LinAlgError, RuntimeError):
    rotation_matrix = torch.mm(I - A, torch.linalg.pinv(stabilized_I_plus_A))
```

### 5.2 Why This Solution is Optimal

**Targeted:** It directly addresses the point of failure without changing the core algorithm.

**Safe:** It leverages the existing fallback mechanism, which is designed specifically for singular matrix situations.

**Performance-Preserving:** It maintains the original performance characteristics by keeping the efficient Cayley transform as the primary method.

**Robust:** It catches both the specific `LinAlgError` and the more general `RuntimeError`, ensuring comprehensive error handling.

**Minimal:** It requires changing only one line of code, reducing the risk of introducing new bugs.

## 6. Implementation Details

### 6.1 File to Modify
- **File:** `core/geometric_training.py`
- **Function:** `GeometricDataRotator.create_rotation_matrix`
- **Line:** Exception handling block (approximately line 93-97)

### 6.2 Verification Steps
After implementing the fix:
1. Run the same training command that previously failed
2. Verify that training progresses beyond the first few steps
3. Monitor performance to ensure no degradation
4. Check that the fallback mechanism is working when needed

## 7. Technical Background: The Cayley Transform

### 7.1 Mathematical Foundation
The Cayley transform is a method for generating orthogonal matrices from skew-symmetric matrices:
- Given a skew-symmetric matrix `A`, the Cayley transform produces an orthogonal matrix `R`
- Formula: `R = (I - A)(I + A)^(-1)`
- This ensures that `R^T R = I` (orthogonality property)

### 7.2 Advantages
- Computationally efficient for generating rotation matrices
- Preserves orthogonality exactly (within numerical precision)
- Differentiable, allowing gradient-based optimization

### 7.3 Numerical Challenges
- Requires matrix inversion, which can be unstable for singular matrices
- The condition number of `(I + A)` determines numerical stability
- Large rotation angles can lead to ill-conditioned matrices

## 8. Alternative Approaches (Not Recommended)

### 8.1 Matrix Regularization
Adding epsilon to the diagonal was attempted but proved insufficient for the more general `RuntimeError`.

### 8.2 Alternative Rotation Methods
- Householder reflections (too slow)
- Direct parameterization (less mathematically sound)
- Givens rotations (more complex implementation)

These alternatives either sacrifice performance or mathematical rigor.

## 9. Conclusion and Next Steps

### 9.1 Immediate Action Required
Implement the enhanced error handling as described in Section 5.1. This single line change should resolve the training instability.

### 9.2 Long-term Monitoring
- Monitor the frequency of fallback usage during training
- If fallbacks become frequent, consider tuning the geometric learning rate
- Track performance metrics to ensure the fix doesn't impact convergence

### 9.3 Success Criteria
- Training completes without `RuntimeError` crashes
- Performance remains comparable to pre-issue levels
- Geometric Constrained Learning continues to show the documented improvements

---

**Report prepared:** December 13, 2025  
**Issue Status:** Requires implementation of enhanced error handling  
**Priority:** High - blocking research progress  
**Estimated Fix Time:** < 5 minutes
