# GNN-MoE Sweep Analysis & HGNN Optimization Insights

**Date:** 2025-06-01  
**Source:** Analysis of comprehensive GNN-MoE hyperparameter sweeps from `/gnn-moe-sweeps/`

## 📊 GNN-MoE Sweep Results Summary

### 1. Number of Experts Analysis
| Experts | Eval Loss | Perplexity | Training Time | Total Params | Performance/Cost |
|---------|-----------|------------|---------------|--------------|------------------|
| 2       | **4.5267** | 92.45      | 9.3 min       | 55.9M        | ⭐ BEST EFFICIENCY |
| 4       | 4.5336    | 93.09      | 16.4 min      | 70.1M        | Good balance |
| 8       | 4.5637    | 95.93      | 42.7 min      | 98.4M        | Diminishing returns |
| 16      | FAILED    | -          | -             | -            | Too many experts |

**Key Insight:** 🎯 **2 experts performed best** with fastest training. Diminishing returns beyond 4 experts.

### 2. Embedding Dimension Analysis
| Embed Dim | Eval Loss | Perplexity | Training Time | Total Params | Efficiency |
|-----------|-----------|------------|---------------|--------------|------------|
| 256       | 4.5834    | 97.85      | 11.7 min      | 39.8M        | Fast but limited |
| 384       | 4.5336    | 93.09      | 16.4 min      | 70.1M        | Good balance |
| 512       | **4.4988** | **89.91**  | 23.0 min      | 107.3M       | ⭐ BEST PERFORMANCE |

**Key Insight:** 🎯 **512 embed_dim is optimal** for performance, 384 for efficiency.

### 3. Learning Rate Analysis
| Learning Rate | Eval Loss | Perplexity | Performance |
|---------------|-----------|------------|-------------|
| 0.001         | 4.5281    | 92.58      | Good |
| **0.0005**    | **4.5113** | **91.04**  | ⭐ OPTIMAL |
| 0.0003        | 4.5336    | 93.09      | Baseline |
| 0.0001        | 4.7838    | 119.56     | Too low |

**Key Insight:** 🎯 **0.0005 learning rate is optimal** - sweet spot for GNN-MoE.

### 4. GNN Layers Analysis  
| GNN Layers | Eval Loss | Perplexity | Training Time | Performance |
|------------|-----------|------------|---------------|-------------|
| 1          | 4.5378    | 93.49      | 13.6 min      | Fast |
| 2          | 4.5336    | 93.09      | 16.4 min      | Good |
| **3**      | **4.5103** | **90.95**  | 19.2 min      | ⭐ BEST |

**Key Insight:** 🎯 **3 GNN layers optimal** for expert coordination.

### 5. Model Depth Analysis
| Model Layers | Eval Loss | Perplexity | Training Time | Performance |
|--------------|-----------|------------|---------------|-------------|
| 4            | 4.5336    | 93.09      | 16.8 min      | Baseline |
| 6            | 4.5000    | 90.02      | 22.5 min      | Better |
| **8**        | **4.4845** | **88.63**  | 28.7 min      | ⭐ BEST |

**Key Insight:** 🎯 **Deeper models perform better** but with significant compute cost.

## 🚀 HGNN Optimization Strategy

### Immediate HGNN Recommendations

Based on GNN sweep results, here are optimized HGNN configurations:

#### Configuration 1: Efficiency-Focused HGNN
```bash
!cd hgnn_moe_dev && python run_gnn_moe.py \
  --coupler_type HGNN \
  --static_hyperedge_strategy all_pairs \
  --hgnn_learnable_edge_weights \
  --num_experts 2 \
  --embed_dim 384 \
  --num_layers 6 \
  --gnn_layers 3 \
  --learning_rate 0.0005 \
  --epochs 15 \
  --max_batches_per_epoch 30 \
  --eval_every 20 \
  --num_train_samples 3000 \
  --num_eval_samples 750 \
  --run_name hgnn_efficiency_optimized
```
**Expected:** Fast training (~15-20 min), good performance, efficient resource usage.

#### Configuration 2: Performance-Focused HGNN
```bash
!cd hgnn_moe_dev && python run_gnn_moe.py \
  --coupler_type HGNN \
  --static_hyperedge_strategy all_pairs \
  --hgnn_learnable_edge_weights \
  --num_experts 4 \
  --embed_dim 512 \
  --num_layers 8 \
  --gnn_layers 3 \
  --learning_rate 0.0005 \
  --epochs 12 \
  --max_batches_per_epoch 35 \
  --eval_every 25 \
  --num_train_samples 4000 \
  --num_eval_samples 1000 \
  --run_name hgnn_performance_optimized
```
**Expected:** Best possible performance, longer training (~35-45 min).

#### Configuration 3: HGNN vs GNN Direct Comparison
```bash
# GNN Baseline with optimal settings
!cd hgnn_moe_dev && python run_gnn_moe.py \
  --coupler_type GNN \
  --num_experts 2 \
  --embed_dim 512 \
  --num_layers 8 \
  --gnn_layers 3 \
  --learning_rate 0.0005 \
  --epochs 10 \
  --max_batches_per_epoch 30 \
  --eval_every 20 \
  --num_train_samples 3000 \
  --num_eval_samples 750 \
  --run_name gnn_optimized_baseline

# HGNN with identical settings
!cd hgnn_moe_dev && python run_gnn_moe.py \
  --coupler_type HGNN \
  --static_hyperedge_strategy all_pairs \
  --hgnn_learnable_edge_weights \
  --num_experts 2 \
  --embed_dim 512 \
  --num_layers 8 \
  --gnn_layers 3 \
  --learning_rate 0.0005 \
  --epochs 10 \
  --max_batches_per_epoch 30 \
  --eval_every 20 \
  --num_train_samples 3000 \
  --num_eval_samples 750 \
  --run_name hgnn_optimized_comparison
```

### HGNN-Specific Research Questions

1. **Hyperedge Strategy Impact**: Does `all_triplets` vs `all_pairs` show different scaling with expert count?
2. **Expert Count Scaling**: Do HGNNs handle more experts better than GNNs due to richer connections?
3. **Learnable Weights**: How much do learnable hyperedge weights improve performance?
4. **Computational Efficiency**: How does HGNN training time compare to GNN at equivalent performance?

### Expected HGNN Performance Targets

Based on GNN baseline performance and theoretical HGNN advantages:

| Configuration | Expected Eval Loss | Expected Perplexity | Training Time |
|---------------|---------------------|---------------------|---------------|
| Efficiency    | 4.45-4.50          | 85-90              | 15-20 min     |
| Performance   | 4.35-4.45          | 75-85              | 35-45 min     |
| Comparison    | 4.40-4.48          | 80-88              | 25-35 min     |

## 📈 Key Optimization Principles

### 1. Sweet Spot Configurations
- **Learning Rate:** 0.0005 (proven optimal)
- **Experts:** 2-4 (diminishing returns beyond this)
- **GNN/HGNN Layers:** 3 layers (good performance/cost balance)
- **Model Depth:** 6-8 layers (deeper = better, but expensive)

### 2. Scaling Strategies
- **For Speed:** Use 2 experts, 384 embed_dim, 6 model layers
- **For Performance:** Use 4 experts, 512 embed_dim, 8 model layers  
- **For Research:** Test both strategies with HGNN vs GNN comparison

### 3. HGNN-Specific Considerations
- **Hyperedge Strategy:** Start with `all_pairs`, test `all_triplets` for 4+ experts
- **Learnable Weights:** Always enable for maximum expressiveness
- **Batch Processing:** Our optimization ensures efficient training regardless of size

## 🎯 Success Metrics for HGNN

### Performance Targets
- **Beat GNN baseline:** Eval loss < 4.50 (vs GNN ~4.53)
- **Maintain efficiency:** Training time within 2x of equivalent GNN
- **Show learning:** Clear hyperedge weight evolution patterns

### Research Validation
- **Convergence:** Stable loss reduction over 10+ epochs
- **Generalization:** Good eval performance vs training performance
- **Interpretability:** Meaningful hyperedge weight patterns

---

**This analysis provides evidence-based optimization strategies for HGNN research, leveraging comprehensive GNN-MoE baseline performance data.**
