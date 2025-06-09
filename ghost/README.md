# Ghost Expert Architecture (Phase 4)

**Adaptive Capacity & Overflow Specialization for HGNN-MoE**

[![Status](https://img.shields.io/badge/Status-Operational-green)](https://shields.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Ghost%20Experts-purple)](https://shields.io/)
[![Dataset](https://img.shields.io/badge/Dataset-WikiText--2-blue)](https://shields.io/)

---

## 🔥 Overview

The Ghost Expert Architecture represents the cutting-edge evolution of the adaptive orthogonal HGNN-MoE system. It introduces **dynamic capacity scaling** through "Ghost Experts" that remain dormant until primary experts reach saturation, then activate to handle representational overflow without disrupting established specializations.

### Core Innovation

Unlike traditional MoE architectures with fixed expert counts, Ghost Experts provide **adaptive model capacity** that scales automatically based on task complexity. This allows the model to:

- **Preserve specializations** of highly-tuned primary experts
- **Handle overflow patterns** through dynamically activated ghost experts  
- **Scale capacity** without predefined limits
- **Maintain orthogonality** while expanding representational power

---

## 🚀 Quick Start

### Basic Training

```bash
# Train with default Ghost Expert configuration
python -m ghost.run_gnn_moe

# Train with custom parameters
python -m ghost.run_gnn_moe --num_ghost_experts 8 --ghost_activation_threshold 0.8 --epochs 5
```

### Hyperparameter Sweeps

```bash
# Test different architecture configurations (16 combinations)
python ghost/tests/run_architecture_sweep.py

# Optimize ghost expert parameters (27 combinations)  
python ghost/tests/run_ghost_sweep.py

# Tune training hyperparameters (27 combinations)
python ghost/tests/run_training_sweep.py
```

### Analyze Results

```bash
# Process sweep results with enhanced analysis
python ghost/tests/enhanced_analysis.py ghost/tests/sweeps/architecture_YYYYMMDD_HHMMSS/
```

---

## 🏗️ Architecture Components

### 1. **Ghost Expert System**

#### `GhostAwareExpertBlock`
Expert blocks that can be dynamically scaled by activation levels (0.0 = dormant, 1.0 = fully active).

```python
class GhostAwareExpertBlock(nn.Module):
    def forward(self, x, activation_level=1.0):
        return self.mlp(x) * activation_level
```

#### `ExpertSaturationMonitor`
Detects when primary experts reach capacity using orthogonality and unexplained variance metrics.

**Key Metrics:**
- **Orthogonality Score**: Measures expert specialization (higher = more specialized)
- **Saturation Level**: Combined metric indicating when experts need assistance
- **Unexplained Variance**: Residual patterns primary experts cannot handle

### 2. **Activation Control System**

#### `GhostActivationController`
Manages ghost expert lifecycle through state transitions:

- **Dormant** → **Activating** → **Active**
- Dynamic activation based on saturation thresholds
- Smooth transitions to prevent training instability

#### `PrimaryGhostLRScheduler`
Novel **inverse learning rate dynamics**:

- **Primary experts**: Decreasing LR over time (fine-tuning specializations)
- **Ghost experts**: Increasing LR when activated (learning new patterns)
- **Natural handoff**: Ensures proper division of labor

### 3. **Triple Hypergraph Communication**

#### `TripleHypergraphCoupler`
Multi-level communication system with three specialized couplers:

1. **Primary-only**: Preserves existing expert specializations
2. **Ghost-only**: Coordinates newly activated capacities  
3. **Mixed primary-ghost**: Enables compositional reasoning

---

## ⚙️ Configuration

### Ghost Expert Parameters

```python
@dataclass
class GhostMoEConfig(GNNMoEConfig):
    # Core ghost parameters
    num_ghost_experts: int = 4                    # Number of ghost experts
    ghost_activation_threshold: float = 0.7       # Saturation threshold for activation
    ghost_learning_rate: float = 1e-4            # Initial LR for ghost experts
    
    # Activation dynamics
    ghost_activation_schedule: str = "gradual"    # "gradual", "binary", "selective"
    saturation_monitoring_window: int = 100       # Steps for saturation averaging
    
    # Learning rate coupling  
    ghost_lr_coupling: str = "inverse"            # "inverse", "complementary"
    
    # Hypergraph configuration
    ghost_hypergraph_strategy: str = "all"        # "primary_only", "ghost_only", "all"
    mixed_coupling_weight: float = 0.33           # Weight for mixed interactions
```

### Training Configuration

```python
# Fast validation runs
epochs: 1
max_batches_per_epoch: 100
eval_every: 50

# Dataset
dataset_name: "wikitext"
dataset_config_name: "wikitext-2-v1"
num_train_samples: 2000
num_eval_samples: 400
```

---

## 📊 Hyperparameter Sweeps

### Architecture Sweep
Tests core model architecture stability:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `embed_dim` | [128, 256] | Model size scaling |
| `num_layers` | [4, 6] | Depth effects |
| `num_experts` | [4, 8] | Expert count impact |
| `gnn_layers` | [2, 3] | Coupling complexity |

**Total combinations**: 16

### Ghost Parameter Sweep  
Optimizes ghost expert behavior:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `num_ghost_experts` | [2, 4, 8] | Capacity scaling |
| `ghost_activation_threshold` | [0.6, 0.75, 0.9] | Activation sensitivity |
| `ghost_learning_rate` | [1e-4, 5e-5, 1e-5] | Learning dynamics |

**Total combinations**: 27

### Training Parameter Sweep
Optimizes training hyperparameters:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `learning_rate` | [5e-4, 1e-4, 5e-5] | Primary expert tuning |
| `batch_size` | [16, 32, 64] | Memory vs. stability |
| `dropout_rate` | [0.1, 0.2, 0.3] | Regularization |

**Total combinations**: 27

---

## 📈 Output & Analysis

### Training Logs

Each run generates comprehensive logs in `training_log.json`:

```json
{
  "step": 50,
  "train_loss": 3.245,
  "eval_loss": 3.189,
  "primary_lr": 5e-4,
  "ghost_lrs": [1e-4, 1e-4, 0, 0],
  "ghost_activations": [1.0, 0.8, 0.0, 0.0],
  "saturation_level": 0.75,
  "orthogonality_score": 0.92
}
```

### Real-time Monitoring

Training provides continuous updates every 5 steps:

```
Step 45: Loss=3.245, LR=5.0e-04, Ghost Activations=['1.00', '0.80', '0.00', '0.00']
Step 50: Loss=3.189, LR=5.0e-04, Ghost Activations=['1.00', '0.85', '0.12', '0.00']
```

### Analysis Outputs

The enhanced analysis system generates:

- **Individual run plots**: Training curves and ghost activation patterns
- **Comparison visualizations**: Cross-run performance analysis  
- **Performance rankings**: CSV summaries with best model identification
- **Best model checkpoints**: Automatic saving with metadata

---

## 🔬 Key Innovations

### 1. Saturation-Based Activation

Ghost experts activate based on **intelligent saturation detection**:

```python
saturation_score = (orthogonality_weight * orthogonality_score + 
                   variance_weight * unexplained_variance_score)

if saturation_score > ghost_activation_threshold:
    activate_next_ghost_expert()
```

### 2. Inverse Learning Rate Dynamics

Novel scheduling creates natural division of labor:

- **Primary experts**: `lr_primary = initial_lr * (1 - progress)`
- **Ghost experts**: `lr_ghost = initial_lr * activation_level * progress`

### 3. Adaptive Capacity Scaling

Model capacity scales automatically without predefined limits:

- Monitors primary expert utilization
- Activates ghost experts on-demand
- Preserves existing specializations
- Enables compositional reasoning

---

## 🧬 Building on Previous Phases

### Phase 3 Foundation: Adaptive Orthogonal HGNN-MoE

Ghost Experts seamlessly integrate with:

- **Orthogonality constraints**: Maintains expert specialization
- **Adaptive controllers**: Preserves learned orthogonal relationships
- **HGNN coupling**: Extends hypergraph communication patterns
- **Performance monitoring**: Builds on existing metrics

### Architectural Progression

```
Phase 1: GNN-MoE (Graph coupling)
    ↓
Phase 2: HGNN-MoE (Hypergraph coupling)  
    ↓
Phase 3: Adaptive Orthogonality (Enforced specialization)
    ↓
Phase 4: Ghost Experts (Adaptive capacity)
```

---

## 📁 Directory Structure

```
ghost/
├── README.md                    # This file
├── run_gnn_moe.py              # Main training script
├── gnn_moe_architecture.py     # Ghost Expert components
├── gnn_moe_config.py           # Configuration classes
├── gnn_moe_training.py         # Training loop with enhanced logging
├── gnn_moe_data.py             # Data loading utilities
├── analysis.py                 # Model analysis tools
├── tests/                      # Testing and sweep infrastructure
│   ├── test_ghost_components.py # Unit tests
│   ├── sweep_framework.py      # Core sweep infrastructure
│   ├── enhanced_analysis.py    # Results analysis
│   ├── run_architecture_sweep.py
│   ├── run_ghost_sweep.py
│   ├── run_training_sweep.py
│   └── sweep_configs/          # Sweep parameter definitions
├── project-knowledge/          # Documentation and logs
│   ├── ghost_implementation_log.md
│   ├── hyperparameter_sweep_completion_log.md
│   └── sweep_debugging_log.md
└── __init__.py
```

---

## 🎯 Usage Examples

### Custom Ghost Configuration

```python
from ghost.gnn_moe_config import GhostMoEConfig

config = GhostMoEConfig(
    embed_dim=256,
    num_experts=6,
    num_ghost_experts=4,
    ghost_activation_threshold=0.8,
    ghost_learning_rate=5e-5,
    epochs=3
)
```

### Advanced Sweep Configuration

```json
{
    "sweep_params": {
        "ghost_activation_threshold": {
            "values": [0.6, 0.7, 0.8, 0.9],
            "prefix": "gat"
        },
        "num_ghost_experts": {
            "values": [2, 4, 6, 8],
            "prefix": "nge"  
        }
    },
    "static_params": {
        "epochs": 2,
        "eval_every": 25
    }
}
```

---

## 🔧 Development

### Running Tests

```bash
# Test ghost expert components
python -m unittest ghost.tests.test_ghost_components

# Test complete workflow
python -m ghost.run_gnn_moe --epochs 1 --max_batches_per_epoch 10
```

### Adding New Sweeps

1. Create configuration in `ghost/tests/sweep_configs/`
2. Create runner script using `SweepRunner` framework
3. Process results with `enhanced_analysis.py`

---

## 🎯 Research Applications

The Ghost Expert architecture enables research into:

- **Dynamic neural architecture search**
- **Adaptive model capacity allocation**  
- **Overflow pattern specialization**
- **Multi-scale expert hierarchies**
- **Compositional reasoning through expert cooperation**

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers
- Datasets (Hugging Face)
- NumPy, Pandas, Matplotlib, Seaborn

---

## 🤝 Contributing

1. **Testing**: Run unit tests before submitting changes
2. **Documentation**: Update relevant documentation for new features
3. **Sweeps**: Add sweep configurations for new parameters
4. **Analysis**: Ensure new metrics are captured in training logs

---

## 📚 References

- **Phase 3 Implementation**: `../orthogon/adaptive-orthogonal/`
- **Technical Details**: `project-knowledge/ghost_implementation_log.md`
- **Sweep Infrastructure**: `project-knowledge/hyperparameter_sweep_completion_log.md`

---

**Ghost Experts: Where adaptive capacity meets intelligent specialization.** 👻✨
