# Unified Mixture-of-Experts (MoE) Research Framework

[![Status](https://img.shields.io/badge/Status-Active%20Development-blue)](https://shields.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://shields.io/)

## 🚀 Overview

This repository contains a research framework for developing and testing novel Mixture-of-Experts (MoE) architectures for language models. The project is structured as a series of evolutionary phases, with each phase building upon the last to introduce new capabilities.

The entire codebase has been refactored into a unified, backwards-compatible system. A single `run.py` entry point, driven by a unified configuration, can train, evaluate, and analyze any of the architectures developed throughout the project's history.

---

## 🏗️ Architectural Evolution

The project has progressed through four distinct phases, each introducing a key innovation in expert-based language modeling.

### Phase 1: GNN-MoE (Graph-Coupled MoE)
*   **Core Idea**: Introduce communication between experts within a Transformer layer using a standard Graph Neural Network (GNN).
*   **Goal**: To investigate if learned, graph-based coordination can lead to better expert specialization compared to independent experts.
*   **Status**: Foundational architecture.

### Phase 2: HGNN-MoE (Hypergraph-Coupled MoE)
*   **Core Idea**: Upgrade the communication backbone from a GNN to a Hypergraph Neural Network (HGNN).
*   **Goal**: To model more complex, higher-order relationships between experts, allowing for more sophisticated coordination patterns (e.g., triplets of experts working together).
*   **Status**: Enhanced communication model.

### Phase 3: Orthogonal HGNN-MoE
*   **Core Idea**: Introduce an adaptive orthogonality loss to the expert weights.
*   **Goal**: To explicitly enforce expert specialization by penalizing representational overlap. The system intelligently adjusts the strength of this constraint based on real-time specialization scores, ensuring robust training without expert collapse.
*   **Status**: Enforced specialization model.

### Phase 4: Ghost Expert HGNN-MoE
*   **Core Idea**: Introduce "Ghost Experts" – a pool of dormant experts that are dynamically activated when the primary experts saturate.
*   **Goal**: To create a model with adaptive capacity that can scale its complexity on-demand in response to task difficulty, without disrupting the specializations of the primary experts.
*   **Status**: Current state-of-the-art, adaptive capacity model.

---

## ✨ Unified Codebase

The project has been refactored into a centralized and unified structure to streamline experimentation and improve maintainability.

*   **`core/`**: A new top-level directory containing the single source of truth for all definitive modules:
    *   `config.py`: A unified, backwards-compatible configuration for all architectures.
    *   `architecture.py`: A single, configurable `MoEModel` that can represent any of the four architectural phases.
    *   `training.py`: The unified training, validation, and checkpointing loop.
    *   `data.py`: The unified data loading utilities.
    *   `analysis.py`: The unified, mode-aware analysis and plotting script.
*   **`run.py`**: The single, top-level entry point for all experiments.
*   **`gnn_MoE/`, `hgnn_MoE/`, `orthogon/`, `ghost/`**: These directories now only contain historical documentation and project knowledge logs. All active code has been removed.

---

## 🚀 Quick Start

All experiments are now run from the top-level `run.py` script. The desired architecture is selected using the `--architecture_mode` flag.

### 1. Basic Training

Use `python3 run.py` with the desired arguments to launch a training run.

**Example: Train the latest "Ghost" architecture**
```bash
python3 run.py \
    --architecture_mode ghost \
    --run_name my_ghost_run \
    --batch_size 8 \
    --embed_dim 256 \
    --num_experts 8 \
    --ghost_num_ghost_experts 4 \
    --epochs 5
```

**Example: Train the "Orthogonal" architecture**
(This is achieved by setting the mode and not specifying ghost expert parameters)
```bash
python3 run.py \
    --architecture_mode orthogonal \
    --run_name my_orthogonal_run \
    --batch_size 16 \
    --epochs 5
```

**Example: Train the baseline "GNN" architecture**
```bash
python3 run.py \
    --architecture_mode gnn \
    --run_name my_gnn_run \
    --batch_size 16 \
    --epochs 5
```

### 2. Resuming from a Checkpoint

To resume a previous run, use the `--resume_checkpoint` flag.

```bash
python3 run.py \
    --run_name my_ghost_run \
    --resume_checkpoint checkpoints/my_ghost_run/checkpoint.pt
```
*Note: The script will automatically use the configuration saved in the checkpoint's directory.*

### 3. Reducing Memory Usage

If you encounter an Out-of-Memory (OOM) error, reduce the memory-intensive parameters.

**Example: Low-memory configuration**
```bash
python3 run.py \
    --architecture_mode ghost \
    --run_name low_mem_test \
    --batch_size 4 \
    --embed_dim 128 \
    --num_layers 2 \
    --num_experts 4 \
    --ghost_num_ghost_experts 2
```

---

## ⚙️ Configuration

All parameters are defined in `core/config.py` and can be overridden from the command line via `run.py`.

#### Key Parameters
*   `--architecture_mode`: `[gnn, hgnn, orthogonal, ghost]` - Selects the model architecture and automatically sets the correct feature flags.
*   `--batch_size`: The number of sequences per batch. **(Primary dial for memory usage)**.
*   `--embed_dim`: The core embedding dimension of the model.
*   `--num_layers`: The number of MoE layers in the model.
*   `--num_experts`: The number of primary experts in each layer.
*   `--ghost_num_ghost_experts`: The number of ghost experts. Setting this to `0` disables the ghost mechanism.
*   `--ghost_activation_threshold`: The saturation threshold to activate ghost experts (default: `0.01`).

---

## 📊 Analysis

After each run, a suite of analysis plots is automatically generated in the run's checkpoint directory (e.g., `checkpoints/my_ghost_run/`). The analysis script is mode-aware and will only generate plots relevant to the architecture that was run.

**Generated Plots May Include:**
*   Training & Evaluation Loss
*   Perplexity
*   Learning Rate Schedule
*   Expert Connection Heatmaps (for `hgnn`, `orthogonal`, `ghost` modes)
*   Ghost Expert Activations & Saturation Levels (for `ghost` mode)
*   Expert Load Distribution (for `ghost` mode)

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers
- Datasets (Hugging Face)
- NumPy, Pandas, Matplotlib, Seaborn
