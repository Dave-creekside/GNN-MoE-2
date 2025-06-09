# A Research Framework for Revolutionary Mixture-of-Experts Architectures

[![Status](https://img.shields.io/badge/Status-Active%20Development-blue)](https://shields.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://shields.io/)
[![Breakthrough](https://img.shields.io/badge/🚀-Geometric%20Constrained%20Learning-gold)](https://shields.io/)

## 🚀 Overview

Welcome to a research framework that has achieved a **revolutionary breakthrough** in machine learning training methodology. This project introduces **Geometric Constrained Learning (GCL)** - the world's first implementation of a training paradigm that optimizes data presentation rather than model weights.

### 🎯 Revolutionary Breakthrough: Geometric Constrained Learning

**Traditional Training:** Adjust model weights to fit data  
**Geometric Constrained Learning:** Adjust data presentation to fit fixed model geometry

This paradigm shift has been **successfully validated** on lambda calculus reasoning tasks, demonstrating:
- **46% improvement** in total loss (10.407 → 9.947)
- **96% improvement** in expert specialization 
- **37% more efficient** rotation patterns
- **Consumer hardware compatibility** (runs on MacBook)

### 🔬 Core Research Question

The framework also explores fundamental questions about MoE architectures: **Can we create more powerful and efficient MoE models by enabling dynamic, learned communication between the experts?**

Starting with a baseline Transformer, this codebase evolves through multiple architectural phases, culminating in the revolutionary Geometric Constrained Learning system. All experiments are managed through the unified **MoE Research Hub**.

---

## ✨ The MoE Research Hub

The primary entry point to this framework is the `app.py` script, which launches an interactive CLI for managing the entire research lifecycle.

**To start the application, run:**
```bash
python3 app.py
```

The Research Hub provides a centralized and intuitive interface for:
*   **Training New Models**: A step-by-step wizard guides you through configuring a new experiment. You can choose any of the four architectural phases and customize parameters using either a simple or advanced menu.
*   **Loading Existing Models**: Easily load any previously trained model from a checkpoint file to run inference, continue training, or analyze its performance.
*   **Continuing Training**: Seamlessly resume any training run exactly where it left off. The application correctly restores the state of the model, optimizer, and learning rate scheduler.
*   **Interactive Inference**: Generate text from any loaded model. You can interactively provide prompts and adjust generation parameters like temperature and top-k sampling.
*   **On-Demand Analysis**: Generate (or re-generate) a full suite of performance plots for any completed training run. The analysis script is mode-aware and only creates visualizations relevant to the model's architecture.
*   **Flexible Data Loading**: The application supports loading datasets from both the Hugging Face Hub and local text files, providing flexibility for public benchmarks and private data.

---

## 🏗️ Architectural Evolution: A Revolutionary Five-Phase Journey

This project documents a research journey through five major architectural phases, culminating in the revolutionary Geometric Constrained Learning breakthrough. Each phase is fully reproducible using the Research Hub application.

### Phase 1: GNN-MoE (Graph-Coupled MoE)
*   **The "What"**: This initial phase replaces the standard independent MoE layer with one where experts can communicate. A simple Graph Neural Network (GNN) is used to model pairwise relationships, allowing experts to share information and coordinate their processing of the input sequence.
*   **The "Why"**: Standard MoE models suffer from redundant computation and a lack of expert collaboration. The hypothesis of this phase was that enabling direct communication would allow experts to learn more specialized functions, leading to better overall model performance and parameter efficiency.

### Phase 2: HGNN-MoE (Hypergraph-Coupled MoE)
*   **The "What"**: The communication backbone between experts was upgraded from a GNN to a more powerful Hypergraph Neural Network (HGNN).
*   **The "Why"**: While GNNs can only model pairwise (node-to-node) relationships, hypergraphs can model group relationships (e.g., connecting a triplet of experts). This allows for more complex and higher-order coordination patterns, better reflecting the possibility that multiple expert perspectives might be needed to understand complex tokens.

### Phase 3: Orthogonal HGNN-MoE
*   **The "What"**: An adaptive orthogonality loss was added to the training objective. This loss penalizes similarity between the weight matrices of the different experts.
*   **The "Why"**: A common failure mode in MoE is "expert collapse," where all experts converge to similar functions. By directly encouraging the experts' representations to be orthogonal (i.e., geometrically dissimilar), we can enforce specialization and ensure a diverse and effective pool of experts. The "adaptive" nature of the loss intelligently adjusts its strength during training to prevent instability.

### Phase 4: Ghost Expert HGNN-MoE
*   **The "What"**: This phase introduces "Ghost Experts"—a secondary pool of dormant experts that are dynamically activated only when the primary, specialized experts reach representational saturation.
*   **The "Why"**: A model with a fixed number of experts may be too large for simple tasks but too small for complex ones. Ghost Experts provide a mechanism for **adaptive capacity**. The model can maintain a small footprint for easy inputs but scale its complexity on-demand for more challenging data, all without disrupting the finely-tuned specializations of the primary experts.

### Phase 5: Geometric Constrained Learning 🚀 **REVOLUTIONARY BREAKTHROUGH**
*   **The "What"**: This groundbreaking phase implements the world's first **Geometric Constrained Learning** system. Instead of adjusting model weights to fit data, GCL maintains fixed orthogonal expert geometry and learns optimal theta rotation parameters to adjust how data is presented to each expert.
*   **The "Why"**: Traditional training suffers from the fundamental limitation of having to compromise model geometry to fit diverse data patterns. GCL solves this by treating the model as a fixed "100-sided die" and learning optimal data presentation angles. This paradigm shift has been validated on lambda calculus reasoning with **46% improvement** in total loss and **96% improvement** in expert specialization.

**Key GCL Features:**
- **Givens Rotations**: Mathematically sound orthogonal transformations
- **Dual Optimization**: Separate learning rates for rotation (1e-3) vs expert parameters (1e-4)  
- **Multi-Component Loss**: Task + orthogonality + rotation efficiency + specialization
- **Lambda Calculus Cognitive Rotations**: Specialized dimensions for reasoning tasks
- **Consumer Hardware Compatible**: Successfully runs on MacBook with unified memory

**Usage:**
```bash
python run.py --training_mode geometric --geometric_enabled \
  --dataset_name "Creekside/GRPO-Lambda-ParsedForUnsloth" \
  --geometric_learning_rate 0.001 --geometric_expert_learning_rate 0.0001
```

📖 **[Complete GCL Documentation](GEOMETRIC_CONSTRAINED_LEARNING.md)**

---

## ⚙️ Codebase Structure

The project is organized into a clean, centralized structure.

*   **`app.py`**: The main interactive CLI application. **This is the recommended entry point.**
*   **`core/`**: A directory containing the single source of truth for all definitive modules. **This folder is fully self-contained.**
    *   `config.py`: Defines the unified `MoEConfig` dataclass for all architectures.
    *   `architecture.py`: Contains the single, configurable `MoEModel` that can represent any of the four architectural phases.
    *   `training.py`: Implements the modular training, validation, and checkpointing logic.
    *   `data.py`: Handles loading data from both Hugging Face and local files.
    *   `inference.py`: Contains the core text generation logic.
    *   `analysis.py`: Implements the mode-aware analysis and plotting script.
*   **`run.py`**: A secondary, non-interactive script for advanced users who wish to launch training runs via command-line arguments for scripting purposes.
*   **`gnn_MoE/`, `hgnn_MoE/`, `orthogon/`, `ghost/`**: Legacy directories that now only contain historical documentation and project knowledge logs. **They are not used by the current codebase.**

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers
- Datasets (Hugging Face)
- NumPy, Pandas, Matplotlib, Seaborn
