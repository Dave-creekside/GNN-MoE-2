# GNN-Coupled Mixture of Experts (GNN-MoE) Language Model

## Overview

This project implements a Transformer-based language model utilizing a Mixture of Experts (MoE) architecture. The core innovation lies in employing Graph Neural Networks (GNNs) to facilitate dynamic, learned communication and coordination among the experts within each MoE layer. The aim is to explore if GNN-based coordination can lead to more effective expert specialization and utilization.

## Key Features

*   **Modular Codebase:** Organized into separate Python modules for configuration, architecture, data handling, training, and analysis.
*   **GNN Expert Coupling:** Experts in each MoE layer communicate via GNN-based "coupler" modules.
*   **Comprehensive Configuration:** Single training runs are highly configurable via command-line arguments.
*   **Automated Sweeps:** Includes scripts for both single-parameter sweeps and more complex multi-parameter sweeps.
*   **Checkpointing:** Supports saving and resuming training.
*   **Analysis Utilities:** Tools for visualizing training progress and expert communication patterns.

## File Structure

*   `gnn_moe_config.py`: Defines the `GNNMoEConfig` dataclass for all hyperparameters.
*   `gnn_moe_architecture.py`: Contains all PyTorch `nn.Module` classes for the model.
*   `gnn_moe_data.py`: Handles data loading using Hugging Face `datasets`.
*   `gnn_moe_training.py`: Implements the training loop, evaluation, and checkpointing logic.
*   `gnn_moe_analysis.py`: Provides functions for plotting results and analyzing model behavior.
*   `run_gnn_moe.py`: Main executable for launching a single training run.
*   `sweep_param_*.py`: Individual scripts for sweeping a single hyperparameter (e.g., `sweep_param_embed_dim.py`). Values to sweep are set within each script.
*   `full_sweep_gnn_moe.py`: Script for running predefined, potentially multi-parameter sweep configurations.
*   `requirements.txt`: Lists project dependencies.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Setup & Installation

1.  **Prerequisites:** Python 3.9+ is recommended.
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    Install dependencies using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Single Training Experiment (`run_gnn_moe.py`)

Use `run_gnn_moe.py` to launch a single training run with specific hyperparameters. Most parameters defined in `GNNMoEConfig` can be set via CLI arguments.

**Example:**
```bash
python run_gnn_moe.py \
    --embed_dim 256 \
    --num_layers 4 \
    --num_experts 4 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --epochs 10 \
    --run_name my_custom_run \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-v1
```
*   Use `python run_gnn_moe.py --help` to see all available arguments.

### 2. Single-Parameter Sweeps (`sweep_param_*.py`)

Dedicated scripts are provided for sweeping individual hyperparameters.
*   To configure a sweep, edit the `SWEEP_VALUES` list at the top of the respective `sweep_param_*.py` script.
*   Then, execute the script directly.

**Example (sweeping embedding dimension):**
```bash
# 1. Edit SWEEP_VALUES in sweep_param_embed_dim.py if needed
#    e.g., SWEEP_VALUES = [128, 256, 512]
# 2. Run the script:
python sweep_param_embed_dim.py
```
This will iterate through the specified `SWEEP_VALUES`, keeping other parameters at their baseline settings, and log results to a CSV (e.g., `sweep_results_embed_dim_TIMESTAMP.csv`).

**Available single-parameter sweep scripts:**
*   `sweep_param_batch_size.py`
*   `sweep_param_dropout_rate.py`
*   `sweep_param_embed_dim.py`
*   `sweep_param_gnn_layers.py`
*   `sweep_param_learning_rate.py`
*   `sweep_param_num_experts.py`
*   `sweep_param_num_layers.py`

### 3. Multi-Configuration Sweeps (`full_sweep_gnn_moe.py`)

The `full_sweep_gnn_moe.py` script allows running more complex, predefined sweep configurations.
*   Define sweep configurations within the `sweep_configs` dictionary in the script.
*   Select a configuration to run using the `--sweep_name` argument.

**Example:**
```bash
python full_sweep_gnn_moe.py --sweep_name full_factorial_small
```
This will run all combinations defined in the `full_factorial_small` configuration and log results to a CSV (e.g., `sweep_results_full_factorial_small_TIMESTAMP.csv`).

## Output

*   **Checkpoints:**
    *   Single runs (`run_gnn_moe.py`): Saved under `./checkpoints/<run_name>/`.
    *   Sweep runs: Saved under `./checkpoints_sweep_runs/<run_name>/`.
*   **Plots:** Training progress and analysis plots are saved in `./plots/`, prefixed with the `run_name`.
*   **Sweep CSV Results:** Each sweep script execution generates a CSV file in the root directory (e.g., `sweep_results_embed_dim_TIMESTAMP.csv`) detailing the parameters and performance for each run in the sweep.

## License
This project is currently unlicensed. You may choose to add a license file (e.g., MIT, Apache 2.0) later.
