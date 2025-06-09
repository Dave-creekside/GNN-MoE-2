# Sweep Functionality: Debugging and Enhancement Log

**Date:** 2025-06-07

## 1. Objective

The goal was to create a hyperparameter sweep framework to test the new Ghost Expert architecture. This involved creating a `run_sweep.py` script to programmatically execute training runs with different hyperparameter combinations and organize the results.

## 2. Initial Implementation and Debugging

The initial implementation of the sweep framework encountered a series of technical issues that required systematic debugging.

### 2.1. `ImportError: attempted relative import with no known parent package`
- **Problem:** The initial execution command (`python ghost/run_gnn_moe.py`) did not treat the `ghost` directory as a Python package, causing all relative imports within the module to fail.
- **Solution:**
    1.  Created `ghost/__init__.py` and `ghost/tests/__init__.py` to formally define the directories as Python packages.
    2.  Modified the execution command in `run_sweep.py` to the module-aware `python -m ghost.run_gnn_moe`.
    3.  Standardized all intra-package imports to be relative (e.g., `from . import gnn_moe_config`).

### 2.2. `NameError: name 'fields' is not defined`
- **Problem:** The `run_gnn_moe.py` script used the `fields` function from the `dataclasses` module to auto-generate command-line arguments but was missing the corresponding import.
- **Solution:** Added `from dataclasses import fields` to `run_gnn_moe.py`.

### 2.3. `TypeError` and `unrecognized arguments`
- **Problem:** The `argparse` library, when used with `**vars(args)`, had issues with `Optional[str]` type hints and boolean flags (`action='store_true'`). This led to errors where it tried to pass `None` or `True` as invalid values to arguments.
- **Solution:**
    1.  Removed all `Optional[str]` type hints from the `GhostMoEConfig` dataclass, changing them to `str = None`.
    2.  Modified the `run_sweep.py` script to handle boolean flags correctly, appending only the flag (e.g., `--quiet`) without a value.
    3.  Modified `run_gnn_moe.py` to handle the `--quiet` flag separately and pop it from the arguments dictionary before passing the remaining arguments to the `GhostMoEConfig` constructor.

## 3. Output Logging and Analysis Deficiencies

After fixing the execution bugs, a significant logical flaw was identified: the sweep was running but producing minimal, unhelpful output (a single JSON file with one data point per run).

- **Problem:** The framework was not designed to capture the detailed, step-by-step metrics needed for proper hyperparameter analysis, nor was it generating any visualizations.
- **Correction & User Feedback:** I initially proposed a fix based on the incorrect assumption that a `checkpoints` directory was being created. The user correctly pointed out that this directory did not exist and that my reasoning was flawed. This was a critical correction.

## 4. Revised Plan for Enhanced Logging and Analysis

Based on the user's feedback, a new, more robust plan was formulated to make the sweep framework genuinely useful.

1.  **Detailed JSON Logging:** The `gnn_moe_training.py` script will be modified to log a complete list of metric snapshots (dictionaries) at each evaluation step, capturing everything from losses to ghost activations. This detailed log will be saved as `training_log.json` for each run.
2.  **Automated Plotting & Analysis:** A new script, `ghost/tests/analyze_sweep.py`, will be created. After the sweep completes, this script will automatically:
    -   Find all `training_log.json` files.
    -   Generate and save plots for each individual run, visualizing the training dynamics.
    -   Generate and save comparison plots that overlay the key metrics from all runs, providing a clear visual summary of the hyperparameter effects.
    -   Create a `sweep_summary.csv` file for a quick, tabular overview of the final performance of each run.
3.  **Directory Creation Fix:** The `run_gnn_moe.py` script will be updated to proactively create the checkpoint directory, preventing errors when it needs to save the log file.

This revised plan will ensure the sweep framework produces valuable, actionable insights.
