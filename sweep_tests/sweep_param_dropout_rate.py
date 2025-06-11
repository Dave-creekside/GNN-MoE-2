#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_param_dropout_rate.py

Sweeps the 'dropout_rate' hyperparameter for the GCL system.
Updated to use run.py with current parameter names and robust OOM handling.
"""
import subprocess
import csv
import os
import json
import datetime
import re
import torch

# --- Configuration for this specific sweep ---
PARAMETER_BEING_SWEPT = "dropout_rate"
SWEEP_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # Values to test for dropout_rate

# Baseline values optimized for GCL system with wikitext-2
baseline_config = {
    'training_mode': ['geometric'],
    'geometric_enabled': [True],
    'geometric_learning_rate': [0.001],
    'geometric_expert_learning_rate': [0.0001], 
    'geometric_rotation_dimensions': [4],
    'geometric_orthogonality_weight': [0.5],
    'geometric_rotation_efficiency_weight': [0.2],
    'geometric_specialization_weight': [0.3],
    'geometric_lambda_cognitive_rotations': [False],  # False for wikitext
    'embed_dim': [128],
    'num_layers': [4],
    'num_experts': [2],
    'batch_size': [2],
    'learning_rate': [3e-4],
    'dropout_rate': [0.1],  # This will be overridden by SWEEP_VALUES
    'epochs': [3],
    'max_batches_per_epoch': [50],
    'eval_every': [25],
    'dataset_name': ['wikitext'],
    'dataset_config_name': ['wikitext-2-v1'],
    'dataset_source': ['huggingface'],
    'num_train_samples': [-1],
    'num_eval_samples': [-1],
    'num_workers_dataloader': [2],
    'seed': [42]
}

def cleanup_memory():
    """Clean up GPU memory between runs to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        import gc
        gc.collect()

def categorize_error(stdout_text, stderr_text=""):
    """Categorize the type of error that occurred."""
    combined_text = (stdout_text + " " + stderr_text).lower()
    
    if "cuda out of memory" in combined_text or "cublas_status_alloc_failed" in combined_text:
        return "OOM_CUDA"
    elif "mps out of memory" in combined_text or "mps backend out of memory" in combined_text:
        return "OOM_MPS"
    elif "out of memory" in combined_text:
        return "OOM_GENERIC"
    elif "runtime error" in combined_text:
        return "RUNTIME_ERROR"
    elif "import error" in combined_text or "module not found" in combined_text:
        return "IMPORT_ERROR"
    else:
        return "UNKNOWN_ERROR"

# --- CSV Output Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"sweep_results_{PARAMETER_BEING_SWEPT}_{timestamp}.csv"

all_possible_param_names = list(baseline_config.keys())
csv_fieldnames = all_possible_param_names + [
    'run_name', 'total_params_str', 'best_eval_loss',
    'best_eval_perplexity', 'training_time_min', 'data_mode', 
    'error_message', 'error_category', 'memory_peak_mb'
]
seen_fields = set()
unique_csv_fieldnames = [x for x in csv_fieldnames if not (x in seen_fields or seen_fields.add(x))]

with open(csv_filename, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=unique_csv_fieldnames)
    writer.writeheader()
    print(f"📝 Results will be saved to: {csv_filename}")

print(f"🚀 Starting sweep for '{PARAMETER_BEING_SWEPT}' with values: {SWEEP_VALUES}")

# --- Main Sweep Loop ---
for i, sweep_value in enumerate(SWEEP_VALUES):
    cleanup_memory()
    
    current_run_params = {key: val[0] for key, val in baseline_config.items()}
    current_run_params[PARAMETER_BEING_SWEPT] = sweep_value

    run_name_parts = [f"sweep_{PARAMETER_BEING_SWEPT}_{timestamp}_run{i+1:02d}"]
    run_name_parts.append(f"dr{sweep_value}")
    run_name = "_".join(run_name_parts)

    print(f"\n{'='*60}")
    print(f"🧪 Running {i+1}/{len(SWEEP_VALUES)}: {PARAMETER_BEING_SWEPT}={sweep_value}")
    print(f"📁 Run: {run_name}")
    print(f"{'='*60}")

    command = ["python", "run.py", "--run_name", run_name]
    for param_name, param_val in current_run_params.items():
        if isinstance(param_val, bool):
            if param_val:
                command.append(f"--{param_name}")
        else:
            command.append(f"--{param_name}")
            command.append(str(param_val))

    command.extend([
        "--checkpoint_dir", "checkpoints_sweep_runs"
    ])

    result_row = {key: current_run_params.get(key, baseline_config.get(key, [None])[0]) for key in all_possible_param_names}
    result_row['run_name'] = run_name
    result_row['error_message'] = ''
    result_row['error_category'] = ''
    result_row['memory_peak_mb'] = ''

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print(f"Executing: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )

        print(f"\n--- Live Output for {run_name} ---")
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end='')
            output_lines.append(line)
        
        return_code = process.wait()
        
        completed_stdout = "".join(output_lines)
        completed_stderr = "" # Merged into stdout

        if return_code != 0:
            error_category = categorize_error(completed_stdout, completed_stderr)
            result_row['error_category'] = error_category
            print(f"⚠️ Run {run_name} FAILED with return code {return_code} ({error_category})")
        else:
            print(f"✅ Run {run_name} completed successfully!")

        summary_file_path = os.path.join("checkpoints_sweep_runs", run_name, "run_summary.json")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as f_json:
                summary_data = json.load(f_json)
            result_row.update(summary_data)
        else:
            result_row['error_message'] += "; SummaryJSONNotFound"

    except Exception as e:
        print(f"❌ CRITICAL ERROR for {run_name}: {e}")
        result_row['error_message'] = str(e)
        result_row['error_category'] = "SUBPROCESS_ERROR"

    finally:
        with open(csv_filename, 'a', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=unique_csv_fieldnames)
            writer.writerow(result_row)
        cleanup_memory()

print(f"\n🎉 Sweep for '{PARAMETER_BEING_SWEPT}' finished! Results in {csv_filename}")
