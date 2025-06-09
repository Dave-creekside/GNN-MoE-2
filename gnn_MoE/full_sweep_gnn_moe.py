#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_gnn_moe.py

Runs hyperparameter sweeps for the GNN-MoE model by calling run_gnn_moe.py
with different configurations and logs results to a CSV.
"""
import subprocess
import csv
import os
import itertools
import json # For parsing summary JSON
import datetime
import argparse # Added for sweep_name argument
import re # For parsing floating point numbers

# --- Define Hyperparameter Configurations for Different Sweeps ---
# Baseline values (used when a parameter is not being swept)
baseline_config = {
    'embed_dim': [384],
    'num_layers': [4],
    'num_experts': [4],
    'batch_size': [64], 
    'learning_rate': [3e-4],
    'gnn_layers': [2], 
    'dropout_rate': [0.1],
    'epochs': [10], 
    'max_batches_per_epoch': [-1],
    'dataset_name': ['wikitext'],
    'dataset_config_name': ['wikitext-2-v1'],
    'num_train_samples': [-1], 
    'num_eval_samples': [-1],  
    'eval_every': [250], 
    'num_workers_dataloader': [2],
    'seed': [42]
}

sweep_configs = {
    "embed_dim_sweep": {
        **baseline_config, # Start with baseline
        'embed_dim': [256, 384, 512], # Vary this
    },
    "num_layers_sweep": {
        **baseline_config,
        'num_layers': [4, 6, 8],
    },
    "num_experts_sweep": {
        **baseline_config,
        'num_experts': [2, 4, 8, 16],
    },
    "lr_sweep": {
        **baseline_config,
        'learning_rate': [1e-3, 5e-4, 3e-4, 1e-4],
    },
    "batch_size_sweep": {
        **baseline_config,
        'batch_size': [32, 64, 128],
    },
    "full_factorial_small": { # A smaller version of the original multi-param sweep
        'embed_dim': [256, 384],
        'num_layers': [4],
        'num_experts': [4, 8],
        'batch_size': [32, 64],
        'learning_rate': [5e-4, 3e-4],
        # Fixed params for this sweep
        'gnn_layers': [2], 'dropout_rate': [0.1],'epochs': [10], 'max_batches_per_epoch': [-1],
        'dataset_name': ['wikitext'],'dataset_config_name': ['wikitext-2-v1'],
        'num_train_samples': [-1],'num_eval_samples': [-1],'eval_every': [250], 
        'num_workers_dataloader': [2],'seed': [42]
    }
}

# --- Argument Parser for Sweep Script ---
parser_sweep = argparse.ArgumentParser(description="GNN-MoE Sweep Orchestrator")
parser_sweep.add_argument('--sweep_name', type=str, default="full_factorial_small", 
                          choices=list(sweep_configs.keys()), 
                          help="Name of the sweep configuration to run.")
args_sweep = parser_sweep.parse_args()

selected_sweep_params = sweep_configs[args_sweep.sweep_name]
print(f"Selected sweep configuration: {args_sweep.sweep_name}")

# --- CSV Output Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"sweep_results_{args_sweep.sweep_name}_{timestamp}.csv" # Include sweep_name in CSV filename

# Define fieldnames for the CSV: all potential params + key results
# Create a union of all keys from baseline_config and the selected_sweep_params to ensure all columns are present
all_possible_param_names = list(dict(baseline_config, **selected_sweep_params).keys())
csv_fieldnames = all_possible_param_names + [
    'run_name', 'total_params_str', 'best_eval_loss', 
    'best_eval_perplexity', 'training_time_min', 'data_mode', 'error_message'
]
# Ensure no duplicates in fieldnames, preserving order
seen_fields = set()
unique_csv_fieldnames = [x for x in csv_fieldnames if not (x in seen_fields or seen_fields.add(x))]


with open(csv_filename, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=unique_csv_fieldnames)
    writer.writeheader()
    print(f"📝 Results will be saved to: {csv_filename}")

# --- Generate all combinations for the selected sweep ---
param_names = list(selected_sweep_params.keys())
param_value_lists = [selected_sweep_params[k] for k in param_names]
all_combinations = list(itertools.product(*param_value_lists))

print(f"🚀 Starting '{args_sweep.sweep_name}' with {len(all_combinations)} combinations.")

# --- Main Sweep Loop ---
for i, combo in enumerate(all_combinations):
    current_run_params = dict(zip(param_names, combo))
    
    # Construct a descriptive run_name
    run_name_parts = [f"sweep_{timestamp}_run{i+1}"]
    # Add key varying parameters to the run_name for quick identification
    # Use selected_sweep_params to check which params are being varied in *this specific sweep*
    for k, v_list in selected_sweep_params.items(): 
        if len(v_list) > 1: 
            current_val = current_run_params[k]
            short_k = k.replace('_dim','D').replace('_layers','L').replace('_experts','E').replace('_rate','R').replace('_size','BS').replace('learning','lr')
            run_name_parts.append(f"{short_k}{current_val}")
    run_name = "_".join(run_name_parts)
            
    print(f"\n--- Running Combination {i+1}/{len(all_combinations)} ({args_sweep.sweep_name}): {run_name} ---")
    print(f"Parameters: {current_run_params}")

    command = ["python", "run_gnn_moe.py", "--run_name", run_name]
    for param_name, param_val in current_run_params.items():
        command.append(f"--{param_name}")
        command.append(str(param_val))
    
    command.append("--checkpoint_dir")
    # Store checkpoints for sweep runs in a dedicated base directory, sub-foldered by run_name
    command.append("checkpoints_sweep_runs") 
    command.append("--quiet") # Add quiet flag for cleaner sweep logs

    # Ensure all possible params are in result_row for consistent CSV writing
    result_row = {key: current_run_params.get(key, baseline_config.get(key, [None])[0]) for key in all_possible_param_names}
    result_row['run_name'] = run_name
    result_row['error_message'] = '' 

    try:
        # Using Popen to stream output, but can be complex.
        # For simplicity in parsing, consider check_output if runs are not excessively long,
        # or redirect output to a file per run and parse that.
        # For now, live printing and parsing from stdout.
        
        # Ensure PYTHONUNBUFFERED is set for live output in some environments
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        print(f"Executing: {' '.join(command)}")
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1, universal_newlines=True)
        
        full_stdout = []
        print(f"\n--- Live Output for {run_name} ---")
        for line in process.stdout:
            print(line, end='') # Print live output
            full_stdout.append(line)
        
        process.wait() # Wait for the subprocess to complete
        return_code = process.returncode
        
        # Join all captured stdout for parsing fallback
        completed_stdout = "".join(full_stdout)

        # Try to load results from JSON summary
        summary_file_path = os.path.join("checkpoints_sweep_runs", run_name, "run_summary.json")
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as f:
                    summary_data = json.load(f)
                result_row['best_eval_loss'] = summary_data.get('best_eval_loss', float('nan'))
                result_row['best_eval_perplexity'] = summary_data.get('best_eval_perplexity', float('nan'))
                result_row['data_mode'] = summary_data.get('data_mode', 'N/A')
            except Exception as json_e:
                print(f"Error parsing summary JSON for {run_name}: {json_e}")
                result_row['error_message'] += f"; JSON_parse_error: {json_e}"
        else:
            print(f"⚠️ Summary JSON file not found for {run_name} at {summary_file_path}")
            result_row['error_message'] += "; SummaryJSONNotFound"

        # Fallback parsing for time and params from the full captured stdout
        time_match = re.search(r"Total time for this run:\s*([\d\.]+)\s*minutes", completed_stdout)
        if time_match: result_row['training_time_min'] = float(time_match.group(1))
        else: result_row['training_time_min'] = float('nan')

        params_match = re.search(r"Total Parameters:\s*([\d,]+)", completed_stdout)
        if params_match: result_row['total_params_str'] = params_match.group(1).replace(',','')
        else: result_row['total_params_str'] = "N/A"
            
        if return_code != 0:
            print(f"⚠️ Run {run_name} FAILED with return code {return_code}")
            result_row['error_message'] = result_row.get('error_message','') + f"; Failed_code_{return_code}"
            if result_row.get('best_eval_loss', float('nan')) is float('nan'): # If no loss from JSON
                 result_row['best_eval_loss'] = "RUN_FAILED"
        else:
            print(f"✅ Run {run_name} completed (return code 0).")

    except Exception as e:
        print(f"❌ CRITICAL ERROR launching/managing subprocess for {run_name}: {e}")
        result_row['error_message'] = str(e)
        result_row['best_eval_loss'] = "SUBPROCESS_ERROR"
    
    finally: # Ensure results are written even if errors occur
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=unique_csv_fieldnames)
            writer.writerow(result_row)

print(f"\n🎉 Hyperparameter sweep '{args_sweep.sweep_name}' finished! All results saved to {csv_filename}")
