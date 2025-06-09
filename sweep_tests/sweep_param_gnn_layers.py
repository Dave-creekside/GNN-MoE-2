#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_param_gnn_layers.py

Sweeps the 'gnn_layers' hyperparameter for the GNN-MoE model.
"""
import subprocess
import csv
import os
import json
import datetime
import re

# --- Configuration for this specific sweep ---
PARAMETER_BEING_SWEPT = "gnn_layers"
SWEEP_VALUES = [1, 2, 3]  # Values to test for gnn_layers

# Baseline values (used when a parameter is not being swept)
baseline_config = {
    'embed_dim': [384],
    'num_layers': [4],
    'num_experts': [4],
    'batch_size': [64],
    'learning_rate': [3e-4],
    'gnn_layers': [2],  # This will be overridden by SWEEP_VALUES
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

# --- CSV Output Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"sweep_results_{PARAMETER_BEING_SWEPT}_{timestamp}.csv"

all_possible_param_names = list(baseline_config.keys())
csv_fieldnames = all_possible_param_names + [
    'run_name', 'total_params_str', 'best_eval_loss',
    'best_eval_perplexity', 'training_time_min', 'data_mode', 'error_message'
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
    current_run_params = {key: val[0] for key, val in baseline_config.items()}
    current_run_params[PARAMETER_BEING_SWEPT] = sweep_value

    run_name_parts = [f"sweep_{PARAMETER_BEING_SWEPT}_{timestamp}_run{i+1}"]
    short_k = PARAMETER_BEING_SWEPT.replace('_dim','D').replace('_layers','L').replace('_experts','E').replace('_rate','R').replace('_size','BS').replace('learning','lr')
    run_name_parts.append(f"{short_k}{sweep_value}")
    run_name = "_".join(run_name_parts)

    print(f"\n--- Running Combination {i+1}/{len(SWEEP_VALUES)} ({PARAMETER_BEING_SWEPT}={sweep_value}): {run_name} ---")
    print(f"Parameters: {current_run_params}")

    command = ["python", "run_gnn_moe.py", "--run_name", run_name]
    for param_name, param_val in current_run_params.items():
        command.append(f"--{param_name}")
        command.append(str(param_val))

    command.append("--checkpoint_dir")
    command.append("checkpoints_sweep_runs")
    command.append("--quiet")

    result_row = {key: current_run_params.get(key, baseline_config.get(key, [None])[0]) for key in all_possible_param_names}
    result_row['run_name'] = run_name
    result_row['error_message'] = ''

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print(f"Executing: {' '.join(command)}")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1, universal_newlines=True)
        full_stdout = []
        print(f"\n--- Live Output for {run_name} ---")
        for line in process.stdout:
            print(line, end='')
            full_stdout.append(line)
        process.wait()
        return_code = process.returncode
        completed_stdout = "".join(full_stdout)

        summary_file_path = os.path.join("checkpoints_sweep_runs", run_name, "run_summary.json")
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as f_json:
                    summary_data = json.load(f_json)
                result_row['best_eval_loss'] = summary_data.get('best_eval_loss', float('nan'))
                result_row['best_eval_perplexity'] = summary_data.get('best_eval_perplexity', float('nan'))
                result_row['data_mode'] = summary_data.get('data_mode', 'N/A')
            except Exception as json_e:
                print(f"Error parsing summary JSON for {run_name}: {json_e}")
                result_row['error_message'] += f"; JSON_parse_error: {json_e}"
        else:
            print(f"⚠️ Summary JSON file not found for {run_name} at {summary_file_path}")
            result_row['error_message'] += "; SummaryJSONNotFound"

        time_match = re.search(r"Total time for this run:\s*([\d\.]+)\s*minutes", completed_stdout)
        if time_match: result_row['training_time_min'] = float(time_match.group(1))
        else: result_row['training_time_min'] = float('nan')

        params_match = re.search(r"Total Parameters:\s*([\d,]+)", completed_stdout)
        if params_match: result_row['total_params_str'] = params_match.group(1).replace(',','')
        else: result_row['total_params_str'] = "N/A"

        if return_code != 0:
            print(f"⚠️ Run {run_name} FAILED with return code {return_code}")
            result_row['error_message'] = result_row.get('error_message','') + f"; Failed_code_{return_code}"
            if result_row.get('best_eval_loss', float('nan')) is float('nan'):
                 result_row['best_eval_loss'] = "RUN_FAILED"
        else:
            print(f"✅ Run {run_name} completed (return code 0).")

    except Exception as e:
        print(f"❌ CRITICAL ERROR launching/managing subprocess for {run_name}: {e}")
        result_row['error_message'] = str(e)
        result_row['best_eval_loss'] = "SUBPROCESS_ERROR"

    finally:
        with open(csv_filename, 'a', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=unique_csv_fieldnames)
            writer.writerow(result_row)

print(f"\n🎉 Sweep for '{PARAMETER_BEING_SWEPT}' finished! All results saved to {csv_filename}")
