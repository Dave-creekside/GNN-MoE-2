#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_architecture_sweep.py

Runs a hyperparameter sweep focused on model architecture parameters.
"""
import os
from sweep_framework import SweepRunner, load_sweep_config

def main():
    """
    Loads the architecture sweep config and executes the sweep.
    """
    # Construct the path to the config file relative to this script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, 'sweep_configs', 'architecture_sweep.json')
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Sweep config file not found at {config_path}")
        return
        
    print("Loading sweep configuration for architecture...")
    sweep_config = load_sweep_config(config_path)
    
    runner = SweepRunner(sweep_config=sweep_config, sweep_name="architecture")
    runner.run()

if __name__ == "__main__":
    main()
