#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_training_sweep.py

Runs a hyperparameter sweep focused on training-related parameters.
"""
import os
from sweep_framework import SweepRunner, load_sweep_config

def main():
    """
    Loads the training sweep config and executes the sweep.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, 'sweep_configs', 'training_sweep.json')
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Sweep config file not found at {config_path}")
        return
        
    print("Loading sweep configuration for training parameters...")
    sweep_config = load_sweep_config(config_path)
    
    runner = SweepRunner(sweep_config=sweep_config, sweep_name="training")
    runner.run()

if __name__ == "__main__":
    main()
