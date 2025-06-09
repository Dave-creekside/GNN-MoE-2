import os
import subprocess
import itertools
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List

class SweepRunner:
    """
    A configurable framework for running hyperparameter sweeps.
    """
    def __init__(self, sweep_config: Dict[str, Any], sweep_name: str):
        """
        Initializes the SweepRunner.

        Args:
            sweep_config: A dictionary defining the sweep parameters.
            sweep_name: A descriptive name for the sweep (e.g., 'architecture_sweep').
        """
        self.base_command = sweep_config.get("base_command", "python -m ghost.run_gnn_moe")
        self.sweep_params = sweep_config.get("sweep_params", {})
        self.static_params = sweep_config.get("static_params", {})
        self.sweep_name = sweep_name

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_root_dir = os.path.join('ghost', 'tests', 'sweeps', f'{self.sweep_name}_{self.timestamp}')
        os.makedirs(self.sweep_root_dir, exist_ok=True)
        print(f"📁 Created root directory for this sweep: {self.sweep_root_dir}")

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generates all hyperparameter combinations."""
        param_names = list(self.sweep_params.keys())
        param_values = [self.sweep_params[name]['values'] for name in param_names]
        
        combinations = list(itertools.product(*param_values))
        
        run_configs = []
        for combo in combinations:
            run_config = {}
            for i, value in enumerate(combo):
                run_config[param_names[i]] = value
            run_configs.append(run_config)
            
        return run_configs

    def _construct_run_command(self, run_config: Dict[str, Any]) -> (List[str], str):
        """Constructs the command and run name for a single experiment."""
        cmd = list(self.base_command.split())
        run_name_parts = []

        # Add sweep parameters
        for name, value in run_config.items():
            prefix = self.sweep_params[name].get('prefix', name)
            cmd.extend([f"--{name}", str(value)])
            run_name_parts.append(f"{prefix}_{value:.2e}" if isinstance(value, float) else f"{prefix}_{value}")

        # Add static parameters
        for name, value in self.static_params.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{name}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{name}", str(value)])
        
        run_name = "_".join(run_name_parts)
        run_output_dir = os.path.join(self.sweep_root_dir, run_name)
        
        cmd.extend(["--checkpoint_dir", run_output_dir])
        cmd.extend(["--run_name", run_name])
        
        return cmd, run_name

    def run(self):
        """Executes the full hyperparameter sweep."""
        run_configs = self._generate_combinations()
        print(f"🔬 Starting sweep '{self.sweep_name}' with {len(run_configs)} combinations...")

        for i, run_config in enumerate(run_configs):
            cmd, run_name = self._construct_run_command(run_config)
            
            print(f"\n--- Running Combination {i+1}/{len(run_configs)}: {run_name} ---")
            print(f"   Command: {' '.join(cmd)}")
            
            try:
                # Don't capture output so we can see progress in real time
                result = subprocess.run(cmd, check=True)
                print(f"   ✅ Run {run_name} completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ ERROR: Run failed for combination {run_name}")
                print(f"   Return code: {e.returncode}")
                # Try to get some information about what went wrong
                print("   Check the output above for error details.")
            except Exception as e:
                print(f"   ❌ An unexpected error occurred: {e}")

        print(f"\n🎉 Sweep '{self.sweep_name}' finished!")
        print(f"📊 Check individual run folders for logs and plots inside {self.sweep_root_dir}")

def load_sweep_config(config_path: str) -> Dict[str, Any]:
    """Loads a sweep configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)
