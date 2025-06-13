#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py

A cross-platform setup script for the GNN-MoE-2 environment.
This script creates a virtual environment, installs dependencies,
and handles platform-specific requirements.
"""

import os
import sys
import subprocess
import shutil

# --- Configuration ---
VENV_DIR = "venv"
PYTHON_EXE = os.path.join(VENV_DIR, 'bin', 'python') if sys.platform != 'win32' else os.path.join(VENV_DIR, 'Scripts', 'python.exe')
PIP_EXE = os.path.join(VENV_DIR, 'bin', 'pip') if sys.platform != 'win32' else os.path.join(VENV_DIR, 'Scripts', 'pip.exe')

# --- Helper Functions ---
def run_command(command, description):
    """Runs a command and prints its description."""
    print(f"--- {description} ---")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✅ {description} successful.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during: {description}")
        print(e)
        sys.exit(1)

def create_venv():
    """Creates a virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"--- Creating virtual environment in '{VENV_DIR}' ---")
        run_command(f"{sys.executable} -m venv {VENV_DIR}", "Virtual environment creation")
    else:
        print(f"--- Virtual environment already exists in '{VENV_DIR}' ---")

def install_dependencies():
    """Installs all dependencies using the notebook's sequence."""
    
    # 1. Install basic packages
    run_command(f"{PIP_EXE} install torch datasets numpy matplotlib seaborn", "Installing base packages")
    
    # 2. Uninstall conflicting versions
    run_command(f"{PIP_EXE} uninstall -y datasets fsspec huggingface_hub transformers tokenizers", "Uninstalling conflicting packages")
    
    # 3. Clear HuggingFace cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        print(f"--- Clearing HuggingFace datasets cache at {cache_dir} ---")
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared.\n")

    # 4. Install specific working versions
    run_command(f"{PIP_EXE} install datasets==2.14.7 fsspec==2023.10.0 huggingface_hub==0.17.3 transformers==4.35.2 tokenizers==0.15.0", "Installing specific package versions")
    
    # 5. Install PyTorch Geometric
    run_command(f"{PIP_EXE} install torch-geometric", "Installing PyTorch Geometric")
    
    # 6. Install PyG extensions with CUDA-specific wheels
    # Note: This wheel is for CUDA 12.4 and PyTorch 2.6.0. Adjust if your system differs.
    # We will attempt to get the torch version dynamically.
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]
        # A simple mapping for CUDA, assuming a common case. This might need adjustment.
        cuda_version = "cu121" if "12.1" in torch.version.cuda else "cu124"
        pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"
        run_command(f"{PIP_EXE} install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f {pyg_wheel_url}", f"Installing PyG extensions for torch {torch_version}")
    except Exception as e:
        print(f"⚠️ Could not dynamically determine PyTorch/CUDA version for PyG wheels: {e}")
        print("Attempting a generic install. If this fails, please install manually.")
        run_command(f"{PIP_EXE} install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv", "Installing PyG extensions (generic)")


if __name__ == "__main__":
    print("🚀 Starting GNN-MoE-2 Environment Setup 🚀\n")
    create_venv()
    install_dependencies()
    print("🎉 Environment setup complete! 🎉")
    print(f"To activate the environment, run:")
    if sys.platform != 'win32':
        print(f"source {VENV_DIR}/bin/activate")
    else:
        print(f"{VENV_DIR}\\Scripts\\activate")
