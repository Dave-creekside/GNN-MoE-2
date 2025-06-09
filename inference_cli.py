#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_cli.py

A simple command-line interface to test the inference module.
"""

import torch
import os
import argparse
import json
from transformers import AutoTokenizer

from core.config import MoEConfig
from core.architecture import MoEModel
from core.inference import generate_text

def main():
    parser = argparse.ArgumentParser(description="Simple Inference CLI for MoE Models")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument('--prompt', type=str, required=True, help="The text prompt to start generation from.")
    parser.add_argument('--max_length', type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.8, help="Controls randomness. Higher is more random.")
    parser.add_argument('--top_k', type=int, default=50, help="Filters to top-k most likely tokens.")
    args = parser.parse_args()

    # --- Load Configuration ---
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in checkpoint directory: {checkpoint_dir}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # The config is saved as a dict, so we load it using the from_dict classmethod
    config = MoEConfig.from_dict(config_dict)
    print("✅ Configuration loaded.")

    # --- Setup Device and Tokenizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Running on device: {device}")

    # --- Load Model ---
    model = MoEModel(config).to(device)
    
    # Use torch.load with weights_only=False as we are loading a full checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {args.checkpoint_path}")

    # --- Generate Text ---
    generated_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )

    # --- Print Result ---
    print("\n" + "="*20 + " GENERATED TEXT " + "="*20)
    print(generated_output)
    print("="*56 + "\n")

if __name__ == "__main__":
    main()
