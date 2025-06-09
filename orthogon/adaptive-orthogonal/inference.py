#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py

Modular inference script for GNN-Coupled MoE models with adaptive weight orthogonality.
Designed for easy use in Google Colab and other environments.
"""

import torch
import torch.nn.functional as F
import json
import argparse
import os
import random
import numpy as np
from transformers import AutoTokenizer
from typing import Optional, Dict, Any

# Import local modules
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("🚀 Using CPU")
    return device


def load_config_from_json(config_path: str) -> GNNMoEConfig:
    """Load GNNMoEConfig from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"📋 Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create a new GNNMoEConfig instance and update its attributes
    config = GNNMoEConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"⚠️ Warning: Unknown config key '{key}' with value '{value}' - skipping")
    
    # Call post_init to ensure any derived values are computed correctly
    config.__post_init__()
    
    print(f"✅ Config loaded: {config.num_experts} experts, {config.embed_dim}d, {config.num_layers} layers")
    print(f"   Vocab size: {config.vocab_size}, Max seq length: {config.max_seq_length}")
    
    return config


def load_model_and_checkpoint(config: GNNMoEConfig, checkpoint_path: str, device: torch.device) -> GNNMoEModel:
    """Load the model and its trained weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"🧠 Creating GNNMoEModel...")
    model = GNNMoEModel(config)
    
    print(f"🔄 Loading weights from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Assume the checkpoint is the state_dict directly
            model_state_dict = checkpoint
        
        model.load_state_dict(model_state_dict)
        print("✅ Model weights loaded successfully")
        
        # Print some checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"   Checkpoint from epoch {checkpoint['epoch']}")
            if 'best_eval_loss' in checkpoint:
                print(f"   Best eval loss: {checkpoint['best_eval_loss']:.4f}")
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model loaded: {total_params:,} total parameters")
    
    return model


def load_tokenizer(tokenizer_name: Optional[str], config: GNNMoEConfig) -> AutoTokenizer:
    """Load the appropriate tokenizer."""
    # Determine tokenizer name
    if tokenizer_name:
        print(f"🔤 Using specified tokenizer: {tokenizer_name}")
        final_tokenizer_name = tokenizer_name
    else:
        # Try to infer from config
        if hasattr(config, 'dataset_name') and config.dataset_name:
            # Common mapping for dataset names to tokenizers
            dataset_to_tokenizer = {
                'wikitext': 'gpt2',
                'openwebtext': 'gpt2',
                'pile': 'EleutherAI/gpt-neo-125M',
            }
            
            # Check if dataset name maps to a known tokenizer
            for dataset_key, tokenizer_default in dataset_to_tokenizer.items():
                if dataset_key.lower() in config.dataset_name.lower():
                    final_tokenizer_name = tokenizer_default
                    print(f"🔤 Inferred tokenizer from dataset '{config.dataset_name}': {final_tokenizer_name}")
                    break
            else:
                # Default fallback
                final_tokenizer_name = 'gpt2'
                print(f"🔤 Using default tokenizer (couldn't infer from dataset '{config.dataset_name}'): {final_tokenizer_name}")
        else:
            # Ultimate fallback
            final_tokenizer_name = 'gpt2'
            print(f"🔤 Using default tokenizer: {final_tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(final_tokenizer_name)
        
        # Ensure pad_token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"📝 Set pad_token to eos_token: '{tokenizer.eos_token}'")
        
        # Validate vocab size matches config
        tokenizer_vocab_size = len(tokenizer)
        if tokenizer_vocab_size != config.vocab_size:
            print(f"⚠️  WARNING: Tokenizer vocab size ({tokenizer_vocab_size}) != config vocab size ({config.vocab_size})")
            print(f"   This may cause errors during inference. Consider using a different tokenizer.")
            print(f"   You can override with --tokenizer_name argument.")
        else:
            print(f"✅ Tokenizer vocab size matches config: {tokenizer_vocab_size}")
        
        return tokenizer
    
    except Exception as e:
        print(f"❌ Error loading tokenizer '{final_tokenizer_name}': {e}")
        raise


def apply_sampling(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Apply temperature, top-k, and top-p sampling to logits."""
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Ensure top_k doesn't exceed vocab size
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        # Set all other logits to -inf
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
        logits = logits_filtered
    
    # Apply top-p (nucleus) sampling
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    return logits


def generate_text(
    model: GNNMoEModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: Optional[int] = None
) -> str:
    """Generate text using autoregressive decoding."""
    
    if seed is not None:
        set_seed(seed)
    
    print(f"🎯 Generating {max_new_tokens} new tokens...")
    print(f"   Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"📝 Prompt encoded to {input_ids.shape[1]} tokens")
    
    # Generation loop
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get current sequence length
            current_length = generated_ids.shape[1]
            
            # Check if we exceed max sequence length
            if current_length >= model.config.max_seq_length:
                print(f"⚠️  Reached max sequence length ({model.config.max_seq_length}), stopping generation")
                break
            
            # Forward pass
            outputs = model(generated_ids, attention_mask=attention_mask, return_loss=False)
            logits = outputs['logits']
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
            
            # Apply sampling
            filtered_logits = apply_sampling(next_token_logits, temperature, top_k, top_p)
            
            # Sample next token
            if temperature == 0.0 or (top_k == 1):
                # Greedy decoding
                next_token_id = torch.argmax(filtered_logits, dim=-1, keepdim=True)
            else:
                # Multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
            
            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                print(f"📄 EOS token generated at step {step + 1}")
                break
            
            # Progress indicator for longer generations
            if (step + 1) % 20 == 0:
                print(f"   Generated {step + 1}/{max_new_tokens} tokens...")
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract only the newly generated part
    prompt_length = len(prompt)
    new_text = generated_text[prompt_length:]
    
    print(f"✅ Generation complete: {generated_ids.shape[1] - input_ids.shape[1]} new tokens")
    
    return generated_text, new_text


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for GNN-Coupled MoE models with adaptive weight orthogonality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --checkpoint_path checkpoints/best_model.pth.tar --config_path checkpoints/config.json --prompt "The future of AI is"
  
  python inference.py --checkpoint_path best_model.pth.tar --config_path config.json --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.8 --top_k 40
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth.tar)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config JSON file')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt to start generation from')
    
    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature. Lower = more deterministic (default: 0.7)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling. 0 to disable (default: 50)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling. 0 to disable (default: 0.9)')
    
    # Model and system parameters
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help='Hugging Face tokenizer name or path (default: auto-detect)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for inference (default: auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible generation')
    
    # Output options
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save generated text to file (optional)')
    parser.add_argument('--show_new_only', action='store_true',
                        help='Display only the newly generated text (without prompt)')
    
    args = parser.parse_args()
    
    print("🚀 GNN-MoE Adaptive Orthogonal Inference")
    print("=" * 50)
    
    # Set random seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Determine device
    if args.device == 'auto':
        device = detect_device()
    else:
        device = torch.device(args.device)
        print(f"🚀 Using specified device: {device}")
    
    try:
        # Load configuration
        config = load_config_from_json(args.config_path)
        
        # Load model
        model = load_model_and_checkpoint(config, args.checkpoint_path, device)
        
        # Load tokenizer
        tokenizer = load_tokenizer(args.tokenizer_name, config)
        
        print(f"\n📝 Prompt: '{args.prompt}'")
        print("=" * 50)
        
        # Generate text
        full_text, new_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed
        )
        
        print("\n📄 Generated Text:")
        print("=" * 50)
        
        if args.show_new_only:
            print(new_text)
        else:
            print(full_text)
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"\n💾 Generated text saved to: {args.output_file}")
        
        print("\n✅ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
