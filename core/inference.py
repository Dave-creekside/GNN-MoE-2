#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py

Core text generation and inference logic for MoE models.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .architecture import MoEModel

def top_k_sampling(logits, k):
    """
    Filters logits to only the top k values, then renormalizes.
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def generate_text(
    model: MoEModel,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50
):
    """
    Generates text from a prompt using a trained MoE model.
    """
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated_ids = input_ids
    
    print("Generating text...")
    with torch.no_grad():
        for _ in tqdm(range(max_length)):
            # Use a dummy step value for the forward pass
            outputs = model(generated_ids, step=0, return_loss=False)
            next_token_logits = outputs['logits'][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                next_token_logits = top_k_sampling(next_token_logits, top_k)

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the generated token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
