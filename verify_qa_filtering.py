#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_qa_filtering.py

Quick verification that QA dataset filtering works correctly.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer

def verify_qa_filtering():
    """Verify that the QA dataset correctly filters out metadata."""
    print("🔍 Verifying QA Dataset Filtering")
    print("=" * 50)
    
    # Load the pretokenized data
    dataset_path = Path("data-preprocessed/local_test_qa_dataset_gpt2_128")
    
    if not dataset_path.exists():
        print("❌ Pretokenized dataset not found. Run quick_test_datasets.py first.")
        return
    
    # Load metadata
    import json
    with open(dataset_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Dataset Metadata:")
    print(f"   Source: {metadata['dataset_source']}")
    print(f"   Train samples: {metadata['num_train_samples']}")
    print(f"   Eval samples: {metadata['num_eval_samples']}")
    
    # Load tokenized data
    train_tokens = torch.load(dataset_path / 'train_tokens.pt')
    
    # Decode to see the actual text
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\n🔍 Decoded Samples (checking for metadata filtering):")
    for i in range(train_tokens.shape[0]):
        decoded_text = tokenizer.decode(train_tokens[i], skip_special_tokens=True)
        print(f"\n   Sample {i+1}:")
        print(f"   {decoded_text}")
        
        # Check if filtering worked
        has_reasoning = "reasoning" in decoded_text.lower()
        has_metadata = any(field in decoded_text.lower() for field in ['metadata', 'extra_field', 'source', 'year'])
        
        if has_reasoning or has_metadata:
            print(f"   ❌ FILTERING FAILED - Contains metadata!")
        else:
            print(f"   ✅ Clean QA format - metadata successfully filtered")
    
    print(f"\n{'='*50}")
    print("✅ QA filtering verification complete!")

if __name__ == "__main__":
    verify_qa_filtering()
