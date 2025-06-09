#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_data.py

Data loading utilities for GNN-Coupled MoE models.
- SimpleTextDataset
- load_data function
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random

# Assuming GNNMoEConfig will be imported from gnn_moe_config.py in the main script
# from gnn_moe_config import GNNMoEConfig

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0), # Squeeze batch dim
            'attention_mask': encoding['attention_mask'].squeeze(0) # Squeeze batch dim
        }

def load_data(config): # config is a GNNMoEConfig instance
    print(f"🚀 Setting up data loading for {config.dataset_name} / {config.dataset_config_name}...")
    try:
        from transformers import AutoTokenizer
        import datasets as hf_datasets
        
        # For now, hardcode gpt2 tokenizer. Could be made configurable.
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Ensure config's vocab_size matches tokenizer, update if necessary
        if config.vocab_size != tokenizer.vocab_size:
            print(f"⚠️ Config vocab_size {config.vocab_size} != tokenizer vocab_size {tokenizer.vocab_size}. Updating config to use tokenizer's.")
            config.vocab_size = tokenizer.vocab_size

        print(f"📦 Attempting {config.dataset_name} ({config.dataset_config_name}) dataset loading...")
        # Removed trust_remote_code=True as it caused issues with older datasets lib
        train_dataset_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="train")
        eval_dataset_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="validation")
        
        # Process lines: split documents into lines, strip whitespace, filter by length
        train_texts_all = [line.strip() for item in train_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
        eval_texts_all = [line.strip() for item in eval_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
        
        print(f"Raw lines >30 chars: Train {len(train_texts_all)}, Eval {len(eval_texts_all)}")
        
        # Shuffle all available texts before sampling
        random.shuffle(train_texts_all)
        random.shuffle(eval_texts_all)

        # Determine number of samples to use (-1 means all)
        num_train_samples = len(train_texts_all) if config.num_train_samples == -1 else min(len(train_texts_all), config.num_train_samples)
        num_eval_samples = len(eval_texts_all) if config.num_eval_samples == -1 else min(len(eval_texts_all), config.num_eval_samples)

        if num_train_samples < 100 or num_eval_samples < 50: # Basic check for sufficient data
            raise ValueError(f"Dataset too small after filtering. Effective train: {num_train_samples}, eval: {num_eval_samples}")

        train_texts = train_texts_all[:num_train_samples]
        eval_texts = eval_texts_all[:num_eval_samples]

        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        
        print(f"✅ SUCCESS: Real {config.dataset_config_name} data loaded!")
        data_mode = f"REAL_{config.dataset_config_name.upper().replace('-', '_')}" # Sanitize name for mode
        
    except Exception as e:
        print(f"⚠️ Real data ({config.dataset_config_name}) loading failed: {e}")
        import traceback # Add this for detailed error logging
        print("--- Full Traceback for Data Loading Failure ---")
        traceback.print_exc() # Add this to print full traceback
        print("---------------------------------------------")
        print("🔄 Using synthetic fallback...")
        from transformers import AutoTokenizer # Ensure tokenizer is available for fallback
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size: # Ensure config matches
            print(f"⚠️ Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size} for synthetic data.")
            config.vocab_size = tokenizer.vocab_size

        # Determine number of synthetic samples based on config or defaults
        num_train_synth = config.num_train_samples if config.num_train_samples != -1 else 2000
        num_eval_synth = config.num_eval_samples if config.num_eval_samples != -1 else 500
        total_synthetic_needed = num_train_synth + num_eval_synth
        
        # Simple synthetic text
        base_synthetic_text = "The transformer architecture revolutionized natural language processing and related fields significantly. "
        # Ensure enough variety if total_synthetic_needed is large, or just repeat.
        # For simplicity, just repeat.
        synthetic_texts_list = [base_synthetic_text * (config.max_seq_length // len(base_synthetic_text) + 1)] * total_synthetic_needed
        
        train_texts = synthetic_texts_list[:num_train_synth]
        eval_texts = synthetic_texts_list[num_train_synth : num_train_synth + num_eval_synth]
        
        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        print("✅ Synthetic data ready!")
        data_mode = "SYNTHETIC_FALLBACK"

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers_dataloader, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers_dataloader, pin_memory=True)
    
    print(f"\n✅ DATA LOADING COMPLETE!")
    print(f"🎯 Mode: {data_mode}")
    print(f"📊 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print(f"📦 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
    print(f"🔤 Vocabulary: {tokenizer.vocab_size:,} tokens (using {tokenizer.name_or_path})")
    return train_loader, eval_loader, tokenizer, data_mode

if __name__ == '__main__':
    # Example of using load_data (requires GNNMoEConfig to be defined/imported)
    # from gnn_moe_config import GNNMoEConfig # Would be needed here
    
    # Dummy config for testing data loading
    @dataclass
    class DummyConfigForData:
        vocab_size: int = 50257; max_seq_length: int = 32; batch_size: int = 2
        dataset_name: str = "sst2" # sst2 is small and quick to download
        dataset_config_name: str = "default" # sst2 doesn't have specific configs like wikitext
        num_train_samples: int = 10; num_eval_samples: int = 5
        num_workers_dataloader: int = 0 # Simpler for direct test

    test_data_cfg = DummyConfigForData()
    print("\nTesting load_data with dummy config for SST2:", test_data_cfg)
    
    try:
        tl, el, tok, mode = load_data(test_data_cfg)
        print(f"load_data returned: mode={mode}, train_batches={len(tl)}, eval_batches={len(el)}")
        # Try to get a batch
        print("Sample train batch:", next(iter(tl)))
        print("Sample eval batch:", next(iter(el)))
    except Exception as e:
        print(f"Error during data loading test: {e}")
