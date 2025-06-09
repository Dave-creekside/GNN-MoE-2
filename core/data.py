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

from .config import MoEConfig

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

def load_data(config: MoEConfig):
    print(f"🚀 Setting up data loading for {config.dataset_name} / {config.dataset_config_name}...")
    try:
        from transformers import AutoTokenizer
        import datasets as hf_datasets
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        if config.vocab_size != tokenizer.vocab_size:
            config.vocab_size = tokenizer.vocab_size

        train_dataset_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="train")
        eval_dataset_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="validation")
        
        train_texts_all = [line.strip() for item in train_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
        eval_texts_all = [line.strip() for item in eval_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
        
        random.shuffle(train_texts_all)
        random.shuffle(eval_texts_all)

        num_train_samples = len(train_texts_all) if config.num_train_samples == -1 else min(len(train_texts_all), config.num_train_samples)
        num_eval_samples = len(eval_texts_all) if config.num_eval_samples == -1 else min(len(eval_texts_all), config.num_eval_samples)

        train_texts = train_texts_all[:num_train_samples]
        eval_texts = eval_texts_all[:num_eval_samples]

        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        
        data_mode = f"REAL_{config.dataset_config_name.upper().replace('-', '_')}"
        
    except Exception as e:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size:
            config.vocab_size = tokenizer.vocab_size

        num_train_synth = config.num_train_samples if config.num_train_samples != -1 else 2000
        num_eval_synth = config.num_eval_samples if config.num_eval_samples != -1 else 500
        total_synthetic_needed = num_train_synth + num_eval_synth
        
        base_synthetic_text = "The transformer architecture revolutionized natural language processing and related fields significantly. "
        synthetic_texts_list = [base_synthetic_text * (config.max_seq_length // len(base_synthetic_text) + 1)] * total_synthetic_needed
        
        train_texts = synthetic_texts_list[:num_train_synth]
        eval_texts = synthetic_texts_list[num_train_synth : num_train_synth + num_eval_synth]
        
        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        data_mode = "SYNTHETIC_FALLBACK"

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers_dataloader, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers_dataloader, pin_memory=True)
    
    return train_loader, eval_loader, tokenizer, data_mode
