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
import os
from transformers import AutoTokenizer

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
    print(f"🚀 Setting up data loading...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size:
            config.vocab_size = tokenizer.vocab_size

        if config.dataset_source == "huggingface":
            print(f"   Loading from Hugging Face Hub: {config.dataset_name} / {config.dataset_config_name}")
            import datasets as hf_datasets
            train_dataset_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="train")
            eval_dataset_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="validation")
            train_texts_all = [line.strip() for item in train_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
            eval_texts_all = [line.strip() for item in eval_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
            data_mode = f"HF_{config.dataset_name}"
        
        elif config.dataset_source == "local_file":
            print(f"   Loading from local file: {config.dataset_name}")
            all_texts = []
            file_ext = os.path.splitext(config.dataset_name)[1]

            if file_ext == ".txt":
                with open(config.dataset_name, 'r', encoding='utf-8') as f:
                    all_texts = [line.strip() for line in f if len(line.strip()) > 30]
            
            elif file_ext in [".json", ".jsonl"]:
                import json
                with open(config.dataset_name, 'r', encoding='utf-8') as f:
                    if file_ext == ".jsonl":
                        json_data = [json.loads(line) for line in f]
                    else:
                        json_data = json.load(f)
                    
                    # Heuristic to find the text field
                    for item in json_data:
                        if "text" in item:
                            all_texts.append(item["text"])
                        elif "question" in item and "answer" in item:
                            # Handle the user's specific GRPO format
                            text = f"Question: {item.get('question', '')}\nReasoning: {item.get('reasoning', '')}\nAnswer: {item.get('answer', '')}"
                            all_texts.append(text)
                        else:
                            print(f"   ⚠️ Warning: Could not find a 'text' or 'question'/'answer' key in JSON object: {item}")

            else:
                raise ValueError(f"Unsupported local file format: {file_ext}")

            random.shuffle(all_texts)
            split_idx = int(len(all_texts) * 0.9)
            train_texts_all = all_texts[:split_idx]
            eval_texts_all = all_texts[split_idx:]
            data_mode = f"LOCAL_{os.path.basename(config.dataset_name)}"

        else:
            raise ValueError(f"Unknown dataset_source: {config.dataset_source}")

        random.shuffle(train_texts_all)
        random.shuffle(eval_texts_all)

        num_train_samples = len(train_texts_all) if config.num_train_samples == -1 else min(len(train_texts_all), config.num_train_samples)
        num_eval_samples = len(eval_texts_all) if config.num_eval_samples == -1 else min(len(eval_texts_all), config.num_eval_samples)

        train_texts = train_texts_all[:num_train_samples]
        eval_texts = eval_texts_all[:num_eval_samples]

        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        
    except Exception as e:
        print(f"   ⚠️ Error loading data: {e}. Falling back to synthetic data.")
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
