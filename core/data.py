#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py

Data loading utilities for GNN-Coupled MoE models.
Supports both legacy on-the-fly tokenization and new pretokenized pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
from transformers import AutoTokenizer

from .config import MoEConfig
from .preprocessor import DatasetPreprocessor
from .pretokenized_data import load_pretokenized_data

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

def validate_hf_dataset(dataset_name, config_name=None):
    """
    Validate that a Hugging Face dataset is compatible with our framework.
    Returns (is_valid, error_message, dataset_info)
    """
    try:
        import datasets as hf_datasets
        
        # First, check if the dataset exists and can be loaded
        try:
            dataset_info = hf_datasets.get_dataset_infos(dataset_name)
        except Exception as e:
            return False, f"Dataset '{dataset_name}' not found on Hugging Face Hub. Error: {str(e)}", None
        
        # For datasets with configs, validate the config exists
        if config_name:
            if config_name not in dataset_info:
                available_configs = list(dataset_info.keys())
                return False, f"Config '{config_name}' not found in dataset '{dataset_name}'. Available configs: {available_configs}", None
        else:
            # Use default config if available
            config_name = list(dataset_info.keys())[0] if dataset_info else None
        
        # Try to load a small sample to check structure
        try:
            sample_dataset = hf_datasets.load_dataset(dataset_name, name=config_name, split="train[:10]")
        except Exception as e:
            return False, f"Failed to load dataset sample. Error: {str(e)}", None
        
        # Check available splits
        try:
            dataset_splits = hf_datasets.load_dataset(dataset_name, name=config_name)
            available_splits = list(dataset_splits.keys())
        except:
            available_splits = ["train"]  # Fallback assumption
        
        # Validate required splits
        has_train = "train" in available_splits
        has_eval = any(split in available_splits for split in ["validation", "test", "dev"])
        
        if not has_train:
            return False, f"Dataset must have a 'train' split. Available splits: {available_splits}", None
        
        # Check data structure
        if len(sample_dataset) == 0:
            return False, "Dataset appears to be empty", None
        
        sample_item = sample_dataset[0]
        
        # Check for compatible text fields
        has_text_field = "text" in sample_item
        has_qa_fields = all(field in sample_item for field in ["question", "answer"])
        
        if not (has_text_field or has_qa_fields):
            available_fields = list(sample_item.keys())
            return False, f"Dataset must have either a 'text' field or 'question'/'answer' fields. Available fields: {available_fields}", None
        
        return True, "", {
            "config_name": config_name,
            "available_splits": available_splits,
            "has_eval_split": has_eval,
            "text_format": "text_field" if has_text_field else "qa_fields",
            "sample_fields": list(sample_item.keys())
        }
        
    except ImportError:
        return False, "datasets library not installed. Install with: pip install datasets", None
    except Exception as e:
        return False, f"Unexpected error during validation: {str(e)}", None

def load_hf_dataset(config: MoEConfig, dataset_info):
    """Load and process a validated Hugging Face dataset."""
    import datasets as hf_datasets
    
    config_name = dataset_info["config_name"]
    print(f"   Loading from Hugging Face Hub: {config.dataset_name}" + (f" / {config_name}" if config_name else ""))
    
    # Load full datasets
    train_dataset_raw = hf_datasets.load_dataset(config.dataset_name, name=config_name, split="train")
    
    # Handle evaluation split
    if dataset_info["has_eval_split"]:
        eval_split_name = next(split for split in ["validation", "test", "dev"] if split in dataset_info["available_splits"])
        eval_dataset_raw = hf_datasets.load_dataset(config.dataset_name, name=config_name, split=eval_split_name)
    else:
        # Split train dataset if no eval split exists
        print(f"   No validation split found. Using 10% of training data for evaluation.")
        dataset_split = hf_datasets.load_dataset(config.dataset_name, name=config_name, split="train").train_test_split(test_size=0.1)
        train_dataset_raw = dataset_split["train"]
        eval_dataset_raw = dataset_split["test"]
    
    # Extract text based on format
    if dataset_info["text_format"] == "text_field":
        # Process text field (handle both single-line and multi-line text)
        train_texts_all = []
        for item in train_dataset_raw:
            text = item['text'].strip()
            if len(text) > 30:
                # Check if this looks like multi-line text (like wikitext)
                if '\n' in text and len(text.splitlines()) > 1:
                    # Process line by line for multi-line datasets
                    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
                    train_texts_all.extend(lines)
                else:
                    # Use as single sample
                    train_texts_all.append(text)
        
        eval_texts_all = []
        for item in eval_dataset_raw:
            text = item['text'].strip()
            if len(text) > 30:
                if '\n' in text and len(text.splitlines()) > 1:
                    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
                    eval_texts_all.extend(lines)
                else:
                    eval_texts_all.append(text)
    
    else:  # qa_fields format
        # Handle question/answer format (like lambda calculus dataset)
        train_texts_all = []
        for item in train_dataset_raw:
            text_parts = []
            if "question" in item:
                text_parts.append(f"Question: {item['question']}")
            if "reasoning" in item:
                text_parts.append(f"Reasoning: {item['reasoning']}")
            if "answer" in item:
                text_parts.append(f"Answer: {item['answer']}")
            
            if text_parts:
                combined_text = "\n".join(text_parts)
                if len(combined_text) > 30:
                    train_texts_all.append(combined_text)
        
        eval_texts_all = []
        for item in eval_dataset_raw:
            text_parts = []
            if "question" in item:
                text_parts.append(f"Question: {item['question']}")
            if "reasoning" in item:
                text_parts.append(f"Reasoning: {item['reasoning']}")
            if "answer" in item:
                text_parts.append(f"Answer: {item['answer']}")
            
            if text_parts:
                combined_text = "\n".join(text_parts)
                if len(combined_text) > 30:
                    eval_texts_all.append(combined_text)
    
    return train_texts_all, eval_texts_all

def load_data(config: MoEConfig):
    print(f"🚀 Setting up data loading...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    if config.vocab_size != tokenizer.vocab_size:
        config.vocab_size = tokenizer.vocab_size

    if config.dataset_source == "huggingface":
        # Validate dataset compatibility first
        is_valid, error_msg, dataset_info = validate_hf_dataset(config.dataset_name, config.dataset_config_name)
        
        if not is_valid:
            print(f"   ❌ Dataset validation failed: {error_msg}")
            print(f"   📋 Compatible dataset requirements:")
            print(f"      - Must have a 'train' split")
            print(f"      - Must have either a 'text' field OR 'question'/'answer' fields")
            print(f"      - Recommended datasets: wikitext, openwebtext, bookcorpus, c4")
            raise ValueError(f"Dataset validation failed: {error_msg}")
        
        print(f"   ✅ Dataset validation passed")
        train_texts_all, eval_texts_all = load_hf_dataset(config, dataset_info)
        data_mode = f"HF_{config.dataset_name}"
    
    elif config.dataset_source == "local_file":
        print(f"   Loading from local file: {config.dataset_name}")
        
        if not os.path.exists(config.dataset_name):
            raise FileNotFoundError(f"Local file not found: {config.dataset_name}")
        
        all_texts = []
        file_ext = os.path.splitext(config.dataset_name)[1]

        if file_ext == ".txt":
            with open(config.dataset_name, 'r', encoding='utf-8') as f:
                all_texts = [line.strip() for line in f if len(line.strip()) > 30]
        
        elif file_ext in [".json", ".jsonl"]:
            import json
            try:
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
                        
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in file {config.dataset_name}: {str(e)}")

        else:
            raise ValueError(f"Unsupported local file format: {file_ext}. Supported formats: .txt, .json, .jsonl")

        if not all_texts:
            raise ValueError(f"No valid text data found in file: {config.dataset_name}")

        random.shuffle(all_texts)
        split_idx = int(len(all_texts) * 0.9)
        train_texts_all = all_texts[:split_idx]
        eval_texts_all = all_texts[split_idx:]
        data_mode = f"LOCAL_{os.path.basename(config.dataset_name)}"

    else:
        raise ValueError(f"Unknown dataset_source: '{config.dataset_source}'. Must be 'huggingface' or 'local_file'")

    # Validate we have data
    if not train_texts_all:
        raise ValueError("No training data found after processing")
    if not eval_texts_all:
        raise ValueError("No evaluation data found after processing")

    random.shuffle(train_texts_all)
    random.shuffle(eval_texts_all)

    num_train_samples = len(train_texts_all) if config.num_train_samples == -1 else min(len(train_texts_all), config.num_train_samples)
    num_eval_samples = len(eval_texts_all) if config.num_eval_samples == -1 else min(len(eval_texts_all), config.num_eval_samples)

    train_texts = train_texts_all[:num_train_samples]
    eval_texts = eval_texts_all[:num_eval_samples]

    # Final validation
    if not train_texts:
        raise ValueError("No training samples available after filtering")
    if not eval_texts:
        raise ValueError("No evaluation samples available after filtering")

    print(f"   ✅ Successfully loaded {len(train_texts)} training samples and {len(eval_texts)} evaluation samples")

    train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
    eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers_dataloader, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers_dataloader, pin_memory=True)
    
    return train_loader, eval_loader, tokenizer, data_mode

def load_data_with_preprocessing(config: MoEConfig):
    """
    Main data loading function with automatic preprocessing.
    Always uses pretokenized data for maximum performance.
    """
    print(f"🚀 Setting up data loading with preprocessing...")
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(config)
    
    # Check if pretokenized dataset exists
    if not preprocessor.dataset_exists():
        print(f"   🔄 Pretokenized dataset not found. Starting preprocessing...")
        try:
            dataset_path = preprocessor.preprocess_dataset()
            print(f"   ✅ Preprocessing complete: {dataset_path}")
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {str(e)}")
            raise
    else:
        print(f"   ✅ Using existing pretokenized dataset: {preprocessor.get_dataset_path()}")
    
    # Load pretokenized data
    try:
        train_loader, eval_loader, vocab_size, data_mode = load_pretokenized_data(config)
        
        # Update config vocab size if needed
        if config.vocab_size != vocab_size:
            config.vocab_size = vocab_size
        
        # Create a dummy tokenizer for compatibility (not used in training)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        return train_loader, eval_loader, tokenizer, data_mode
        
    except Exception as e:
        print(f"   ❌ Failed to load pretokenized data: {str(e)}")
        print(f"   🔄 Falling back to legacy data loading...")
        # Fallback to legacy loading if pretokenized fails
        return load_data(config)

# For backward compatibility, make the new function the default
def load_data_fast(config: MoEConfig):
    """Fast data loading with automatic preprocessing (recommended)."""
    return load_data_with_preprocessing(config)
