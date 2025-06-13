#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessor.py

Dataset preprocessing pipeline for creating pretokenized datasets.
Handles download, tokenization, and saving of datasets for fast training.
"""

import torch
import json
import os
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer

from .config import MoEConfig

class DatasetPreprocessor:
    """Handles preprocessing of raw datasets into tokenized format."""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.datasets_dir = Path("data-preprocessed")
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def get_dataset_fingerprint(self) -> str:
        """Generate unique fingerprint for dataset + config combination."""
        fingerprint_data = {
            'dataset_source': self.config.dataset_source,
            'dataset_name': self.config.dataset_name,
            'dataset_config_name': self.config.dataset_config_name,
            'max_seq_length': self.config.max_seq_length,
            'tokenizer': 'gpt2',
            'num_train_samples': self.config.num_train_samples,
            'num_eval_samples': self.config.num_eval_samples
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
    
    def get_dataset_path(self) -> Path:
        """Get the path where the dataset should be saved."""
        if self.config.dataset_source == "huggingface":
            # Replace slashes with underscores to avoid nested directories
            safe_dataset_name = self.config.dataset_name.replace("/", "_")
            dataset_id = f"{safe_dataset_name}_{self.config.dataset_config_name or 'default'}"
        else:
            dataset_id = f"local_{Path(self.config.dataset_name).stem}"
        
        dataset_id += f"_gpt2_{self.config.max_seq_length}"
        return self.datasets_dir / dataset_id
    
    def dataset_exists(self) -> bool:
        """Check if preprocessed dataset already exists and is valid."""
        dataset_path = self.get_dataset_path()
        if not dataset_path.exists():
            return False
        
        # Check required files exist
        required_files = ['metadata.json', 'train_tokens.pt', 'eval_tokens.pt', 'fingerprint.json']
        for file in required_files:
            if not (dataset_path / file).exists():
                return False
        
        # Validate fingerprint
        try:
            with open(dataset_path / 'fingerprint.json', 'r') as f:
                stored_fingerprint = json.load(f)['fingerprint']
            current_fingerprint = self.get_dataset_fingerprint()
            return stored_fingerprint == current_fingerprint
        except:
            return False
    
    def validate_hf_dataset(self, dataset_name: str, config_name: str = None) -> Tuple[bool, str, Dict]:
        """Validate Hugging Face dataset compatibility."""
        try:
            import datasets as hf_datasets
            
            # SMART FIX: Detect common mistake where dataset/config are swapped
            if config_name and '/' in config_name and not dataset_name:
                # Likely mistake: config_name contains the actual dataset name
                print(f"   🔄 Detected potential config/dataset swap. Trying '{config_name}' as dataset name...")
                return self.validate_hf_dataset(config_name, "default")
            
            if config_name and '/' in config_name and dataset_name and not '/' in dataset_name:
                # Another common mistake: config has dataset path, dataset has simple name
                print(f"   🔄 Detected potential config/dataset swap. Swapping '{dataset_name}' <-> '{config_name}'...")
                return self.validate_hf_dataset(config_name, dataset_name)
            
            # Check if dataset exists
            try:
                dataset_info = hf_datasets.get_dataset_infos(dataset_name)
            except Exception as e:
                # If dataset not found and config looks like a dataset path, suggest fix
                if config_name and '/' in config_name:
                    return False, f"Dataset '{dataset_name}' not found. Did you mean to use '{config_name}' as the dataset name instead?", {}
                return False, f"Dataset '{dataset_name}' not found on Hugging Face Hub.", {}
            
            # Validate config
            if config_name:
                if config_name not in dataset_info:
                    available_configs = list(dataset_info.keys())
                    # Special error message if config looks like a dataset path
                    if '/' in config_name:
                        return False, f"Config '{config_name}' looks like a dataset name. Please use '{config_name}' as Dataset Name and '{available_configs[0] if available_configs else 'default'}' as Config Name.", {}
                    return False, f"Config '{config_name}' not found. Available: {available_configs}", {}
            else:
                config_name = list(dataset_info.keys())[0] if dataset_info else None
            
            # Try loading sample
            try:
                sample_dataset = hf_datasets.load_dataset(dataset_name, name=config_name, split="train[:10]")
            except Exception as e:
                return False, f"Failed to load sample: {str(e)}", {}
            
            # Check structure
            if len(sample_dataset) == 0:
                return False, "Dataset appears to be empty", {}
            
            sample_item = sample_dataset[0]
            has_text_field = "text" in sample_item
            has_qa_fields = all(field in sample_item for field in ["question", "answer"])
            
            if not (has_text_field or has_qa_fields):
                available_fields = list(sample_item.keys())
                return False, f"Must have 'text' field or 'question'/'answer' fields. Available: {available_fields}", {}
            
            # Get available splits
            try:
                dataset_splits = hf_datasets.load_dataset(dataset_name, name=config_name)
                available_splits = list(dataset_splits.keys())
            except:
                available_splits = ["train"]
            
            has_train = "train" in available_splits
            has_eval = any(split in available_splits for split in ["validation", "test", "dev"])
            
            if not has_train:
                return False, f"Must have 'train' split. Available: {available_splits}", {}
            
            return True, "", {
                "config_name": config_name,
                "available_splits": available_splits,
                "has_eval_split": has_eval,
                "text_format": "text_field" if has_text_field else "qa_fields",
                "sample_fields": list(sample_item.keys())
            }
            
        except ImportError:
            return False, "datasets library not installed. Install with: pip install datasets", {}
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", {}
    
    def process_hf_dataset(self, dataset_info: Dict) -> Tuple[List[str], List[str]]:
        """Process Hugging Face dataset into text samples."""
        import datasets as hf_datasets
        
        config_name = dataset_info["config_name"]
        print(f"   📥 Loading from Hugging Face: {self.config.dataset_name}" + 
              (f" / {config_name}" if config_name else ""))
        
        # Load datasets
        train_dataset_raw = hf_datasets.load_dataset(
            self.config.dataset_name, name=config_name, split="train"
        )
        
        # Handle evaluation split
        if dataset_info["has_eval_split"]:
            eval_split_name = next(
                split for split in ["validation", "test", "dev"] 
                if split in dataset_info["available_splits"]
            )
            eval_dataset_raw = hf_datasets.load_dataset(
                self.config.dataset_name, name=config_name, split=eval_split_name
            )
        else:
            print(f"   📋 No validation split found. Using 10% of training data for evaluation.")
            dataset_split = hf_datasets.load_dataset(
                self.config.dataset_name, name=config_name, split="train"
            ).train_test_split(test_size=0.1)
            train_dataset_raw = dataset_split["train"]
            eval_dataset_raw = dataset_split["test"]
        
        # Process based on format
        if dataset_info["text_format"] == "text_field":
            train_texts = self._process_text_format(train_dataset_raw)
            eval_texts = self._process_text_format(eval_dataset_raw)
        else:  # qa_fields
            train_texts = self._process_qa_format(train_dataset_raw)
            eval_texts = self._process_qa_format(eval_dataset_raw)
        
        return train_texts, eval_texts
    
    def _process_text_format(self, dataset) -> List[str]:
        """Process text field format (like WikiText)."""
        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 30:
                # Handle multi-line text
                if '\n' in text and len(text.splitlines()) > 1:
                    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
                    texts.extend(lines)
                else:
                    texts.append(text)
        return texts
    
    def _process_qa_format(self, dataset) -> List[str]:
        """Process question/answer format - extract ONLY question and answer fields."""
        texts = []
        for item in dataset:
            # Extract only question and answer, ignore all metadata fields
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            # Only process if both question and answer exist and are substantial
            if question and answer and len(question) > 5 and len(answer) > 5:
                combined_text = f"Question: {question}\nAnswer: {answer}"
                if len(combined_text) > 30:
                    texts.append(combined_text)
        return texts
    
    def process_local_file(self) -> Tuple[List[str], List[str]]:
        """Process local file into text samples."""
        print(f"   📁 Loading from local file: {self.config.dataset_name}")
        
        if not os.path.exists(self.config.dataset_name):
            raise FileNotFoundError(f"Local file not found: {self.config.dataset_name}")
        
        all_texts = []
        file_ext = os.path.splitext(self.config.dataset_name)[1]
        
        if file_ext == ".txt":
            with open(self.config.dataset_name, 'r', encoding='utf-8') as f:
                all_texts = [line.strip() for line in f if len(line.strip()) > 30]
        
        elif file_ext in [".json", ".jsonl"]:
            import json
            try:
                with open(self.config.dataset_name, 'r', encoding='utf-8') as f:
                    if file_ext == ".jsonl":
                        json_data = [json.loads(line) for line in f]
                    else:
                        json_data = json.load(f)
                
                for item in json_data:
                    if "text" in item:
                        text = item["text"].strip()
                        if len(text) > 30:
                            all_texts.append(text)
                    elif "question" in item and "answer" in item:
                        # Clean QA format - extract ONLY question and answer fields
                        question = item.get('question', '').strip()
                        answer = item.get('answer', '').strip()
                        
                        # Only process if both exist and are substantial
                        if question and answer and len(question) > 5 and len(answer) > 5:
                            text = f"Question: {question}\nAnswer: {answer}"
                            if len(text) > 30:
                                all_texts.append(text)
                    else:
                        # Show available fields for debugging (but don't show full item data)
                        available_fields = list(item.keys()) if isinstance(item, dict) else "non-dict"
                        print(f"   ⚠️  Skipping item - requires 'text' or 'question'/'answer' fields. Found: {available_fields}")
                        
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {self.config.dataset_name}: {str(e)}")
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: .txt, .json, .jsonl")
        
        if not all_texts:
            raise ValueError(f"No valid text data found in: {self.config.dataset_name}")
        
        # Split into train/eval
        random.shuffle(all_texts)
        split_idx = int(len(all_texts) * 0.9)
        train_texts = all_texts[:split_idx]
        eval_texts = all_texts[split_idx:]
        
        return train_texts, eval_texts
    
    def tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """Tokenize list of texts into tensor format."""
        print(f"   🔤 Tokenizing {len(texts)} samples...")
        
        all_tokens = []
        for text in texts:
            encoding = self.tokenizer(
                text, 
                max_length=self.config.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            all_tokens.append(encoding['input_ids'].squeeze(0))
        
        return torch.stack(all_tokens)
    
    def save_dataset(self, train_tokens: torch.Tensor, eval_tokens: torch.Tensor, 
                    train_texts: List[str], eval_texts: List[str]) -> Path:
        """Save preprocessed dataset to disk."""
        dataset_path = self.get_dataset_path()
        dataset_path.mkdir(exist_ok=True)
        
        print(f"   💾 Saving to: {dataset_path}")
        
        # Save tokenized data
        torch.save(train_tokens, dataset_path / 'train_tokens.pt')
        torch.save(eval_tokens, dataset_path / 'eval_tokens.pt')
        
        # Save metadata
        metadata = {
            'dataset_source': self.config.dataset_source,
            'dataset_name': self.config.dataset_name,
            'dataset_config_name': self.config.dataset_config_name,
            'tokenizer': 'gpt2',
            'max_seq_length': self.config.max_seq_length,
            'vocab_size': self.tokenizer.vocab_size,
            'num_train_samples': len(train_texts),
            'num_eval_samples': len(eval_texts),
            'train_shape': list(train_tokens.shape),
            'eval_shape': list(eval_tokens.shape)
        }
        
        with open(dataset_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save fingerprint
        fingerprint_data = {'fingerprint': self.get_dataset_fingerprint()}
        with open(dataset_path / 'fingerprint.json', 'w') as f:
            json.dump(fingerprint_data, f, indent=2)
        
        print(f"   ✅ Saved {len(train_texts)} train + {len(eval_texts)} eval samples")
        return dataset_path
    
    def preprocess_dataset(self) -> Path:
        """Main preprocessing pipeline."""
        print(f"🔄 Preprocessing dataset...")
        
        # Check if already exists
        if self.dataset_exists() and not getattr(self.config, 'force_reprocess', False):
            dataset_path = self.get_dataset_path()
            print(f"   ✅ Found existing dataset: {dataset_path}")
            return dataset_path
        
        # Process based on source
        if self.config.dataset_source == "huggingface":
            # Validate dataset
            is_valid, error_msg, dataset_info = self.validate_hf_dataset(
                self.config.dataset_name, self.config.dataset_config_name
            )
            
            if not is_valid:
                print(f"   ❌ Dataset validation failed: {error_msg}")
                raise ValueError(f"Dataset validation failed: {error_msg}")
            
            print(f"   ✅ Dataset validation passed")
            train_texts, eval_texts = self.process_hf_dataset(dataset_info)
            
        elif self.config.dataset_source == "local_file":
            train_texts, eval_texts = self.process_local_file()
        
        else:
            raise ValueError(f"Unknown dataset_source: {self.config.dataset_source}")
        
        # Limit samples if specified
        if self.config.num_train_samples != -1:
            train_texts = train_texts[:self.config.num_train_samples]
        if self.config.num_eval_samples != -1:
            eval_texts = eval_texts[:self.config.num_eval_samples]
        
        # Validate we have data
        if not train_texts:
            raise ValueError("No training samples after processing")
        if not eval_texts:
            raise ValueError("No evaluation samples after processing")
        
        # Tokenize
        train_tokens = self.tokenize_texts(train_texts)
        eval_tokens = self.tokenize_texts(eval_texts)
        
        # Save
        dataset_path = self.save_dataset(train_tokens, eval_tokens, train_texts, eval_texts)
        
        return dataset_path
