#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretokenized_data.py

Fast data loading for pretokenized datasets.
"""

import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, Tuple

from .config import MoEConfig

class PretokenizedDataset(Dataset):
    """Dataset that loads pretokenized data directly."""
    
    def __init__(self, tokens_path: Path, max_length: int = None):
        """Initialize with path to tokenized data file."""
        self.tokens = torch.load(tokens_path, map_location='cpu')
        self.max_length = max_length
        
        # If max_length is specified and different from stored length, truncate/pad
        if max_length and self.tokens.size(1) != max_length:
            current_length = self.tokens.size(1)
            if current_length > max_length:
                # Truncate
                self.tokens = self.tokens[:, :max_length]
            else:
                # Pad
                padding = torch.zeros(self.tokens.size(0), max_length - current_length, dtype=self.tokens.dtype)
                self.tokens = torch.cat([self.tokens, padding], dim=1)
    
    def __len__(self):
        return self.tokens.size(0)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # Assuming PAD token is 50256 (GPT-2 eos token used as pad)
        attention_mask = (tokens != 50256).long()
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }

class PretokenizedDataLoader:
    """Manages loading of pretokenized datasets."""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.datasets_dir = Path("data-preprocessed")
        
    def get_dataset_path(self) -> Path:
        """Get path to the pretokenized dataset directory."""
        if self.config.dataset_source == "huggingface":
            # Replace slashes with underscores to avoid nested directories
            safe_dataset_name = self.config.dataset_name.replace("/", "_")
            dataset_id = f"{safe_dataset_name}_{self.config.dataset_config_name or 'default'}"
        else:
            dataset_id = f"local_{Path(self.config.dataset_name).stem}"
        
        dataset_id += f"_gpt2_{self.config.max_seq_length}"
        return self.datasets_dir / dataset_id
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        dataset_path = self.get_dataset_path()
        metadata_path = dataset_path / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def validate_dataset(self) -> bool:
        """Check if the pretokenized dataset exists and is valid."""
        dataset_path = self.get_dataset_path()
        
        required_files = ['metadata.json', 'train_tokens.pt', 'eval_tokens.pt', 'fingerprint.json']
        for file in required_files:
            if not (dataset_path / file).exists():
                return False
        
        return True
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, int]:
        """Create train and eval dataloaders from pretokenized data."""
        if not self.validate_dataset():
            raise FileNotFoundError(
                f"Pretokenized dataset not found. "
                f"Expected path: {self.get_dataset_path()}\n"
                f"Run preprocessing first to create the dataset."
            )
        
        dataset_path = self.get_dataset_path()
        metadata = self.load_metadata()
        
        print(f"📂 Loading pretokenized dataset from: {dataset_path}")
        print(f"   📊 Train samples: {metadata['num_train_samples']}")
        print(f"   📊 Eval samples: {metadata['num_eval_samples']}")
        print(f"   📏 Sequence length: {metadata['max_seq_length']}")
        print(f"   🔤 Vocab size: {metadata['vocab_size']}")
        
        # Create datasets
        train_dataset = PretokenizedDataset(
            dataset_path / 'train_tokens.pt',
            max_length=self.config.max_seq_length
        )
        
        eval_dataset = PretokenizedDataset(
            dataset_path / 'eval_tokens.pt', 
            max_length=self.config.max_seq_length
        )
        
        # Apply sample limits if specified
        if self.config.num_train_samples != -1:
            # Create subset if needed
            train_indices = list(range(min(len(train_dataset), self.config.num_train_samples)))
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        if self.config.num_eval_samples != -1:
            eval_indices = list(range(min(len(eval_dataset), self.config.num_eval_samples)))
            eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=True
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=True
        )
        
        actual_train_size = len(train_dataset)
        actual_eval_size = len(eval_dataset)
        
        print(f"   ✅ Created dataloaders: {actual_train_size} train, {actual_eval_size} eval samples")
        
        return train_loader, eval_loader, metadata['vocab_size']
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset."""
        if not self.validate_dataset():
            return {
                'exists': False,
                'path': str(self.get_dataset_path()),
                'error': 'Dataset not found'
            }
        
        try:
            metadata = self.load_metadata()
            dataset_path = self.get_dataset_path()
            
            # Calculate dataset size on disk
            total_size = 0
            for file_path in dataset_path.iterdir():
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            size_mb = total_size / (1024 * 1024)
            
            return {
                'exists': True,
                'path': str(dataset_path),
                'dataset_source': metadata['dataset_source'],
                'dataset_name': metadata['dataset_name'],
                'config_name': metadata.get('dataset_config_name'),
                'tokenizer': metadata['tokenizer'],
                'max_seq_length': metadata['max_seq_length'],
                'vocab_size': metadata['vocab_size'],
                'num_train_samples': metadata['num_train_samples'],
                'num_eval_samples': metadata['num_eval_samples'],
                'train_shape': metadata['train_shape'],
                'eval_shape': metadata['eval_shape'],
                'size_mb': round(size_mb, 2),
                'files': [f.name for f in dataset_path.iterdir() if f.is_file()]
            }
            
        except Exception as e:
            return {
                'exists': True,
                'path': str(self.get_dataset_path()),
                'error': f'Failed to load metadata: {str(e)}'
            }

def load_pretokenized_data(config: MoEConfig) -> Tuple[DataLoader, DataLoader, int, str]:
    """
    Main function to load pretokenized data.
    Returns (train_loader, eval_loader, vocab_size, data_mode)
    """
    loader = PretokenizedDataLoader(config)
    train_loader, eval_loader, vocab_size = loader.create_dataloaders()
    
    # Create data mode string for compatibility
    if config.dataset_source == "huggingface":
        data_mode = f"PRETOKENIZED_HF_{config.dataset_name}"
    else:
        data_mode = f"PRETOKENIZED_LOCAL_{Path(config.dataset_name).stem}"
    
    return train_loader, eval_loader, vocab_size, data_mode
