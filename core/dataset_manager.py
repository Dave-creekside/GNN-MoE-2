#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_manager.py

Management utilities for pretokenized datasets.
Handles listing, validation, and cleanup of stored datasets.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class DatasetManager:
    """Manages pretokenized datasets storage and cleanup."""
    
    def __init__(self, datasets_dir: str = "data-preprocessed"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available pretokenized datasets."""
        datasets = []
        
        if not self.datasets_dir.exists():
            return datasets
        
        for dataset_path in self.datasets_dir.iterdir():
            if dataset_path.is_dir():
                dataset_info = self._get_dataset_info(dataset_path)
                if dataset_info:
                    datasets.append(dataset_info)
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return datasets
    
    def _get_dataset_info(self, dataset_path: Path) -> Dict[str, Any]:
        """Get information about a specific dataset directory."""
        try:
            metadata_path = dataset_path / 'metadata.json'
            fingerprint_path = dataset_path / 'fingerprint.json'
            
            if not metadata_path.exists():
                return {
                    'name': dataset_path.name,
                    'path': str(dataset_path),
                    'status': 'invalid',
                    'error': 'Missing metadata.json',
                    'size_mb': self._get_directory_size_mb(dataset_path)
                }
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required files
            required_files = ['metadata.json', 'train_tokens.pt', 'eval_tokens.pt', 'fingerprint.json']
            missing_files = []
            for file in required_files:
                if not (dataset_path / file).exists():
                    missing_files.append(file)
            
            # Get creation time
            created_at = datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat()
            
            # Calculate total size
            size_mb = self._get_directory_size_mb(dataset_path)
            
            status = 'valid' if not missing_files else 'incomplete'
            
            info = {
                'name': dataset_path.name,
                'path': str(dataset_path),
                'status': status,
                'dataset_source': metadata.get('dataset_source', 'unknown'),
                'dataset_name': metadata.get('dataset_name', 'unknown'),
                'config_name': metadata.get('dataset_config_name'),
                'tokenizer': metadata.get('tokenizer', 'unknown'),
                'max_seq_length': metadata.get('max_seq_length', 0),
                'vocab_size': metadata.get('vocab_size', 0),
                'num_train_samples': metadata.get('num_train_samples', 0),
                'num_eval_samples': metadata.get('num_eval_samples', 0),
                'train_shape': metadata.get('train_shape', []),
                'eval_shape': metadata.get('eval_shape', []),
                'size_mb': size_mb,
                'created_at': created_at,
                'files': [f.name for f in dataset_path.iterdir() if f.is_file()]
            }
            
            if missing_files:
                info['missing_files'] = missing_files
            
            return info
            
        except Exception as e:
            return {
                'name': dataset_path.name,
                'path': str(dataset_path),
                'status': 'error',
                'error': str(e),
                'size_mb': self._get_directory_size_mb(dataset_path)
            }
    
    def _get_directory_size_mb(self, directory: Path) -> float:
        """Calculate total size of directory in MB."""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0
    
    def get_total_storage_mb(self) -> float:
        """Get total storage used by all datasets in MB."""
        if not self.datasets_dir.exists():
            return 0.0
        return self._get_directory_size_mb(self.datasets_dir)
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a specific dataset by name."""
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"   ❌ Dataset '{dataset_name}' not found")
            return False
        
        if not dataset_path.is_dir():
            print(f"   ❌ '{dataset_name}' is not a directory")
            return False
        
        try:
            shutil.rmtree(dataset_path)
            print(f"   ✅ Deleted dataset: {dataset_name}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to delete '{dataset_name}': {str(e)}")
            return False
    
    def clean_invalid_datasets(self) -> int:
        """Remove datasets with invalid or incomplete data."""
        datasets = self.list_datasets()
        cleaned_count = 0
        
        for dataset in datasets:
            if dataset['status'] in ['invalid', 'error', 'incomplete']:
                print(f"   🧹 Cleaning invalid dataset: {dataset['name']}")
                if self.delete_dataset(dataset['name']):
                    cleaned_count += 1
        
        return cleaned_count
    
    def clean_all_datasets(self) -> bool:
        """Delete all datasets."""
        if not self.datasets_dir.exists():
            print("   📂 No datasets directory found")
            return True
        
        try:
            # Remove all subdirectories
            for item in self.datasets_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
            
            print(f"   ✅ Cleaned all datasets from: {self.datasets_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to clean datasets: {str(e)}")
            return False
    
    def get_dataset_by_config(self, dataset_source: str, dataset_name: str, 
                             config_name: str = None, max_seq_length: int = 512) -> Dict[str, Any]:
        """Find dataset matching specific configuration."""
        if dataset_source == "huggingface":
            # Replace slashes with underscores to avoid nested directories
            safe_dataset_name = dataset_name.replace("/", "_")
            expected_name = f"{safe_dataset_name}_{config_name or 'default'}_gpt2_{max_seq_length}"
        else:
            stem = Path(dataset_name).stem
            expected_name = f"local_{stem}_gpt2_{max_seq_length}"
        
        datasets = self.list_datasets()
        for dataset in datasets:
            if dataset['name'] == expected_name:
                return dataset
        
        return {}
    
    def validate_dataset_integrity(self, dataset_name: str) -> Dict[str, Any]:
        """Perform detailed validation of a dataset."""
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            return {'valid': False, 'error': 'Dataset not found'}
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check required files
            required_files = ['metadata.json', 'train_tokens.pt', 'eval_tokens.pt', 'fingerprint.json']
            for file in required_files:
                file_path = dataset_path / file
                if not file_path.exists():
                    validation_result['errors'].append(f"Missing file: {file}")
                    validation_result['valid'] = False
            
            if not validation_result['valid']:
                return validation_result
            
            # Load and validate metadata
            with open(dataset_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Check tensor files
            import torch
            
            train_tokens = torch.load(dataset_path / 'train_tokens.pt', map_location='cpu')
            eval_tokens = torch.load(dataset_path / 'eval_tokens.pt', map_location='cpu')
            
            # Validate shapes match metadata
            if list(train_tokens.shape) != metadata.get('train_shape', []):
                validation_result['warnings'].append(
                    f"Train tensor shape mismatch: {list(train_tokens.shape)} vs {metadata.get('train_shape')}"
                )
            
            if list(eval_tokens.shape) != metadata.get('eval_shape', []):
                validation_result['warnings'].append(
                    f"Eval tensor shape mismatch: {list(eval_tokens.shape)} vs {metadata.get('eval_shape')}"
                )
            
            # Check sample counts
            if train_tokens.shape[0] != metadata.get('num_train_samples', 0):
                validation_result['warnings'].append(
                    f"Train sample count mismatch: {train_tokens.shape[0]} vs {metadata.get('num_train_samples')}"
                )
            
            if eval_tokens.shape[0] != metadata.get('num_eval_samples', 0):
                validation_result['warnings'].append(
                    f"Eval sample count mismatch: {eval_tokens.shape[0]} vs {metadata.get('num_eval_samples')}"
                )
            
            validation_result['metadata'] = metadata
            validation_result['actual_train_shape'] = list(train_tokens.shape)
            validation_result['actual_eval_shape'] = list(eval_tokens.shape)
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def print_datasets_summary(self):
        """Print a formatted summary of all datasets."""
        datasets = self.list_datasets()
        total_storage = self.get_total_storage_mb()
        
        print(f"\n📚 Dataset Storage Summary")
        print(f"{'='*60}")
        print(f"Total storage used: {total_storage:.1f} MB")
        print(f"Number of datasets: {len(datasets)}")
        
        if not datasets:
            print("   📭 No datasets found")
            return
        
        print(f"\n{'Name':<35} {'Status':<10} {'Size':<8} {'Samples':<12}")
        print(f"{'-'*70}")
        
        for dataset in datasets:
            name = dataset['name']
            if len(name) > 32:
                name = name[:29] + "..."
            
            status = dataset['status']
            size = f"{dataset['size_mb']:.1f}MB"
            
            if dataset['status'] == 'valid':
                samples = f"{dataset['num_train_samples']}+{dataset['num_eval_samples']}"
            else:
                samples = "N/A"
            
            print(f"{name:<35} {status:<10} {size:<8} {samples:<12}")
        
        # Show storage breakdown
        valid_count = sum(1 for d in datasets if d['status'] == 'valid')
        invalid_count = len(datasets) - valid_count
        
        if invalid_count > 0:
            print(f"\n⚠️  Found {invalid_count} invalid dataset(s)")
            print(f"   Run cleanup to remove them")
