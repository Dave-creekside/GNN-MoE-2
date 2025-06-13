#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_test_datasets.py

Quick verification test for pretokenized dataset system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import MoEConfig
from core.data import load_data_with_preprocessing

def quick_test():
    """Quick test of the pretokenized dataset system."""
    print("🧪 Quick Dataset Test")
    print("=" * 40)
    
    # Test with local QA dataset (demonstrates metadata filtering)
    config = MoEConfig(
        dataset_source="local_file",
        dataset_name="data/test_qa_dataset.json",
        dataset_config_name=None,
        num_train_samples=4,  # All samples
        num_eval_samples=1,
        max_seq_length=128,
        batch_size=2
    )
    
    try:
        print(f"📥 Testing dataset loading...")
        train_loader, eval_loader, tokenizer, data_mode = load_data_with_preprocessing(config)
        
        print(f"✅ Success!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Eval batches: {len(eval_loader)}")
        print(f"   Data mode: {data_mode}")
        
        # Test a batch
        for batch in train_loader:
            print(f"   Batch shape: {batch['input_ids'].shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print("=" * 40)
    print("✅ Quick test passed!" if success else "❌ Quick test failed!")
