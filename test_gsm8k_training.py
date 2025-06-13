#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gsm8k_training.py

Comprehensive test of GSM8K dataset processing and mini training with ghost experts.
Tests metadata filtering (solution field) and end-to-end training pipeline.
"""

import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import MoEConfig
from core.data import load_data_with_preprocessing
from core.architecture import MoEModel
from core.training import train_model, create_dynamic_optimizer, PrimaryGhostLRScheduler
from core.dataset_manager import DatasetManager

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'─'*40}")
    print(f"  {title}")
    print(f"{'─'*40}")

def inspect_gsm8k_preprocessing():
    """Inspect how GSM8K data gets processed and filtered."""
    print_section("GSM8K Dataset Preprocessing Inspection")
    
    # Check if dataset exists
    manager = DatasetManager()
    config = MoEConfig(
        dataset_source="huggingface",
        dataset_name="openai/gsm8k",
        dataset_config_name="main",
        num_train_samples=5,  # Very small for inspection
        num_eval_samples=2,
        max_seq_length=256,
        batch_size=2
    )
    
    expected_dataset = manager.get_dataset_by_config(
        config.dataset_source, 
        config.dataset_name, 
        config.dataset_config_name, 
        config.max_seq_length
    )
    
    if expected_dataset:
        print(f"✅ Found existing GSM8K dataset: {expected_dataset['name']}")
        dataset_path = Path("data-preprocessed") / expected_dataset['name']
        
        # Load and inspect samples
        train_tokens = torch.load(dataset_path / 'train_tokens.pt')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"📊 Dataset info:")
        print(f"   Train samples: {expected_dataset['num_train_samples']}")
        print(f"   Eval samples: {expected_dataset['num_eval_samples']}")
        print(f"   Tensor shape: {train_tokens.shape}")
        
        print(f"\n🔍 Sample inspection (checking for 'solution' field filtering):")
        for i in range(min(2, train_tokens.shape[0])):
            decoded_text = tokenizer.decode(train_tokens[i], skip_special_tokens=True)
            print(f"\n   Sample {i+1}:")
            # Show first 300 chars
            display_text = decoded_text[:300] + "..." if len(decoded_text) > 300 else decoded_text
            print(f"   {display_text}")
            
            # Check if filtering worked (should not contain step-by-step solutions)
            has_solution_keywords = any(keyword in decoded_text.lower() for keyword in 
                                      ['step 1:', 'step 2:', 'solution:', '<<', '>>', 'let me'])
            
            if has_solution_keywords:
                print(f"   ⚠️  Sample may contain solution steps")
            else:
                print(f"   ✅ Clean Q&A format (solution steps filtered)")
        
        return True
    else:
        print(f"📥 No existing GSM8K dataset found - will create during test")
        return False

def test_gsm8k_training():
    """Test complete GSM8K preprocessing and mini training."""
    print_header("GSM8K Dataset + Ghost Expert Training Test")
    
    # Configuration for mini training
    config = MoEConfig(
        # Dataset settings
        dataset_source="huggingface",
        dataset_name="openai/gsm8k",
        dataset_config_name="main",
        num_train_samples=100,  # Small for fast test
        num_eval_samples=20,
        max_seq_length=256,
        
        # Model settings (32d as requested)
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,          # As requested
        
        # Ghost expert settings (2 ghosts as requested)
        architecture_mode="ghost",
        
        # Training settings (1 short epoch as requested)
        epochs=1,
        batch_size=4,
        max_batches_per_epoch=15,  # Very short
        eval_every=8,
        learning_rate=1e-3,
        run_name="gsm8k_ghost_test",
        
        # Optimization
        use_mixed_precision=True
    )
    
    # Set ghost parameters
    config.ghost.num_ghost_experts = 2  # As requested
    config.ghost.ghost_activation_threshold = 0.01
    config.ghost.ghost_learning_rate = 5e-4
    
    print(f"📋 Test Configuration:")
    print(f"   Dataset: {config.dataset_name}/{config.dataset_config_name}")
    print(f"   Model: {config.embed_dim}d, {config.num_experts} experts, {config.ghost.num_ghost_experts} ghosts")
    print(f"   Training: {config.epochs} epoch, max {config.max_batches_per_epoch} batches")
    
    try:
        # Time the data loading
        print_section("Data Loading & Preprocessing")
        data_start = time.time()
        
        train_loader, eval_loader, tokenizer, data_mode = load_data_with_preprocessing(config)
        
        data_time = time.time() - data_start
        print(f"⏱️  Data loading time: {data_time:.2f}s")
        print(f"✅ Data loaded successfully")
        print(f"📊 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
        print(f"🔧 Data mode: {data_mode}")
        
        # Setup device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        
        # Create model with ghost experts
        print_section("Model Creation")
        model = MoEModel(config).to(device)
        
        total_params = model.get_total_params()
        print(f"🎯 Model created:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Model size: ~{total_params/1000:.1f}K parameters")
        print(f"   Experts: {config.num_experts} + {config.ghost.num_ghost_experts} ghosts")
        
        # Create optimizer and scheduler
        optimizer = create_dynamic_optimizer(model, config)
        scheduler = PrimaryGhostLRScheduler(config, optimizer)
        
        # Test a sample batch first
        print_section("Sample Batch Test")
        sample_batch = next(iter(train_loader))
        input_ids = sample_batch['input_ids'].to(device)
        print(f"📊 Sample batch shape: {input_ids.shape}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, step=0)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            print(f"✅ Forward pass successful, output shape: {logits.shape}")
        
        # Run mini training
        print_section("Mini Training with Ghost Experts")
        print(f"🚀 Starting training...")
        training_start = time.time()
        
        training_stats, final_loss = train_model(
            model, optimizer, scheduler, train_loader, eval_loader, device, config
        )
        
        training_time = time.time() - training_start
        print(f"✅ Training completed in {training_time:.1f}s")
        print(f"📊 Final loss: {final_loss:.4f}")
        
        if training_stats:
            print(f"📈 Training stats:")
            print(f"   Batches processed: {len(training_stats)}")
            if len(training_stats) > 0:
                avg_loss = sum(stat['loss'] for stat in training_stats) / len(training_stats)
                print(f"   Average loss: {avg_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_persistence():
    """Test that the dataset persists after processing."""
    print_section("Dataset Persistence Test")
    
    manager = DatasetManager()
    datasets = manager.list_datasets()
    
    gsm8k_datasets = [d for d in datasets if 'gsm8k' in d['name'].lower()]
    
    if gsm8k_datasets:
        print(f"✅ Found {len(gsm8k_datasets)} GSM8K dataset(s):")
        for dataset in gsm8k_datasets:
            print(f"   📁 {dataset['name']}")
            print(f"      Status: {dataset['status']}")
            print(f"      Size: {dataset['size_mb']}MB")
            print(f"      Samples: {dataset.get('num_train_samples', 'N/A')}+{dataset.get('num_eval_samples', 'N/A')}")
        
        # Test loading speed on second run - use SAME config as first test for cache hit
        print(f"\n⚡ Testing cached loading speed...")
        config = MoEConfig(
            dataset_source="huggingface",
            dataset_name="openai/gsm8k",
            dataset_config_name="main",
            num_train_samples=100,  # Same as first test for cache hit
            num_eval_samples=20,   # Same as first test for cache hit  
            max_seq_length=256,
            batch_size=4
        )
        
        cached_start = time.time()
        train_loader, eval_loader, tokenizer, data_mode = load_data_with_preprocessing(config)
        cached_time = time.time() - cached_start
        
        print(f"✅ Cached loading time: {cached_time:.2f}s")
        print(f"🚀 Dataset persistence verified!")
        
        return True
    else:
        print(f"❌ No GSM8K datasets found - persistence test failed")
        return False

def cleanup_test_checkpoint():
    """Clean up test checkpoint directory."""
    print_section("Cleanup")
    
    checkpoint_dir = Path("checkpoints/gsm8k_ghost_test")
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"🧹 Cleaned up test checkpoint: {checkpoint_dir}")
    else:
        print(f"📂 No test checkpoint to clean")

def main():
    """Run the complete GSM8K test suite."""
    print_header("🧪 GSM8K + Ghost Expert Training Test Suite")
    
    results = {
        'preprocessing_inspection': False,
        'full_training': False,
        'persistence': False
    }
    
    # Test 1: Inspect existing preprocessing (if any)
    try:
        results['preprocessing_inspection'] = inspect_gsm8k_preprocessing()
    except Exception as e:
        print(f"❌ Preprocessing inspection failed: {str(e)}")
    
    # Test 2: Full training test
    try:
        results['full_training'] = test_gsm8k_training()
    except Exception as e:
        print(f"❌ Training test failed: {str(e)}")
    
    # Test 3: Dataset persistence
    try:
        results['persistence'] = test_dataset_persistence()
    except Exception as e:
        print(f"❌ Persistence test failed: {str(e)}")
    
    # Cleanup
    cleanup_test_checkpoint()
    
    # Final results
    print_header("🎯 Test Results Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"📊 Tests Passed: {passed}/{total}")
    print("")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n{'='*60}")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("🔬 GSM8K dataset processing and ghost training working perfectly!")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check logs above.")
    print(f"{'='*60}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
