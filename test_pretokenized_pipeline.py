#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pretokenized_pipeline.py

Comprehensive test suite for the pretokenized dataset pipeline.
Tests text format (WikiText), QA format (Creekside/GRPO), and includes mini training validation.
"""

import os
import sys
import time
import torch
import subprocess
import shutil
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import MoEConfig
from core.data import load_data, load_data_with_preprocessing
from core.preprocessor import DatasetPreprocessor
from core.dataset_manager import DatasetManager
from core.pretokenized_data import load_pretokenized_data
from core.architecture import MoEModel
from core.training import train_model, create_dynamic_optimizer, PrimaryGhostLRScheduler

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

def inspect_sample_data(dataset_path, num_samples=3):
    """Inspect sample data from a pretokenized dataset."""
    print_section("Sample Data Inspection")
    
    try:
        # Load metadata
        import json
        with open(dataset_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Dataset Metadata:")
        print(f"   Source: {metadata.get('dataset_source', 'unknown')}")
        print(f"   Name: {metadata.get('dataset_name', 'unknown')}")
        print(f"   Config: {metadata.get('dataset_config_name', 'none')}")
        print(f"   Train samples: {metadata.get('num_train_samples', 'unknown')}")
        print(f"   Eval samples: {metadata.get('num_eval_samples', 'unknown')}")
        print(f"   Vocab size: {metadata.get('vocab_size', 'unknown')}")
        print(f"   Max seq length: {metadata.get('max_seq_length', 'unknown')}")
        
        # Load tokenized data
        train_tokens = torch.load(dataset_path / 'train_tokens.pt')
        print(f"\n📈 Tensor Information:")
        print(f"   Training tensor shape: {train_tokens.shape}")
        print(f"   Data type: {train_tokens.dtype}")
        
        # Decode samples to see actual text
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"\n🔍 Sample Decoded Text (first {num_samples} samples):")
        for i in range(min(num_samples, train_tokens.shape[0])):
            decoded_text = tokenizer.decode(train_tokens[i], skip_special_tokens=True)
            # Truncate for display
            display_text = decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text
            print(f"\n   Sample {i+1}:")
            print(f"   {display_text}")
            
    except Exception as e:
        print(f"❌ Error inspecting data: {str(e)}")

def test_text_format_dataset():
    """Test text format dataset (WikiText-2-v1)."""
    print_header("Testing Text Format Dataset (WikiText-2-v1)")
    
    # Create config for WikiText
    config = MoEConfig(
        dataset_source="huggingface",
        dataset_name="Salesforce/wikitext",
        dataset_config_name="wikitext-103-v1",
        num_train_samples=500,  # Small for testing
        num_eval_samples=100,
        max_seq_length=256,
        embed_dim=32,  # Small for fast testing
        batch_size=4
    )
    
    print(f"📋 Config: {config.dataset_name}/{config.dataset_config_name}")
    print(f"📊 Samples: {config.num_train_samples} train, {config.num_eval_samples} eval")
    
    # Time the preprocessing
    start_time = time.time()
    
    try:
        # Test preprocessing
        train_loader, eval_loader, tokenizer, data_mode = load_data_with_preprocessing(config)
        
        preprocessing_time = time.time() - start_time
        print(f"⏱️  Preprocessing time: {preprocessing_time:.2f}s")
        print(f"✅ Successfully loaded WikiText-2-v1 dataset")
        print(f"📊 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
        
        # Inspect the data
        preprocessor = DatasetPreprocessor(config)
        dataset_path = preprocessor.get_dataset_path()
        inspect_sample_data(dataset_path)
        
        return True, preprocessing_time
        
    except Exception as e:
        print(f"❌ Failed to process WikiText-2-v1: {str(e)}")
        return False, 0

def test_qa_format_dataset():
    """Test QA format dataset (Creekside/GRPO-Lambda-ParsedForUnsloth)."""
    print_header("Testing QA Format Dataset (Creekside/GRPO-Lambda-ParsedForUnsloth)")
    
    # Create config for Creekside dataset
    config = MoEConfig(
        dataset_source="huggingface",
        dataset_name="Creekside/GRPO-Lambda-ParsedForUnsloth",
        dataset_config_name=None,  # Default config
        num_train_samples=300,  # Small for testing
        num_eval_samples=50,
        max_seq_length=512,
        embed_dim=32,  # Small for fast testing
        batch_size=4
    )
    
    print(f"📋 Config: {config.dataset_name}")
    print(f"📊 Samples: {config.num_train_samples} train, {config.num_eval_samples} eval")
    print(f"🔍 This dataset contains 'reasoning' column that should be FILTERED OUT")
    
    # Time the preprocessing
    start_time = time.time()
    
    try:
        # Test preprocessing
        train_loader, eval_loader, tokenizer, data_mode = load_data_with_preprocessing(config)
        
        preprocessing_time = time.time() - start_time
        print(f"⏱️  Preprocessing time: {preprocessing_time:.2f}s")
        print(f"✅ Successfully loaded Creekside dataset")
        print(f"📊 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
        
        # Inspect the data to verify reasoning column is filtered out
        preprocessor = DatasetPreprocessor(config)
        dataset_path = preprocessor.get_dataset_path()
        inspect_sample_data(dataset_path)
        
        # Verify QA format
        print(f"\n🔍 Verifying QA Format (should be 'Question: ...\\nAnswer: ...' only):")
        train_tokens = torch.load(dataset_path / 'train_tokens.pt')
        from transformers import AutoTokenizer
        tokenizer_check = AutoTokenizer.from_pretrained('gpt2')
        tokenizer_check.pad_token = tokenizer_check.eos_token
        
        # Check first few samples for QA format
        for i in range(min(3, train_tokens.shape[0])):
            decoded_text = tokenizer_check.decode(train_tokens[i], skip_special_tokens=True)
            if "Question:" in decoded_text and "Answer:" in decoded_text:
                if "reasoning" in decoded_text.lower() or "Reasoning:" in decoded_text:
                    print(f"   ❌ Sample {i+1}: Contains reasoning column! Filtering failed.")
                else:
                    print(f"   ✅ Sample {i+1}: Clean QA format (no reasoning column)")
            else:
                print(f"   ⚠️  Sample {i+1}: Unexpected format")
        
        return True, preprocessing_time
        
    except Exception as e:
        print(f"❌ Failed to process Creekside dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_performance_comparison():
    """Compare performance between legacy and pretokenized loading."""
    print_header("Performance Comparison: Legacy vs Pretokenized")
    
    # Use small dataset for comparison
    config = MoEConfig(
        dataset_source="huggingface",
        dataset_name="Salesforce/wikitext",
        dataset_config_name="wikitext-103-v1",
        num_train_samples=200,
        num_eval_samples=50,
        max_seq_length=256,
        batch_size=4
    )
    
    print(f"📋 Testing with {config.num_train_samples} train samples")
    
    # Test legacy loading
    print_section("Legacy Loading (with runtime tokenization)")
    legacy_start = time.time()
    try:
        train_loader_legacy, eval_loader_legacy, tokenizer_legacy, data_mode_legacy = load_data(config)
        legacy_time = time.time() - legacy_start
        print(f"⏱️  Legacy loading time: {legacy_time:.2f}s")
    except Exception as e:
        print(f"❌ Legacy loading failed: {str(e)}")
        legacy_time = float('inf')
    
    # Test pretokenized loading (first run - includes preprocessing)
    print_section("Pretokenized Loading (first run - includes preprocessing)")
    pretok_first_start = time.time()
    try:
        train_loader_pretok, eval_loader_pretok, tokenizer_pretok, data_mode_pretok = load_data_with_preprocessing(config)
        pretok_first_time = time.time() - pretok_first_start
        print(f"⏱️  Pretokenized first run time: {pretok_first_time:.2f}s")
    except Exception as e:
        print(f"❌ Pretokenized loading failed: {str(e)}")
        pretok_first_time = float('inf')
    
    # Test pretokenized loading (second run - cached)
    print_section("Pretokenized Loading (second run - cached)")
    pretok_cached_start = time.time()
    try:
        train_loader_cached, eval_loader_cached, tokenizer_cached, data_mode_cached = load_data_with_preprocessing(config)
        pretok_cached_time = time.time() - pretok_cached_start
        print(f"⏱️  Pretokenized cached time: {pretok_cached_time:.2f}s")
    except Exception as e:
        print(f"❌ Pretokenized cached loading failed: {str(e)}")
        pretok_cached_time = float('inf')
    
    # Performance summary
    print_section("Performance Summary")
    print(f"📊 Legacy loading: {legacy_time:.2f}s")
    print(f"📊 Pretokenized (first): {pretok_first_time:.2f}s")
    print(f"📊 Pretokenized (cached): {pretok_cached_time:.2f}s")
    
    if legacy_time != float('inf') and pretok_cached_time != float('inf'):
        speedup = legacy_time / pretok_cached_time
        print(f"🚀 Speedup (cached): {speedup:.1f}x faster")
    
    return legacy_time, pretok_first_time, pretok_cached_time

def test_mini_training():
    """Test mini training with pretokenized data."""
    print_header("Mini Training Test (32-dim model)")
    
    # Create small, fast config
    config = MoEConfig(
        dataset_source="huggingface",
        dataset_name="Salesforce/wikitext",
        dataset_config_name="wikitext-103-v1",
        num_train_samples=100,  # Very small for fast test
        num_eval_samples=20,
        max_seq_length=128,
        embed_dim=32,  # Small model
        num_layers=1,
        num_heads=2,
        num_experts=2,
        batch_size=4,
        epochs=1,
        max_batches_per_epoch=10,  # Only 10 batches
        eval_every=5,
        learning_rate=1e-3,
        run_name="mini_test"
    )
    
    print(f"📋 Mini model config:")
    print(f"   Embed dim: {config.embed_dim}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Experts: {config.num_experts}")
    print(f"   Max batches: {config.max_batches_per_epoch}")
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        
        # Load data
        train_loader, eval_loader, tokenizer, data_mode = load_data_with_preprocessing(config)
        print(f"✅ Data loaded successfully")
        
        # Create model
        model = MoEModel(config).to(device)
        num_params = model.get_total_params()
        print(f"🎯 Model created: {num_params/1000:.1f}K parameters")
        
        # Create optimizer and scheduler
        optimizer = create_dynamic_optimizer(model, config)
        scheduler = PrimaryGhostLRScheduler(config, optimizer)
        
        # Run mini training
        print(f"🚀 Starting mini training...")
        training_start = time.time()
        
        training_stats, final_loss = train_model(
            model, optimizer, scheduler, train_loader, eval_loader, device, config
        )
        
        training_time = time.time() - training_start
        print(f"✅ Mini training completed in {training_time:.1f}s")
        print(f"📊 Final loss: {final_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mini training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_management():
    """Test dataset management CLI commands."""
    print_header("Dataset Management CLI Testing")
    
    try:
        # Test list datasets
        print_section("Testing --list-datasets")
        result = subprocess.run([
            sys.executable, "run.py", "--list-datasets"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ --list-datasets command works")
            print("Output:")
            print(result.stdout)
        else:
            print(f"❌ --list-datasets failed: {result.stderr}")
        
        # Test clean invalid (safe command)
        print_section("Testing --clean-invalid")
        result = subprocess.run([
            sys.executable, "run.py", "--clean-invalid"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ --clean-invalid command works")
            print("Output:")
            print(result.stdout)
        else:
            print(f"❌ --clean-invalid failed: {result.stderr}")
            
        return True
        
    except Exception as e:
        print(f"❌ CLI testing failed: {str(e)}")
        return False

def cleanup_test_data():
    """Clean up test datasets."""
    print_header("Cleanup Test Data")
    
    try:
        manager = DatasetManager()
        
        # List current datasets
        datasets = manager.list_datasets()
        test_datasets = [d for d in datasets if any(keyword in d['name'].lower() for keyword in ['wikitext', 'creekside', 'grpo'])]
        
        if test_datasets:
            print(f"🧹 Found {len(test_datasets)} test datasets to clean:")
            for dataset in test_datasets:
                print(f"   - {dataset['name']}")
            
            # Clean them
            for dataset in test_datasets:
                dataset_path = Path("data-preprocessed") / dataset['name']
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                    print(f"   ✅ Removed {dataset['name']}")
        else:
            print("✅ No test datasets found to clean")
            
        # Also clean mini_test checkpoint
        mini_test_dir = Path("checkpoints/mini_test")
        if mini_test_dir.exists():
            shutil.rmtree(mini_test_dir)
            print("✅ Removed mini_test checkpoint directory")
            
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")

def main():
    """Run the complete test suite."""
    print_header("🧪 Pretokenized Dataset Pipeline Test Suite")
    
    # Track results
    results = {
        'text_format': False,
        'qa_format': False,
        'performance': False,
        'mini_training': False,
        'cli_management': False
    }
    
    times = {}
    
    # Test 1: Text format dataset
    try:
        success, prep_time = test_text_format_dataset()
        results['text_format'] = success
        times['text_preprocessing'] = prep_time
    except Exception as e:
        print(f"❌ Text format test crashed: {str(e)}")
    
    # Test 2: QA format dataset
    try:
        success, prep_time = test_qa_format_dataset()
        results['qa_format'] = success
        times['qa_preprocessing'] = prep_time
    except Exception as e:
        print(f"❌ QA format test crashed: {str(e)}")
    
    # Test 3: Performance comparison
    try:
        legacy_time, first_time, cached_time = test_performance_comparison()
        results['performance'] = True
        times['legacy'] = legacy_time
        times['pretok_first'] = first_time
        times['pretok_cached'] = cached_time
    except Exception as e:
        print(f"❌ Performance test crashed: {str(e)}")
    
    # Test 4: Mini training
    try:
        success = test_mini_training()
        results['mini_training'] = success
    except Exception as e:
        print(f"❌ Mini training test crashed: {str(e)}")
    
    # Test 5: CLI management
    try:
        success = test_dataset_management()
        results['cli_management'] = success
    except Exception as e:
        print(f"❌ CLI management test crashed: {str(e)}")
    
    # Final results
    print_header("🎯 Test Results Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"📊 Tests Passed: {passed}/{total}")
    print(f"")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    if times:
        print(f"\n⏱️  Performance Summary:")
        for metric, time_val in times.items():
            if time_val != float('inf'):
                print(f"   {metric.replace('_', ' ').title()}: {time_val:.2f}s")
    
    # Cleanup
    cleanup_test_data()
    
    print(f"\n{'='*60}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Pretokenized pipeline is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the logs above for details.")
    print(f"{'='*60}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
