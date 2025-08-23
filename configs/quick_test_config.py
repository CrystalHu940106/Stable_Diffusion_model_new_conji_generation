#!/usr/bin/env python3
"""
Quick Test Configuration for Kanji Diffusion Model
Ultra-fast configuration for testing the training pipeline
"""

import json
from pathlib import Path

# Quick test configuration for rapid validation
QUICK_TEST_CONFIG = {
    # Image settings - ultra fast
    "image_size": 64,  # Very small for fastest training
    "batch_size": 8,   # Larger batch for efficiency
    "learning_rate": 5e-4,  # Higher LR for fast convergence
    "num_epochs": 2,   # Just 2 epochs for testing
    
    # Subset configuration
    "test_dataset_size": 500,  # Use only 500 samples for testing
    "train_split": 0.8,
    
    # Estimated times
    "steps_per_epoch": 500 // 8,  # ~63 steps per epoch
    "minutes_per_epoch": 15,      # ~15 minutes per epoch
    "total_test_time": 30,        # ~30 minutes total
}

def create_test_dataset():
    """Create a small subset of the dataset for quick testing"""
    
    # Load full dataset
    dataset_path = Path("data/fixed_kanji_dataset/metadata/dataset.json")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please run fix_kanji_dataset.py first.")
        return False
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    # Create test subset
    test_size = QUICK_TEST_CONFIG['test_dataset_size']
    test_dataset = full_dataset[:test_size]
    
    # Save test dataset
    test_path = Path("data/fixed_kanji_dataset/metadata/test_dataset.json")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created test dataset with {len(test_dataset)} samples")
    print(f"   Saved to: {test_path}")
    
    return True

def print_quick_test_summary():
    """Print summary of quick test configuration"""
    
    print("🚀 Quick Test Configuration for Training Pipeline")
    print("=" * 52)
    
    config = QUICK_TEST_CONFIG
    
    print(f"\n⚡ Ultra-Fast Settings:")
    print(f"   • Image size: {config['image_size']}x{config['image_size']} pixels")
    print(f"   • Dataset size: {config['test_dataset_size']} samples")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Epochs: {config['num_epochs']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    
    print(f"\n⏱️ Time Estimates:")
    print(f"   • Steps per epoch: {config['steps_per_epoch']}")
    print(f"   • Time per epoch: ~{config['minutes_per_epoch']} minutes")
    print(f"   • Total test time: ~{config['total_test_time']} minutes")
    
    print(f"\n🎯 Purpose:")
    print(f"   • Verify training pipeline works correctly")
    print(f"   • Test data loading and processing")
    print(f"   • Validate model architecture")
    print(f"   • Check checkpoint saving/loading")
    print(f"   • Quick iteration for debugging")
    
    print(f"\n📋 Usage:")
    print(f"   1. python3 quick_test_config.py  # Create test dataset")
    print(f"   2. Modify training script to use test_dataset.json")
    print(f"   3. Run training with ultra-fast settings")
    print(f"   4. If successful, proceed with full training")
    
    print(f"\n✅ Benefits:")
    print(f"   • Rapid validation (30 minutes vs 18 hours)")
    print(f"   • Early error detection")
    print(f"   • Resource efficient")
    print(f"   • Quick experimentation")

def main():
    """Main function for quick test setup"""
    
    print_quick_test_summary()
    
    print(f"\n" + "=" * 52)
    print("Creating test dataset...")
    
    if create_test_dataset():
        print(f"\n🎉 Quick test setup complete!")
        print(f"\nNext steps:")
        print(f"   1. Use test_dataset.json for training")
        print(f"   2. Set image_size=64, batch_size=8")
        print(f"   3. Run 2 epochs (~30 minutes)")
        print(f"   4. Validate training works correctly")
    else:
        print(f"\n❌ Failed to create test dataset")

if __name__ == "__main__":
    main()
