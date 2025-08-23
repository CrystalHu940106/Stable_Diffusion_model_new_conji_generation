#!/usr/bin/env python3
"""
Optimized Training Configuration for Kanji Diffusion Model
Based on task requirements: low resolution, fast training, CPU-friendly
"""

# Training Configuration for CPU-optimized low-resolution training
TRAINING_CONFIG = {
    # Image settings - optimized for speed
    "image_size": 128,  # Start with 128x128, can try 256x256 if needed
    "image_channels": 3,
    
    # Training parameters - optimized for CPU
    "batch_size": 4,  # Small batch for CPU training
    "learning_rate": 2e-4,  # Higher LR for faster convergence
    "num_epochs": 5,  # Sufficient for 6K+ dataset
    "warmup_steps": 100,
    
    # Model configuration - small stable diffusion
    "model_type": "small_unet",  # Smaller model for faster training
    "attention_resolutions": [16, 8],  # Reduced attention layers
    "channel_mult": [1, 2, 4],  # Simpler channel progression
    "num_res_blocks": 2,  # Fewer residual blocks
    
    # Optimization settings
    "gradient_clip_val": 1.0,
    "mixed_precision": False,  # Disabled for CPU
    "optimizer": "adamw",
    "weight_decay": 0.01,
    
    # Data settings
    "dataset_size": 6410,  # Full dataset as mentioned in requirements
    "train_split": 0.9,  # 90% for training, 10% for validation
    "shuffle": True,
    
    # Checkpoint and logging
    "save_every": 1,  # Save every epoch
    "log_every": 50,  # Log every 50 steps
    "max_checkpoints": 3,
    
    # Generation settings
    "num_inference_steps": 20,  # Fewer steps for faster generation
    "guidance_scale": 7.5,
}

# Time estimates based on optimized configuration
TIME_ESTIMATES = {
    "steps_per_epoch": 6410 // 4,  # ~1603 steps per epoch
    "seconds_per_step_cpu": 8,  # Estimated for 128x128 on M3 Pro
    "minutes_per_epoch": (6410 // 4 * 8) // 60,  # ~214 minutes per epoch
    "hours_per_epoch": 3.6,  # ~3.6 hours per epoch
    "total_training_time_hours": 3.6 * 5,  # ~18 hours for 5 epochs
    "total_training_time_days": 0.75,  # Less than 1 day
}

# Alternative configurations for different speed/quality tradeoffs
ALTERNATIVE_CONFIGS = {
    "ultra_fast": {
        "image_size": 64,
        "batch_size": 8,
        "num_epochs": 3,
        "estimated_hours": 6,
    },
    "balanced": {
        "image_size": 128,
        "batch_size": 4,
        "num_epochs": 5,
        "estimated_hours": 18,
    },
    "higher_quality": {
        "image_size": 256,
        "batch_size": 2,
        "num_epochs": 5,
        "estimated_hours": 36,
    }
}

def print_training_summary():
    """Print a summary of the optimized training configuration"""
    print("🎌 Optimized Kanji Diffusion Training Configuration")
    print("=" * 55)
    
    print(f"\n📊 Dataset Information:")
    print(f"   • Total Kanji characters: {TRAINING_CONFIG['dataset_size']:,}")
    print(f"   • Training samples: {int(TRAINING_CONFIG['dataset_size'] * TRAINING_CONFIG['train_split']):,}")
    print(f"   • Validation samples: {int(TRAINING_CONFIG['dataset_size'] * (1 - TRAINING_CONFIG['train_split'])):,}")
    
    print(f"\n🖼️ Image Configuration:")
    print(f"   • Resolution: {TRAINING_CONFIG['image_size']}x{TRAINING_CONFIG['image_size']} pixels")
    print(f"   • Channels: {TRAINING_CONFIG['image_channels']} (RGB)")
    print(f"   • Format: PNG, black strokes on white background")
    
    print(f"\n⚙️ Training Parameters:")
    print(f"   • Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   • Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   • Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   • Steps per epoch: {TIME_ESTIMATES['steps_per_epoch']:,}")
    print(f"   • Optimizer: {TRAINING_CONFIG['optimizer'].upper()}")
    
    print(f"\n⏱️ Time Estimates:")
    print(f"   • Time per epoch: ~{TIME_ESTIMATES['hours_per_epoch']:.1f} hours")
    print(f"   • Total training time: ~{TIME_ESTIMATES['total_training_time_hours']:.1f} hours")
    print(f"   • Estimated completion: ~{TIME_ESTIMATES['total_training_time_days']:.1f} days")
    
    print(f"\n🚀 Hardware Optimization:")
    print(f"   • Platform: Apple M3 Pro (CPU training)")
    print(f"   • Memory: 36 GB (sufficient)")
    print(f"   • Mixed precision: {TRAINING_CONFIG['mixed_precision']} (CPU optimization)")
    print(f"   • Workers: 0 (single-threaded for CPU)")
    
    print(f"\n📈 Alternative Configurations:")
    for name, config in ALTERNATIVE_CONFIGS.items():
        print(f"   • {name.replace('_', ' ').title()}:")
        print(f"     - Resolution: {config['image_size']}x{config['image_size']}")
        print(f"     - Batch size: {config['batch_size']}")
        print(f"     - Epochs: {config['num_epochs']}")
        print(f"     - Estimated time: ~{config['estimated_hours']} hours")
    
    print(f"\n✅ Task Requirements Met:")
    print(f"   • ✅ Low resolution images (128x128)")
    print(f"   • ✅ Thousands of entries (6,410 Kanji)")
    print(f"   • ✅ Faster training (optimized for CPU)")
    print(f"   • ✅ Reduced compute requirements")
    print(f"   • ✅ Author's suggestion: better results at low resolution")

if __name__ == "__main__":
    print_training_summary()
