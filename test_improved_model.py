#!/usr/bin/env python3
"""
Test script for the improved stable diffusion model
"""

import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from improved_stable_diffusion import ImprovedStableDiffusionPipeline
    print("✅ Successfully imported ImprovedStableDiffusionPipeline")
    
    # Test model initialization
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    pipeline = ImprovedStableDiffusionPipeline(device=device)
    print("✅ Model initialized successfully")
    
    # Test generation
    print("🌊 Testing generation...")
    result = pipeline.generate(
        "water",
        height=128,
        width=128,
        num_inference_steps=10,  # Reduced for testing
        guidance_scale=7.5,
        seed=42
    )
    print("✅ Generation completed successfully")
    print(f"📊 Result shape: {result.shape if hasattr(result, 'shape') else 'Unknown'}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and the scripts folder exists")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
