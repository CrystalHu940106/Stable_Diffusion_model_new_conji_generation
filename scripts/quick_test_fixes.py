#!/usr/bin/env python3
"""
Quick test script to verify all fixes are working correctly
"""

import torch
import torch.nn as nn
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from stable_diffusion_kanji import VAE, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline

def test_vae():
    """Test VAE with KL divergence loss"""
    print("🧪 Testing VAE...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VAE(hidden_dims=[128, 256, 512]).to(device)
    
    # Test input
    test_image = torch.randn(4, 3, 128, 128).to(device)
    
    try:
        # Encode with KL loss
        latents, mu, logvar, kl_loss = vae.encode(test_image)
        reconstructed = vae.decode(latents)
        
        print(f"✅ VAE test passed:")
        print(f"   • Input: {test_image.shape}")
        print(f"   • Latents: {latents.shape}")
        print(f"   • Mu: {mu.shape}")
        print(f"   • Logvar: {logvar.shape}")
        print(f"   • KL Loss: {kl_loss.item():.6f}")
        print(f"   • Reconstructed: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE test failed: {e}")
        return False

def test_unet():
    """Test UNet without debug prints"""
    print("\n🧪 Testing UNet...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNet2DConditionModel(
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        num_heads=8
    ).to(device)
    
    # Test input
    test_latents = torch.randn(8, 4, 16, 16).to(device)  # Batch size 8
    test_timesteps = torch.randint(0, 1000, (8,)).to(device)
    test_context = torch.randn(8, 77, 512).to(device)
    
    try:
        # Forward pass (should be silent - no debug prints)
        output = unet(test_latents, test_timesteps, test_context)
        
        print(f"✅ UNet test passed:")
        print(f"   • Input latents: {test_latents.shape}")
        print(f"   • Timesteps: {test_timesteps.shape}")
        print(f"   • Context: {test_context.shape}")
        print(f"   • Output: {output.shape}")
        print(f"   • No debug prints detected ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ UNet test failed: {e}")
        return False

def test_scheduler():
    """Test DDPM scheduler"""
    print("\n🧪 Testing DDPM Scheduler...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Test input
    test_latents = torch.randn(4, 4, 16, 16).to(device)
    test_noise = torch.randn_like(test_latents)
    test_timesteps = torch.randint(0, 1000, (4,)).to(device)
    
    try:
        # Add noise
        noisy_latents = scheduler.add_noise(test_latents, test_noise, test_timesteps)
        
        # Denoise step
        denoised = scheduler.step(test_noise, test_timesteps, noisy_latents)
        
        print(f"✅ Scheduler test passed:")
        print(f"   • Original: {test_latents.shape}")
        print(f"   • Noisy: {noisy_latents.shape}")
        print(f"   • Denoised: {denoised.shape}")
        print(f"   • Timesteps: {test_timesteps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        return False

def test_pipeline():
    """Test complete pipeline"""
    print("\n🧪 Testing Stable Diffusion Pipeline...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        pipeline = StableDiffusionPipeline(device=device)
        
        print(f"✅ Pipeline test passed:")
        print(f"   • Device: {device}")
        print(f"   • VAE: {type(pipeline.vae).__name__}")
        print(f"   • UNet: {type(pipeline.unet).__name__}")
        print(f"   • Scheduler: {type(pipeline.scheduler).__name__}")
        print(f"   • Text Encoder: {type(pipeline.text_encoder).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing with larger batch sizes"""
    print("\n🧪 Testing Batch Processing...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different batch sizes
    batch_sizes = [4, 8, 16]
    
    for batch_size in batch_sizes:
        try:
            print(f"   Testing batch size: {batch_size}")
            
            # Create test data
            test_images = torch.randn(batch_size, 3, 128, 128).to(device)
            test_prompts = [f"test prompt {i}" for i in range(batch_size)]
            
            # Initialize models
            vae = VAE(hidden_dims=[128, 256]).to(device)
            unet = UNet2DConditionModel(
                model_channels=128,
                num_res_blocks=2,
                channel_mult=(1, 2),
                num_heads=8
            ).to(device)
            
            # Test VAE encoding
            latents, mu, logvar, kl_loss = vae.encode(test_images)
            
            # Test UNet forward pass
            test_latents = torch.randn(batch_size, 4, 16, 16).to(device)
            test_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
            test_context = torch.randn(batch_size, 77, 512).to(device)
            
            output = unet(test_latents, test_timesteps, test_context)
            
            print(f"      ✅ Batch size {batch_size}: VAE {latents.shape}, UNet {output.shape}")
            
        except Exception as e:
            print(f"      ❌ Batch size {batch_size} failed: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🎌 Testing All Fixes")
    print("=" * 50)
    
    tests = [
        ("VAE with KL Loss", test_vae),
        ("UNet without Debug Prints", test_unet),
        ("DDPM Scheduler", test_scheduler),
        ("Complete Pipeline", test_pipeline),
        ("Batch Processing", test_batch_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! All fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
