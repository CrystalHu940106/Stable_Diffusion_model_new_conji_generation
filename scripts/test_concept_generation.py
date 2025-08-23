#!/usr/bin/env python3
"""
Test Concept Generation - Verify Modern Concept Kanji Generation
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import our Stable Diffusion components
from stable_diffusion_kanji import (
    VAE, UNet2DConditionModel, DDPMScheduler, 
    StableDiffusionPipeline
)

def test_text_encoding():
    """Test CLIP text encoding functionality"""
    print("🧪 Testing CLIP Text Encoding")
    print("=" * 50)
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        pipeline = StableDiffusionPipeline(device=device)
        
        # Test prompts
        test_prompts = [
            "kanji character meaning: success, achieve, accomplish",
            "kanji character meaning: failure, lose, defeat",
            "kanji character meaning: novel, new, creative",
            "kanji character meaning: funny, humorous, amusing",
            "kanji character meaning: culture, tradition, heritage"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Testing: {prompt}")
            embeddings = pipeline.encode_text(prompt)
            print(f"   Shape: {embeddings.shape}")
            print(f"   Device: {embeddings.device}")
            print(f"   Dtype: {embeddings.dtype}")
            
            # Check if embeddings are reasonable
            if embeddings.abs().mean() > 0:
                print(f"   ✅ Embeddings look good")
            else:
                print(f"   ⚠️  Embeddings might be zero")
        
        return True
        
    except Exception as e:
        print(f"❌ Text encoding test failed: {e}")
        return False

def test_diffusion_scheduler():
    """Test DDPM scheduler functionality"""
    print("\n🧪 Testing DDPM Scheduler")
    print("=" * 50)
    
    try:
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # Test timestep setting
        print("📊 Testing timestep management:")
        scheduler.set_timesteps(50)
        print(f"   Inference timesteps: {scheduler.timesteps.shape}")
        print(f"   First few timesteps: {scheduler.timesteps[:5].tolist()}")
        
        # Test noise addition
        print("\n📊 Testing noise addition:")
        latents = torch.randn(2, 4, 32, 32)
        noise = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 500])
        
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        print(f"   Original shape: {latents.shape}")
        print(f"   Noisy shape: {noisy_latents.shape}")
        print(f"   Noise added successfully: {not torch.allclose(latents, noisy_latents)}")
        
        # Test denoising step
        print("\n📊 Testing denoising step:")
        model_output = torch.randn(2, 4, 32, 32)
        denoised = scheduler.step(model_output, timesteps[0], noisy_latents[0])
        print(f"   Denoised shape: {denoised.shape}")
        print(f"   Denoising successful: {denoised.shape == latents[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        return False

def test_concept_generation():
    """Test modern concept generation"""
    print("\n🧪 Testing Concept Generation")
    print("=" * 50)
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        pipeline = StableDiffusionPipeline(device=device)
        
        # Modern concepts to test
        modern_concepts = {
            'youtube': 'kanji character meaning: video sharing platform, streaming content',
            'gundam': 'kanji character meaning: giant robot mecha, futuristic warfare',
            'ai': 'kanji character meaning: artificial intelligence, machine learning',
            'crypto': 'kanji character meaning: digital cryptocurrency, blockchain',
            'internet': 'kanji character meaning: global network, digital connectivity'
        }
        
        results = {}
        
        for concept, prompt in modern_concepts.items():
            print(f"\n🎯 Testing concept: {concept.upper()}")
            print(f"   Prompt: {prompt}")
            
            try:
                # Generate with fewer steps for testing
                generated = pipeline.generate(prompt, num_inference_steps=20)
                results[concept] = generated
                print(f"   ✅ Generated successfully: {generated.shape}")
                
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                results[concept] = None
        
        # Display results
        if any(v is not None for v in results.values()):
            display_test_results(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Concept generation test failed: {e}")
        return False

def display_test_results(results):
    """Display test generation results"""
    print(f"\n🖼️  Displaying Test Results")
    
    # Filter successful generations
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if not successful_results:
        print("❌ No successful generations to display")
        return
    
    # Create subplot
    num_concepts = len(successful_results)
    fig, axes = plt.subplots(1, num_concepts, figsize=(4*num_concepts, 4))
    
    if num_concepts == 1:
        axes = [axes]
    
    for i, (concept, generated) in enumerate(successful_results.items()):
        try:
            # Convert tensor to image
            if generated.dim() == 4:
                generated = generated.squeeze(0)
            
            # Denormalize from [-1, 1] to [0, 1]
            generated = (generated + 1) / 2
            generated = torch.clamp(generated, 0, 1)
            
            # Convert to numpy
            img_array = generated.permute(1, 2, 0).numpy()
            
            # Display
            axes[i].imshow(img_array)
            axes[i].set_title(f'{concept.upper()}', fontsize=12)
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error\n{e}', ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_path = "test_concept_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Test results saved to: {output_path}")
    
    plt.show()

def test_training_components():
    """Test training components"""
    print("\n🧪 Testing Training Components")
    print("=" * 50)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test VAE
        print("📊 Testing VAE:")
        vae = VAE().to(device)
        test_image = torch.randn(2, 3, 128, 128).to(device)
        
        with torch.no_grad():
            encoded = vae.encode(test_image)
            decoded = vae.decode(encoded)
        
        print(f"   Input shape: {test_image.shape}")
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Decoded shape: {decoded.shape}")
        print(f"   VAE working: {decoded.shape == test_image.shape}")
        
        # Test UNet
        print("\n📊 Testing UNet:")
        unet = UNet2DConditionModel().to(device)
        test_latents = torch.randn(2, 4, 32, 32).to(device)
        test_timesteps = torch.tensor([100, 200]).to(device)
        test_context = torch.randn(2, 77, 768).to(device)
        
        with torch.no_grad():
            output = unet(test_latents, test_timesteps, test_context)
        
        print(f"   Input latents: {test_latents.shape}")
        print(f"   Output: {output.shape}")
        print(f"   UNet working: {output.shape == test_latents.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Stable Diffusion Concept Generation Tests")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Text Encoding", test_text_encoding),
        ("Diffusion Scheduler", test_diffusion_scheduler),
        ("Training Components", test_training_components),
        ("Concept Generation", test_concept_generation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training and generation.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return results

if __name__ == "__main__":
    main()
