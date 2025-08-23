#!/usr/bin/env python3
"""
Quick Test for Stable Diffusion Components
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_basic_components():
    """Test basic component initialization"""
    print("🧪 Testing Basic Components")
    print("=" * 50)
    
    try:
        from stable_diffusion_kanji import VAE, UNet2DConditionModel, DDPMScheduler
        
        # Test VAE
        print("📊 Testing VAE initialization...")
        vae = VAE()
        print(f"   ✅ VAE created: {sum(p.numel() for p in vae.parameters()):,} parameters")
        
        # Test UNet
        print("📊 Testing UNet initialization...")
        unet = UNet2DConditionModel()
        print(f"   ✅ UNet created: {sum(p.numel() for p in unet.parameters()):,} parameters")
        
        # Test Scheduler
        print("📊 Testing DDPM Scheduler initialization...")
        scheduler = DDPMScheduler()
        print(f"   ✅ Scheduler created with {scheduler.num_train_timesteps} timesteps")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic component test failed: {e}")
        return False

def test_text_encoder():
    """Test CLIP text encoder"""
    print("\n🧪 Testing CLIP Text Encoder")
    print("=" * 50)
    
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        
        print("📊 Loading CLIP model...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        print(f"   ✅ Tokenizer loaded: vocab size {tokenizer.vocab_size}")
        print(f"   ✅ Text encoder loaded: {sum(p.numel() for p in text_encoder.parameters()):,} parameters")
        
        # Test tokenization
        test_text = "kanji character meaning: success"
        tokens = tokenizer(test_text, padding=True, return_tensors="pt")
        print(f"   ✅ Tokenization working: {tokens['input_ids'].shape}")
        
        # Test encoding
        with torch.no_grad():
            embeddings = text_encoder(**tokens).last_hidden_state
        print(f"   ✅ Encoding working: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text encoder test failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\n🧪 Testing Pipeline Initialization")
    print("=" * 50)
    
    try:
        from stable_diffusion_kanji import StableDiffusionPipeline
        
        print("📊 Initializing pipeline...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = StableDiffusionPipeline(device=device)
        
        print(f"   ✅ Pipeline initialized on {device}")
        print(f"   ✅ VAE: {pipeline.vae is not None}")
        print(f"   ✅ UNet: {pipeline.unet is not None}")
        print(f"   ✅ Text Encoder: {pipeline.text_encoder is not None}")
        print(f"   ✅ Tokenizer: {pipeline.tokenizer is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_simple_generation():
    """Test simple generation without training"""
    print("\n🧪 Testing Simple Generation")
    print("=" * 50)
    
    try:
        from stable_diffusion_kanji import StableDiffusionPipeline
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = StableDiffusionPipeline(device=device)
        
        print("📊 Testing text encoding...")
        test_prompt = "kanji character meaning: test"
        embeddings = pipeline.encode_text(test_prompt)
        print(f"   ✅ Text encoded: {embeddings.shape}")
        
        print("📊 Testing VAE encoding/decoding...")
        test_image = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            latents = pipeline.encode_image(test_image)
            decoded = pipeline.decode_latent(latents)
        print(f"   ✅ VAE working: {test_image.shape} → {latents.shape} → {decoded.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple generation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Quick Stable Diffusion Tests")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Basic Components", test_basic_components),
        ("Text Encoder", test_text_encoder),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Simple Generation", test_simple_generation)
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
    print("📋 QUICK TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quick tests passed! Ready for full testing.")
        print("\nNext steps:")
        print("   1. Run: python3 scripts/test_concept_generation.py")
        print("   2. Run: python3 scripts/train_stable_diffusion.py")
        print("   3. Run: python3 scripts/advanced_concept_generation.py")
    else:
        print("⚠️  Some quick tests failed. Please check the implementation.")
    
    return results

if __name__ == "__main__":
    main()
