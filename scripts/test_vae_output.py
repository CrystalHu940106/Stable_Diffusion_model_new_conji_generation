#!/usr/bin/env python3
"""
Test VAE output dimensions
"""

import torch
from stable_diffusion_kanji import VAE

def test_vae_dimensions():
    """Test VAE input/output dimensions"""
    print("🔍 Testing VAE Dimensions")
    print("=" * 30)
    
    # Test different VAE configurations
    configs = [
        [128, 256, 512, 1024],  # Training config
        [64, 128, 256, 512],    # Smaller config
        [32, 64, 128, 256]      # Minimal config
    ]
    
    for hidden_dims in configs:
        print(f"\n📐 Testing VAE with hidden_dims: {hidden_dims}")
        
        try:
            vae = VAE(hidden_dims=hidden_dims)
            
            # Test input
            test_input = torch.randn(1, 3, 128, 128)
            print(f"   Input shape: {test_input.shape}")
            
            # Encode
            latents, mu, logvar, kl_loss = vae.encode(test_input)
            print(f"   Latents shape: {latents.shape}")
            
            # Decode
            output = vae.decode(latents)
            print(f"   Output shape: {output.shape}")
            
            # Check if output matches input
            if output.shape[-2:] == test_input.shape[-2:]:
                print("   ✅ Output dimensions match input!")
            else:
                print(f"   ❌ Dimension mismatch: {test_input.shape[-2:]} vs {output.shape[-2:]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_vae_dimensions()
