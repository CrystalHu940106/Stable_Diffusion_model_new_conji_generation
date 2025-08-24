#!/usr/bin/env python3
"""
Test script using the fixed VAE model
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from stable_diffusion_kanji import StableDiffusionPipeline

def test_fixed_model():
    """Test generation with fixed VAE model"""
    print("🎌 Testing Fixed VAE Model for Kanji Generation")
    print("=" * 50)
    
    # Initialize pipeline with correct configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    from stable_diffusion_kanji import VAE, UNet2DConditionModel, DDPMScheduler
    
    vae = VAE(hidden_dims=[128, 256, 512, 1024]).to(device)
    unet = UNet2DConditionModel(
        model_channels=256,
        num_res_blocks=3,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(8,),
        num_heads=16
    ).to(device)
    
    pipeline = StableDiffusionPipeline(device=device)
    pipeline.vae = vae
    pipeline.unet = unet
    
    # Load fixed model
    print("📂 Loading fixed VAE model...")
    try:
        checkpoint = torch.load('fixed_vae_checkpoint_epoch_5.pth', map_location=device)
        pipeline.vae.load_state_dict(checkpoint['vae_state_dict'])
        pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
        print("✅ Fixed model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load fixed model: {e}")
        return
    
    # Test concepts
    concepts = ["water", "future"]
    
    for concept in concepts:
        print(f"\n🌊 Generating Kanji for: {concept.upper()}")
        print("-" * 30)
        
        try:
            # Generate with different parameters
            for guidance_scale in [7.0, 9.0, 11.0]:
                print(f"   Guidance Scale: {guidance_scale}")
                
                # Generate image
                image = pipeline.generate(
                    prompt=f"kanji character for {concept}, traditional calligraphy, black ink on white paper, high contrast, detailed strokes",
                    height=128,
                    width=128,
                    num_inference_steps=50,
                    guidance_scale=guidance_scale,
                    seed=42  # Fixed seed for reproducibility
                )
                
                # Convert to PIL image
                if isinstance(image, torch.Tensor):
                    # Denormalize from [-1, 1] to [0, 1]
                    image = (image + 1) / 2
                    image = torch.clamp(image, 0, 1)
                    
                    # Convert to PIL
                    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                else:
                    image_pil = image
                
                # Save image
                filename = f"fixed_kanji_{concept}_{guidance_scale}.png"
                image_pil.save(filename)
                print(f"   💾 Saved: {filename}")
                
                # Display image
                plt.figure(figsize=(6, 6))
                plt.imshow(image_pil)
                plt.title(f'Fixed Model: Kanji for "{concept}" (Guidance: {guidance_scale})')
                plt.axis('off')
                plt.show()
                
                # Print image statistics
                img_array = np.array(image_pil)
                if len(img_array.shape) == 3:
                    gray = np.mean(img_array, axis=2)
                else:
                    gray = img_array
                
                print(f"   📊 Image stats: size={image_pil.size}, mean={gray.mean():.2f}, std={gray.std():.2f}")
                
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Fixed model test completed!")
    print(f"📁 Generated images saved in current directory")

if __name__ == "__main__":
    test_fixed_model()
