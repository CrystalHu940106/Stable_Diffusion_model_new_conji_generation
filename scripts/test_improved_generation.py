#!/usr/bin/env python3
"""
Improved test script for generating high-quality Kanji
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
from stable_diffusion_kanji import StableDiffusionPipeline
import cv2

def enhance_image_quality(image_pil):
    """Enhance image quality for better Kanji visibility"""
    # Convert to numpy for OpenCV processing
    img_np = np.array(image_pil)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Apply adaptive thresholding for better stroke definition
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Convert back to PIL
    enhanced_pil = Image.fromarray(thresh)
    
    return enhanced_pil

def test_improved_generation():
    """Test improved generation with better parameters"""
    print("🎌 Improved Kanji Generation Test")
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
    
    # Load trained model
    print("📂 Loading trained model...")
    try:
        checkpoint = torch.load('best_model.pth', map_location=device)
        pipeline.vae.load_state_dict(checkpoint['vae_state_dict'])
        pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test concepts with improved parameters
    concepts = ["water", "future"]
    
    for concept in concepts:
        print(f"\n🌊 Generating Kanji for: {concept.upper()}")
        print("-" * 30)
        
        try:
            # Generate with different parameters
            for guidance_scale in [7.0, 9.0, 11.0]:
                for num_steps in [50, 100]:
                    print(f"   Guidance: {guidance_scale}, Steps: {num_steps}")
                    
                    # Generate image with more steps
                    image = pipeline.generate(
                        prompt=f"kanji character for {concept}, traditional calligraphy, black ink on white paper, high contrast, detailed strokes",
                        height=128,
                        width=128,
                        num_inference_steps=num_steps,
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
                    
                    # Enhance image quality
                    enhanced_pil = enhance_image_quality(image_pil)
                    
                    # Save both original and enhanced
                    orig_filename = f"kanji_{concept}_g{guidance_scale}_s{num_steps}_orig.png"
                    enhanced_filename = f"kanji_{concept}_g{guidance_scale}_s{num_steps}_enhanced.png"
                    
                    image_pil.save(orig_filename)
                    enhanced_pil.save(enhanced_filename)
                    
                    print(f"   💾 Saved: {orig_filename}, {enhanced_filename}")
                    
                    # Display comparison
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    
                    ax1.imshow(image_pil)
                    ax1.set_title(f'Original (G:{guidance_scale}, S:{num_steps})')
                    ax1.axis('off')
                    
                    ax2.imshow(enhanced_pil, cmap='gray')
                    ax2.set_title(f'Enhanced (G:{guidance_scale}, S:{num_steps})')
                    ax2.axis('off')
                    
                    plt.suptitle(f'Kanji for "{concept}" - Quality Comparison')
                    plt.tight_layout()
                    plt.show()
                    
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Improved generation test completed!")
    print(f"📁 Generated images saved in current directory")

if __name__ == "__main__":
    test_improved_generation()
