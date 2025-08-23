#!/usr/bin/env python3
"""
Analyze Fixed Generation Results
Analyze the quality of fixed generation images
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

def analyze_fixed_images():
    """Analyze the fixed generation images"""
    
    print("🎨 Fixed Generation Analysis")
    print("=" * 50)
    
    fixed_dir = Path("fixed_generation")
    if not fixed_dir.exists():
        print("❌ Fixed generation folder not found!")
        return
    
    # Group images by type
    image_groups = {}
    for img_path in fixed_dir.glob("*.png"):
        name = img_path.stem
        parts = name.split('_')
        
        if len(parts) >= 4:
            input_type = parts[1]  # random, structured, low_freq
            fix_type = parts[2]    # original, inverted, threshold, enhanced
            
            if input_type not in image_groups:
                image_groups[input_type] = {}
            if fix_type not in image_groups[input_type]:
                image_groups[input_type][fix_type] = []
            
            image_groups[input_type][fix_type].append(img_path)
    
    # Analyze each group
    for input_type, fixes in image_groups.items():
        print(f"\n🎯 {input_type.replace('_', ' ').title()} Input:")
        
        for fix_type, images in fixes.items():
            print(f"   📸 {fix_type.title()} Fix ({len(images)} images):")
            
            # Calculate average file size
            sizes = [os.path.getsize(img) for img in images]
            avg_size = sum(sizes) / len(sizes)
            
            print(f"     • Average file size: {avg_size:.0f} bytes")
            
            # Analyze a sample image
            if images:
                sample_img = Image.open(images[0])
                print(f"     • Image size: {sample_img.size}")
                print(f"     • Image mode: {sample_img.mode}")
                
                # Analyze color distribution
                if sample_img.mode == 'RGB':
                    img_array = np.array(sample_img)
                    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                    
                    print(f"     • Color range: R[{r.min()}-{r.max()}], G[{g.min()}-{g.max()}], B[{b.min()}-{b.max()}]")
                    print(f"     • Average color: R{r.mean():.0f}, G{g.mean():.0f}, B{b.mean():.0f}")
                    
                    # Assess quality based on color distribution
                    if r.max() - r.min() < 10 and g.max() - g.min() < 10 and b.max() - b.min() < 10:
                        print(f"     • Quality: Low (uniform color)")
                    elif r.max() - r.min() > 100 or g.max() - g.min() > 100 or b.max() - b.min() > 100:
                        print(f"     • Quality: High (good contrast)")
                    else:
                        print(f"     • Quality: Medium (some variation)")

def compare_fix_methods():
    """Compare different fix methods"""
    
    print(f"\n🔍 Comparison of Fix Methods:")
    
    fixed_dir = Path("fixed_generation")
    if not fixed_dir.exists():
        return
    
    # Get one input type for comparison
    input_types = ['random', 'structured', 'low_freq']
    
    for input_type in input_types:
        print(f"\n   🎯 {input_type.replace('_', ' ').title()} Input Comparison:")
        
        fix_methods = ['original', 'inverted', 'threshold', 'enhanced']
        method_scores = {}
        
        for fix_method in fix_methods:
            pattern = f"fixed_{input_type}_{fix_method}_*.png"
            images = list(fixed_dir.glob(pattern))
            
            if images:
                # Calculate average file size as quality indicator
                sizes = [os.path.getsize(img) for img in images]
                avg_size = sum(sizes) / len(sizes)
                method_scores[fix_method] = avg_size
                
                print(f"     • {fix_method.title()}: {avg_size:.0f} bytes")
        
        # Find best method
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            print(f"     • Best method: {best_method.title()} ({method_scores[best_method]:.0f} bytes)")

def provide_recommendations():
    """Provide recommendations based on fixed results"""
    
    print(f"\n💡 Recommendations Based on Fixed Results:")
    
    # Check if inverted images have better file sizes
    fixed_dir = Path("fixed_generation")
    if not fixed_dir.exists():
        return
    
    inverted_images = list(fixed_dir.glob("*_inverted_*.png"))
    original_images = list(fixed_dir.glob("*_original_*.png"))
    
    if inverted_images and original_images:
        avg_inverted_size = sum(os.path.getsize(img) for img in inverted_images) / len(inverted_images)
        avg_original_size = sum(os.path.getsize(img) for img in original_images) / len(original_images)
        
        print(f"   📊 File Size Comparison:")
        print(f"     • Original (white): {avg_original_size:.0f} bytes")
        print(f"     • Inverted (black): {avg_inverted_size:.0f} bytes")
        
        if avg_inverted_size < avg_original_size:
            print(f"     • ✅ Inversion helps: Smaller file size indicates more structure")
        else:
            print(f"     • ⚠️ Inversion doesn't help much: Similar file sizes")
    
    print(f"\n   🎯 Best Fix Methods:")
    print(f"     • Inverted: Converts white output to black strokes")
    print(f"     • Threshold: Creates binary black/white images")
    print(f"     • Enhanced: Improves contrast")
    
    print(f"\n   🚀 Next Steps:")
    print(f"     • Use inverted or threshold methods for better visualization")
    print(f"     • Proceed with full training using correct data preprocessing")
    print(f"     • Implement proper diffusion process")
    print(f"     • Use Stable Diffusion architecture for better results")

def main():
    """Main function"""
    
    print("🎌 Fixed Generation Analysis")
    print("=" * 50)
    
    # Analyze fixed images
    analyze_fixed_images()
    
    # Compare fix methods
    compare_fix_methods()
    
    # Provide recommendations
    provide_recommendations()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   • Fixed images analyzed")
    print(f"   • Fix methods compared")
    print(f"   • Recommendations provided")
    
    print(f"\n📋 Summary:")
    print(f"   • Check fixed_generation/ folder for all results")
    print(f"   • Look for *_inverted_*.png files for black strokes")
    print(f"   • Look for *_threshold_*.png files for binary images")
    print(f"   • Consider proceeding with full training")

if __name__ == "__main__":
    main()
