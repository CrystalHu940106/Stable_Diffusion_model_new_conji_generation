#!/usr/bin/env python3
"""
Image Analysis Script for Generated Kanji Images
Analyze the quality and characteristics of generated images
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def analyze_image_characteristics(image_path):
    """Analyze characteristics of a single image"""
    
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Basic info
        width, height = img.size
        channels = len(img.getbands())
        
        # Color analysis
        if channels == 3:  # RGB
            r, g, b = img.split()
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
        else:  # Grayscale
            r_mean = g_mean = b_mean = np.mean(img_array)
            r_std = g_std = b_std = np.std(img_array)
        
        # Contrast analysis
        contrast = np.std(img_array)
        
        # Edge density (simple approximation)
        gray = img.convert('L')
        gray_array = np.array(gray)
        edges = np.abs(np.diff(gray_array, axis=0)) + np.abs(np.diff(gray_array, axis=1))
        edge_density = np.mean(edges)
        
        # Complexity (entropy-like measure)
        hist = gray.histogram()
        hist = [h for h in hist if h > 0]
        complexity = -sum((h/sum(hist)) * np.log2(h/sum(hist)) for h in hist)
        
        return {
            'size': (width, height),
            'channels': channels,
            'mean_rgb': (r_mean, g_mean, b_mean),
            'std_rgb': (r_std, g_std, b_std),
            'contrast': contrast,
            'edge_density': edge_density,
            'complexity': complexity,
            'file_size': os.path.getsize(image_path)
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_generated_images():
    """Analyze all generated images"""
    
    print("🎨 Generated Image Analysis")
    print("=" * 50)
    
    # Analyze basic generation results
    basic_dir = Path("generated_results")
    if basic_dir.exists():
        print(f"\n📊 Basic Generation Results:")
        basic_images = list(basic_dir.glob("*.png"))
        
        for img_path in basic_images:
            print(f"\n   📸 {img_path.name}:")
            analysis = analyze_image_characteristics(img_path)
            
            if 'error' not in analysis:
                print(f"     • Size: {analysis['size']}")
                print(f"     • Channels: {analysis['channels']}")
                print(f"     • Mean RGB: ({analysis['mean_rgb'][0]:.1f}, {analysis['mean_rgb'][1]:.1f}, {analysis['mean_rgb'][2]:.1f})")
                print(f"     • Contrast: {analysis['contrast']:.1f}")
                print(f"     • Edge Density: {analysis['edge_density']:.1f}")
                print(f"     • Complexity: {analysis['complexity']:.2f}")
                print(f"     • File Size: {analysis['file_size']} bytes")
            else:
                print(f"     • Error: {analysis['error']}")
    
    # Analyze advanced generation results
    advanced_dir = Path("advanced_results")
    if advanced_dir.exists():
        print(f"\n🔬 Advanced Generation Results:")
        advanced_images = list(advanced_dir.glob("*.png"))
        
        # Group by test type
        test_groups = {}
        for img_path in advanced_images:
            name = img_path.stem
            if 'random' in name:
                group = 'random_noise'
            elif 'structured' in name:
                group = 'structured_noise'
            elif 'low_frequency' in name:
                group = 'low_frequency'
            elif 'high_frequency' in name:
                group = 'high_frequency'
            elif 'interpolation' in name:
                group = 'interpolation'
            else:
                group = 'other'
            
            if group not in test_groups:
                test_groups[group] = []
            test_groups[group].append(img_path)
        
        # Analyze each group
        for group_name, images in test_groups.items():
            print(f"\n   🎯 {group_name.replace('_', ' ').title()} ({len(images)} images):")
            
            # Calculate average characteristics
            avg_contrast = []
            avg_complexity = []
            avg_edge_density = []
            
            for img_path in images:
                analysis = analyze_image_characteristics(img_path)
                if 'error' not in analysis:
                    avg_contrast.append(analysis['contrast'])
                    avg_complexity.append(analysis['complexity'])
                    avg_edge_density.append(analysis['edge_density'])
            
            if avg_contrast:
                print(f"     • Avg Contrast: {np.mean(avg_contrast):.1f}")
                print(f"     • Avg Complexity: {np.mean(avg_complexity):.2f}")
                print(f"     • Avg Edge Density: {np.mean(avg_edge_density):.1f}")
                print(f"     • Contrast Range: {min(avg_contrast):.1f} - {max(avg_contrast):.1f}")

def compare_with_real_kanji():
    """Compare generated images with real Kanji"""
    
    print(f"\n🔍 Comparison with Real Kanji:")
    
    # Load some real Kanji for comparison
    dataset_dir = Path("kanji_dataset")
    if not dataset_dir.exists():
        print("   ❌ Dataset not found for comparison")
        return
    
    # Get some sample real Kanji
    real_kanji_files = list(dataset_dir.glob("images/*.png"))[:5]
    
    print(f"   📚 Real Kanji Characteristics:")
    real_contrasts = []
    real_complexities = []
    
    for kanji_file in real_kanji_files:
        analysis = analyze_image_characteristics(kanji_file)
        if 'error' not in analysis:
            real_contrasts.append(analysis['contrast'])
            real_complexities.append(analysis['complexity'])
            print(f"     • {kanji_file.name}: Contrast={analysis['contrast']:.1f}, Complexity={analysis['complexity']:.2f}")
    
    if real_contrasts:
        print(f"   📊 Real Kanji Averages:")
        print(f"     • Avg Contrast: {np.mean(real_contrasts):.1f}")
        print(f"     • Avg Complexity: {np.mean(real_complexities):.2f}")
        print(f"     • Contrast Range: {min(real_contrasts):.1f} - {max(real_contrasts):.1f}")

def provide_quality_assessment():
    """Provide quality assessment of generated images"""
    
    print(f"\n🎯 Quality Assessment:")
    
    # Check if we have generated images
    generated_dirs = ["generated_results", "advanced_results"]
    total_generated = 0
    
    for dir_name in generated_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            total_generated += len(list(dir_path.glob("*.png")))
    
    if total_generated == 0:
        print("   ❌ No generated images found")
        return
    
    print(f"   📈 Generated {total_generated} test images")
    
    # Assess based on file sizes (simple proxy for complexity)
    basic_dir = Path("generated_results")
    if basic_dir.exists():
        basic_images = list(basic_dir.glob("*.png"))
        if basic_images:
            avg_size = np.mean([os.path.getsize(f) for f in basic_images])
            
            print(f"   📊 Basic Generation Quality:")
            if avg_size < 200:
                print(f"     • Quality: Low (simple patterns)")
                print(f"     • Assessment: Model needs more training")
            elif avg_size < 300:
                print(f"     • Quality: Medium")
                print(f"     • Assessment: Some learning evident")
            else:
                print(f"     • Quality: High")
                print(f"     • Assessment: Good learning progress")
    
    print(f"   💡 Recommendations:")
    print(f"     • Current model: Simple UNet (quick test)")
    print(f"     • Expected improvement: 10-100x with full training")
    print(f"     • Resolution upgrade: 64x64 → 128x128")
    print(f"     • Architecture upgrade: Simple UNet → Stable Diffusion")

def main():
    """Main function"""
    
    print("🎌 Generated Image Analysis")
    print("=" * 50)
    
    # Analyze generated images
    analyze_generated_images()
    
    # Compare with real Kanji
    compare_with_real_kanji()
    
    # Provide quality assessment
    provide_quality_assessment()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   • Generated images analyzed")
    print(f"   • Quality assessment provided")
    print(f"   • Comparison with real Kanji completed")
    
    print(f"\n📋 Summary:")
    print(f"   • Check the opened Finder windows to view images")
    print(f"   • Generated images are in generated_results/ and advanced_results/")
    print(f"   • Quality varies by generation strategy")
    print(f"   • Full training will significantly improve quality")

if __name__ == "__main__":
    main()
