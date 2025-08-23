#!/usr/bin/env python3
"""
Check if images in the dataset are all black
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def check_image_color(image_path):
    """Check if an image is black or has content"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Convert to grayscale if RGB
        if len(img_array.shape) == 3:
            gray = img.convert('L')
            gray_array = np.array(gray)
        else:
            gray_array = img_array
        
        # Calculate statistics
        mean_value = np.mean(gray_array)
        std_value = np.std(gray_array)
        min_value = np.min(gray_array)
        max_value = np.max(gray_array)
        
        # Check if image is mostly black
        is_black = mean_value < 10  # Very low mean value
        is_flat = std_value < 5     # Very low standard deviation
        
        return {
            'mean': mean_value,
            'std': std_value,
            'min': min_value,
            'max': max_value,
            'is_black': is_black,
            'is_flat': is_flat,
            'has_content': not (is_black and is_flat)
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_dataset():
    """Analyze the entire dataset"""
    
    print("🔍 Analyzing Kanji Dataset")
    print("=" * 50)
    
    # Check fixed_kanji_dataset
    dataset_path = Path("data/fixed_kanji_dataset/images")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Get all PNG files
    image_files = list(dataset_path.glob("*.png"))
    print(f"📊 Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return
    
    # Sample some images for analysis
    sample_size = min(20, len(image_files))
    sample_files = image_files[:sample_size]
    
    print(f"\n🔬 Analyzing {sample_size} sample images:")
    
    black_count = 0
    flat_count = 0
    content_count = 0
    error_count = 0
    
    all_means = []
    all_stds = []
    
    for i, img_file in enumerate(sample_files):
        print(f"\n   📸 {img_file.name}:")
        analysis = check_image_color(img_file)
        
        if 'error' in analysis:
            print(f"     ❌ Error: {analysis['error']}")
            error_count += 1
        else:
            print(f"     • Mean: {analysis['mean']:.2f}")
            print(f"     • Std: {analysis['std']:.2f}")
            print(f"     • Range: {analysis['min']} - {analysis['max']}")
            print(f"     • Is Black: {'Yes' if analysis['is_black'] else 'No'}")
            print(f"     • Is Flat: {'Yes' if analysis['is_flat'] else 'No'}")
            print(f"     • Has Content: {'Yes' if analysis['has_content'] else 'No'}")
            
            all_means.append(analysis['mean'])
            all_stds.append(analysis['std'])
            
            if analysis['is_black']:
                black_count += 1
            if analysis['is_flat']:
                flat_count += 1
            if analysis['has_content']:
                content_count += 1
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   • Total analyzed: {sample_size}")
    print(f"   • Black images: {black_count} ({black_count/sample_size*100:.1f}%)")
    print(f"   • Flat images: {flat_count} ({flat_count/sample_size*100:.1f}%)")
    print(f"   • Images with content: {content_count} ({content_count/sample_size*100:.1f}%)")
    print(f"   • Errors: {error_count}")
    
    if all_means:
        print(f"   • Average mean: {np.mean(all_means):.2f}")
        print(f"   • Average std: {np.mean(all_stds):.2f}")
        print(f"   • Mean range: {min(all_means):.2f} - {max(all_means):.2f}")
    
    # Check if all images are black
    if black_count == sample_size:
        print(f"\n⚠️  WARNING: All sampled images appear to be black!")
        print(f"   This suggests a problem with the dataset processing.")
    elif black_count > sample_size * 0.8:
        print(f"\n⚠️  WARNING: Most images ({black_count/sample_size*100:.1f}%) appear to be black!")
        print(f"   This suggests a problem with the dataset processing.")
    else:
        print(f"\n✅ Dataset appears to have normal content.")
    
    # Show some example images
    print(f"\n🖼️  Displaying first 5 images:")
    for i, img_file in enumerate(sample_files[:5]):
        try:
            img = Image.open(img_file)
            print(f"   {i+1}. {img_file.name}: {img.size}, {img.mode}")
            
            # Save a copy for viewing
            output_dir = Path("debug_images")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"sample_{i+1}_{img_file.name}"
            img.save(output_path)
            print(f"      Saved to: {output_path}")
            
        except Exception as e:
            print(f"   {i+1}. {img_file.name}: Error - {e}")

def main():
    """Main function"""
    analyze_dataset()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   • Check debug_images/ folder for sample images")
    print(f"   • If images are black, there may be a processing issue")

if __name__ == "__main__":
    main()
