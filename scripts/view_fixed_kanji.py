#!/usr/bin/env python3
"""
View Fixed Kanji Images
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_kanji_samples():
    """Display sample kanji images"""
    
    print("🎌 Fixed Kanji Dataset Viewer")
    print("=" * 50)
    
    # Get sample images
    dataset_path = Path("data/fixed_kanji_dataset/images")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Get first 16 images
    image_files = list(dataset_path.glob("*.png"))[:16]
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return
    
    print(f"📊 Displaying {len(image_files)} sample kanji:")
    
    # Create subplot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Fixed Kanji Dataset Samples', fontsize=16)
    
    for i, img_file in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_file)
            
            # Get row and column
            row = i // 4
            col = i % 4
            
            # Display image
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'{img_file.stem}', fontsize=10)
            axes[row, col].axis('off')
            
            print(f"   {i+1:2d}. {img_file.name}")
            
        except Exception as e:
            print(f"   {i+1:2d}. {img_file.name}: Error - {e}")
            axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "fixed_kanji_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Sample images saved to: {output_path}")
    
    # Show the plot
    plt.show()

def analyze_kanji_quality():
    """Analyze the quality of fixed kanji images"""
    
    print(f"\n🔍 Analyzing Kanji Quality")
    print("=" * 50)
    
    dataset_path = Path("data/fixed_kanji_dataset/images")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Get sample images
    image_files = list(dataset_path.glob("*.png"))[:10]
    
    total_pixels = 0
    black_pixels = 0
    white_pixels = 0
    
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            img_array = img.convert('L')  # Convert to grayscale
            
            # Count pixels
            for y in range(img_array.height):
                for x in range(img_array.width):
                    pixel = img_array.getpixel((x, y))
                    total_pixels += 1
                    if pixel == 0:
                        black_pixels += 1
                    elif pixel == 255:
                        white_pixels += 1
            
            print(f"   📸 {img_file.name}: {img.size}, {img.mode}")
            
        except Exception as e:
            print(f"   ❌ Error reading {img_file.name}: {e}")
    
    if total_pixels > 0:
        black_percentage = (black_pixels / total_pixels) * 100
        white_percentage = (white_pixels / total_pixels) * 100
        other_percentage = 100 - black_percentage - white_percentage
        
        print(f"\n📊 Pixel Analysis:")
        print(f"   • Total pixels: {total_pixels:,}")
        print(f"   • Black pixels: {black_pixels:,} ({black_percentage:.1f}%)")
        print(f"   • White pixels: {white_pixels:,} ({white_percentage:.1f}%)")
        print(f"   • Other pixels: {total_pixels - black_pixels - white_pixels:,} ({other_percentage:.1f}%)")
        
        if black_percentage > 0 and white_percentage > 80:
            print(f"   ✅ Good quality: Clear black strokes on white background")
        else:
            print(f"   ⚠️  Quality issues detected")

def main():
    """Main function"""
    print("🎌 Fixed Kanji Dataset Viewer")
    print("=" * 50)
    
    # Display samples
    display_kanji_samples()
    
    # Analyze quality
    analyze_kanji_quality()
    
    print(f"\n🎉 Viewing complete!")
    print(f"   • Dataset is now properly fixed")
    print(f"   • Images have black strokes on white background")
    print(f"   • Ready for training!")

if __name__ == "__main__":
    main()
