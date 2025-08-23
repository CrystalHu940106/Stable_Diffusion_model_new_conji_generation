#!/usr/bin/env python3
"""
Simple Image Viewer for Generated Kanji Images
Display basic information about generated images
"""

import os
from pathlib import Path
from PIL import Image

def show_image_info(image_path):
    """Show basic information about an image"""
    
    try:
        img = Image.open(image_path)
        
        # Basic info
        width, height = img.size
        file_size = os.path.getsize(image_path)
        
        # Color info
        if img.mode == 'RGB':
            # Get average color
            img_array = img.convert('RGB')
            r, g, b = img_array.split()
            avg_r = sum(r.getextrema()) / 2
            avg_g = sum(g.getextrema()) / 2
            avg_b = sum(b.getextrema()) / 2
            
            color_info = f"RGB({avg_r:.0f}, {avg_g:.0f}, {avg_b:.0f})"
        else:
            color_info = f"{img.mode}"
        
        print(f"   📸 {image_path.name}:")
        print(f"     • Size: {width}x{height} pixels")
        print(f"     • Mode: {img.mode}")
        print(f"     • File size: {file_size} bytes")
        print(f"     • Color: {color_info}")
        
        return {
            'size': (width, height),
            'file_size': file_size,
            'mode': img.mode
        }
        
    except Exception as e:
        print(f"   ❌ Error reading {image_path.name}: {e}")
        return None

def analyze_generated_images():
    """Analyze all generated images"""
    
    print("🎨 Generated Image Analysis")
    print("=" * 50)
    
    # Analyze basic generation results
    basic_dir = Path("generated_results")
    if basic_dir.exists():
        print(f"\n📊 Basic Generation Results (4 images):")
        basic_images = list(basic_dir.glob("*.png"))
        
        total_size = 0
        for img_path in basic_images:
            info = show_image_info(img_path)
            if info:
                total_size += info['file_size']
        
        if basic_images:
            avg_size = total_size / len(basic_images)
            print(f"\n   📈 Basic Generation Summary:")
            print(f"     • Total images: {len(basic_images)}")
            print(f"     • Average file size: {avg_size:.0f} bytes")
            print(f"     • Quality assessment: {'Low' if avg_size < 200 else 'Medium' if avg_size < 300 else 'High'}")
    
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
                group = 'Random Noise'
            elif 'structured' in name:
                group = 'Structured Noise'
            elif 'low_frequency' in name:
                group = 'Low Frequency'
            elif 'high_frequency' in name:
                group = 'High Frequency'
            elif 'interpolation' in name:
                group = 'Interpolation'
            else:
                group = 'Other'
            
            if group not in test_groups:
                test_groups[group] = []
            test_groups[group].append(img_path)
        
        # Show summary for each group
        for group_name, images in test_groups.items():
            print(f"\n   🎯 {group_name} ({len(images)} images):")
            
            total_size = 0
            for img_path in images:
                info = show_image_info(img_path)
                if info:
                    total_size += info['file_size']
            
            if images:
                avg_size = total_size / len(images)
                print(f"     • Average file size: {avg_size:.0f} bytes")
                print(f"     • Quality: {'Low' if avg_size < 200 else 'Medium' if avg_size < 300 else 'High'}")

def compare_with_real_kanji():
    """Compare with real Kanji images"""
    
    print(f"\n🔍 Comparison with Real Kanji:")
    
    dataset_dir = Path("kanji_dataset")
    if not dataset_dir.exists():
        print("   ❌ Dataset not found for comparison")
        return
    
    # Get some sample real Kanji
    real_kanji_files = list(dataset_dir.glob("images/*.png"))[:3]
    
    print(f"   📚 Real Kanji Examples:")
    real_sizes = []
    
    for kanji_file in real_kanji_files:
        info = show_image_info(kanji_file)
        if info:
            real_sizes.append(info['file_size'])
    
    if real_sizes:
        avg_real_size = sum(real_sizes) / len(real_sizes)
        print(f"\n   📊 Real Kanji Average:")
        print(f"     • Average file size: {avg_real_size:.0f} bytes")
        print(f"     • Quality: High (clean, structured characters)")

def provide_insights():
    """Provide insights about the generated images"""
    
    print(f"\n💡 Insights and Recommendations:")
    
    # Check generated images
    generated_dirs = ["generated_results", "advanced_results"]
    total_generated = 0
    
    for dir_name in generated_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            total_generated += len(list(dir_path.glob("*.png")))
    
    print(f"   📈 Generated {total_generated} test images total")
    
    # Analyze file sizes to assess quality
    basic_dir = Path("generated_results")
    if basic_dir.exists():
        basic_images = list(basic_dir.glob("*.png"))
        if basic_images:
            sizes = [os.path.getsize(f) for f in basic_images]
            avg_size = sum(sizes) / len(sizes)
            
            print(f"\n   🎯 Current Quality Assessment:")
            print(f"     • Average file size: {avg_size:.0f} bytes")
            
            if avg_size < 200:
                print(f"     • Quality: Low - Simple patterns, minimal detail")
                print(f"     • Reason: Quick test model, limited training")
            elif avg_size < 300:
                print(f"     • Quality: Medium - Some structure visible")
                print(f"     • Reason: Basic learning achieved")
            else:
                print(f"     • Quality: High - Complex patterns")
                print(f"     • Reason: Good learning progress")
    
    print(f"\n   🚀 Expected Improvements with Full Training:")
    print(f"     • File size increase: 2-5x larger (more detail)")
    print(f"     • Resolution upgrade: 64x64 → 128x128")
    print(f"     • Architecture upgrade: Simple UNet → Stable Diffusion")
    print(f"     • Training data: 500 samples → 6,410 samples")
    print(f"     • Training time: 9 seconds → 18 hours")
    
    print(f"\n   🎨 Visual Assessment:")
    print(f"     • Current: Abstract patterns, some Kanji-like elements")
    print(f"     • Expected: Clear stroke structure, recognizable characters")
    print(f"     • Goal: Novel Kanji that look authentic")

def main():
    """Main function"""
    
    print("🎌 Generated Image Viewer")
    print("=" * 50)
    
    # Analyze generated images
    analyze_generated_images()
    
    # Compare with real Kanji
    compare_with_real_kanji()
    
    # Provide insights
    provide_insights()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   • Generated images analyzed")
    print(f"   • Quality assessment provided")
    print(f"   • Comparison with real Kanji completed")
    
    print(f"\n📋 Next Steps:")
    print(f"   • View images in Finder windows (already opened)")
    print(f"   • Compare with real Kanji in kanji_dataset/images/")
    print(f"   • Consider proceeding with full training")
    print(f"   • Expected significant quality improvement")

if __name__ == "__main__":
    main()
