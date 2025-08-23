#!/usr/bin/env python3
"""
View Original Kanji Samples
Show some original Kanji characters from the dataset
"""

import json
import os
from pathlib import Path
from PIL import Image

def show_kanji_samples():
    """Show some sample Kanji characters"""
    
    print("🎌 Original Kanji Samples")
    print("=" * 50)
    
    # Load dataset
    dataset_path = Path("kanji_dataset/metadata/dataset.json")
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"📊 Dataset Overview:")
    print(f"   • Total Kanji: {len(dataset):,}")
    print(f"   • Images folder: kanji_dataset/images/")
    print(f"   • Metadata folder: kanji_dataset/metadata/")
    
    # Show some common Kanji
    common_kanji = ['人', '大', '小', '山', '川', '日', '月', '火', '水', '木', '金', '土', '天', '地', '中', '国']
    
    print(f"\n📚 Common Kanji Samples:")
    found_kanji = []
    
    for kanji in common_kanji:
        for entry in dataset:
            if entry['kanji'] == kanji:
                found_kanji.append(entry)
                break
    
    for i, entry in enumerate(found_kanji[:10]):
        print(f"   {i+1:2d}. {entry['kanji']} (U+{entry['unicode']}): {', '.join(entry['meanings'][:3])}")
        print(f"       • Image: kanji_dataset/images/{entry['image_file']}")
        print(f"       • Metadata: kanji_dataset/metadata/{entry['unicode']}.json")
    
    # Show some success-related Kanji
    print(f"\n🎯 Success-Related Kanji:")
    success_keywords = ['success', 'achieve', 'complete', 'finish', 'win', 'victory']
    success_kanji = []
    
    for entry in dataset:
        meanings = [m.lower() for m in entry['meanings']]
        if any(keyword in meanings for keyword in success_keywords):
            success_kanji.append(entry)
            if len(success_kanji) >= 10:
                break
    
    for i, entry in enumerate(success_kanji):
        print(f"   {i+1:2d}. {entry['kanji']} (U+{entry['unicode']}): {', '.join(entry['meanings'][:3])}")
    
    # Show random samples
    print(f"\n🎲 Random Kanji Samples:")
    import random
    random_samples = random.sample(dataset, 10)
    
    for i, entry in enumerate(random_samples):
        print(f"   {i+1:2d}. {entry['kanji']} (U+{entry['unicode']}): {', '.join(entry['meanings'][:2])}")

def analyze_image_quality():
    """Analyze the quality of original Kanji images"""
    
    print(f"\n🔍 Image Quality Analysis:")
    
    # Check a few sample images
    sample_files = [
        "4e85.png",  # 人 (person)
        "5927.png",  # 大 (large)
        "5c0f.png",  # 小 (small)
        "5c71.png",  # 山 (mountain)
        "65e5.png",  # 日 (sun)
    ]
    
    images_dir = Path("kanji_dataset/images")
    
    for filename in sample_files:
        file_path = images_dir / filename
        if file_path.exists():
            try:
                img = Image.open(file_path)
                file_size = os.path.getsize(file_path)
                
                print(f"   📸 {filename}:")
                print(f"     • Size: {img.size}")
                print(f"     • Mode: {img.mode}")
                print(f"     • File size: {file_size} bytes")
                
                # Check if it's black and white
                if img.mode == 'RGB':
                    img_array = img.convert('RGB')
                    r, g, b = img_array.split()
                    r_extrema = r.getextrema()
                    g_extrema = g.getextrema()
                    b_extrema = b.getextrema()
                    
                    print(f"     • Color range: R{r_extrema}, G{g_extrema}, B{b_extrema}")
                    
                    # Check if it's pure black and white
                    if r_extrema == (0, 255) and g_extrema == (0, 255) and b_extrema == (0, 255):
                        print(f"     • Quality: ✅ Pure black and white")
                    elif r_extrema == (255, 255) and g_extrema == (255, 255) and b_extrema == (255, 255):
                        print(f"     • Quality: ⚠️ All white (problematic)")
                    else:
                        print(f"     • Quality: ⚠️ Mixed colors")
                
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
        else:
            print(f"   ❌ File not found: {filename}")

def show_dataset_statistics():
    """Show dataset statistics"""
    
    print(f"\n📊 Dataset Statistics:")
    
    # Count files
    images_dir = Path("kanji_dataset/images")
    metadata_dir = Path("kanji_dataset/metadata")
    
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        print(f"   • Image files: {len(image_files):,}")
    
    if metadata_dir.exists():
        json_files = list(metadata_dir.glob("*.json"))
        print(f"   • Metadata files: {len(json_files):,}")
    
    # Load dataset for more stats
    dataset_path = Path("kanji_dataset/metadata/dataset.json")
    if dataset_path.exists():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Count meanings
        total_meanings = sum(len(entry['meanings']) for entry in dataset)
        avg_meanings = total_meanings / len(dataset)
        
        print(f"   • Total meanings: {total_meanings:,}")
        print(f"   • Average meanings per Kanji: {avg_meanings:.1f}")
        
        # Unicode range
        unicode_values = [int(entry['unicode'], 16) for entry in dataset]
        min_unicode = min(unicode_values)
        max_unicode = max(unicode_values)
        
        print(f"   • Unicode range: U+{min_unicode:04X} to U+{max_unicode:04X}")

def main():
    """Main function"""
    
    print("🎌 Original Kanji Dataset Viewer")
    print("=" * 50)
    
    # Show Kanji samples
    show_kanji_samples()
    
    # Analyze image quality
    analyze_image_quality()
    
    # Show statistics
    show_dataset_statistics()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   • Original Kanji samples shown")
    print(f"   • Image quality analyzed")
    print(f"   • Dataset statistics provided")
    
    print(f"\n📋 Summary:")
    print(f"   • Original Kanji are in kanji_dataset/images/")
    print(f"   • Metadata is in kanji_dataset/metadata/")
    print(f"   • Total: 6,410 Kanji characters")
    print(f"   • Format: 64x64 PNG, black strokes on white background")

if __name__ == "__main__":
    main()
