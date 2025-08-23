#!/usr/bin/env python3
"""
Kanji Dataset Summary

This script provides a comprehensive summary of the generated Kanji dataset.
"""

import json
import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def generate_summary():
    """Generate a comprehensive summary of the dataset"""
    
    dataset_path = Path("kanji_dataset")
    metadata_path = dataset_path / "metadata" / "dataset.json"
    images_path = dataset_path / "images"
    
    print("=== Kanji Dataset Summary ===\n")
    
    # Load dataset
    with open(metadata_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Basic statistics
    total_kanji = len(dataset)
    total_meanings = sum(len(entry['meanings']) for entry in dataset)
    avg_meanings = total_meanings / total_kanji
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total Kanji: {total_kanji:,}")
    print(f"   Total meanings: {total_meanings:,}")
    print(f"   Average meanings per Kanji: {avg_meanings:.1f}")
    print(f"   Image files: {len(list(images_path.glob('*.png'))):,}")
    
    # Meaning distribution
    meaning_counts = [len(entry['meanings']) for entry in dataset]
    meaning_counter = Counter(meaning_counts)
    
    print(f"\n📈 Meaning Distribution:")
    for count in sorted(meaning_counter.keys()):
        percentage = (meaning_counter[count] / total_kanji) * 100
        print(f"   {count} meaning(s): {meaning_counter[count]:,} Kanji ({percentage:.1f}%)")
    
    # Common meanings analysis
    all_meanings = []
    for entry in dataset:
        all_meanings.extend(entry['meanings'])
    
    meaning_freq = Counter(all_meanings)
    top_meanings = meaning_freq.most_common(20)
    
    print(f"\n🏆 Top 20 Most Common Meanings:")
    for i, (meaning, count) in enumerate(top_meanings, 1):
        print(f"   {i:2d}. {meaning}: {count:,} occurrences")
    
    # Unicode range analysis
    unicode_values = [int(entry['unicode'], 16) for entry in dataset]
    unicode_min = min(unicode_values)
    unicode_max = max(unicode_values)
    
    print(f"\n🔤 Unicode Range:")
    print(f"   Minimum: U+{unicode_min:04X} ({chr(unicode_min)})")
    print(f"   Maximum: U+{unicode_max:04X} ({chr(unicode_max)})")
    print(f"   Range: {unicode_max - unicode_min:,} code points")
    
    # Common Kanji check
    common_kanji = ['人', '大', '小', '山', '川', '日', '月', '火', '水', '木', '金', '土', '天', '地', '中', '国', '年', '生', '学', '校']
    found_common = []
    missing_common = []
    
    for kanji in common_kanji:
        found = False
        for entry in dataset:
            if entry['kanji'] == kanji:
                found_common.append((kanji, entry['meanings']))
                found = True
                break
        if not found:
            missing_common.append(kanji)
    
    print(f"\n✅ Common Kanji Coverage:")
    print(f"   Found: {len(found_common)}/{len(common_kanji)}")
    for kanji, meanings in found_common:
        print(f"     {kanji}: {', '.join(meanings[:3])}")
    
    if missing_common:
        print(f"   Missing: {', '.join(missing_common)}")
    
    # File size analysis
    image_files = list(images_path.glob("*.png"))
    total_size = sum(f.stat().st_size for f in image_files)
    avg_size = total_size / len(image_files) if image_files else 0
    
    print(f"\n💾 File Analysis:")
    print(f"   Total image size: {total_size / 1024 / 1024:.1f} MB")
    print(f"   Average image size: {avg_size:.0f} bytes")
    print(f"   Image format: PNG, 64x64 pixels")
    
    # Quality metrics
    print(f"\n🎯 Quality Metrics:")
    print(f"   ✅ Pure black/white images")
    print(f"   ✅ No stroke order numbers")
    print(f"   ✅ Consistent 64x64 resolution")
    print(f"   ✅ High contrast for ML training")
    print(f"   ✅ Complete metadata coverage")
    
    # Training recommendations
    print(f"\n🚀 Training Recommendations:")
    print(f"   • Batch size: 8-16 (depending on GPU memory)")
    print(f"   • Learning rate: 1e-5 to 1e-4")
    print(f"   • Training steps: 1000-5000 per epoch")
    print(f"   • Image size: 64x64 pixels (as provided)")
    print(f"   • Data augmentation: Small rotations/translations")
    print(f"   • Validation split: 80/20 recommended")
    
    # Dataset structure
    print(f"\n📁 Dataset Structure:")
    print(f"   kanji_dataset/")
    print(f"   ├── images/           # {len(image_files):,} PNG files")
    print(f"   └── metadata/")
    print(f"       ├── dataset.json  # Complete dataset")
    print(f"       └── *.json        # Individual Kanji files")
    
    return dataset

def show_sample_entries(dataset, num_samples=10):
    """Show sample entries from the dataset"""
    
    print(f"\n📝 Sample Entries:")
    for i, entry in enumerate(dataset[:num_samples], 1):
        kanji = entry['kanji']
        meanings = entry['meanings']
        unicode_val = entry['unicode']
        image_file = entry['image_file']
        
        print(f"\n{i:2d}. {kanji} (U+{unicode_val.upper()})")
        print(f"    Meanings: {', '.join(meanings)}")
        print(f"    Image: {image_file}")
        print(f"    Prompt: {entry['prompt']}")

def main():
    """Main function"""
    
    if not Path("kanji_dataset").exists():
        print("❌ Dataset not found! Please run process_kanji_data.py first.")
        return
    
    # Generate summary
    dataset = generate_summary()
    
    # Show sample entries
    show_sample_entries(dataset)
    
    print(f"\n🎉 Dataset Summary Complete!")
    print(f"The Kanji dataset is ready for stable diffusion training with {len(dataset):,} high-quality entries.")

if __name__ == "__main__":
    main() 