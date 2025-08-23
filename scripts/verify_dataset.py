#!/usr/bin/env python3
"""
Verify Kanji Dataset

This script verifies the generated dataset and shows some examples.
"""

import json
import os
from pathlib import Path
from PIL import Image

def verify_dataset():
    """Verify the dataset structure and show examples"""
    
    dataset_path = Path("kanji_dataset")
    metadata_path = dataset_path / "metadata" / "dataset.json"
    images_path = dataset_path / "images"
    
    print("=== Kanji Dataset Verification ===")
    
    # Check if files exist
    if not metadata_path.exists():
        print("❌ Dataset metadata not found!")
        return
    
    if not images_path.exists():
        print("❌ Images directory not found!")
        return
    
    # Load dataset
    with open(metadata_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Dataset loaded: {len(dataset)} entries")
    
    # Check image files
    image_files = list(images_path.glob("*.png"))
    print(f"✅ Images found: {len(image_files)} files")
    
    # Show some examples
    print("\n=== Example Entries ===")
    for i, entry in enumerate(dataset[:10]):
        kanji = entry['kanji']
        meanings = entry['meanings'][:3]  # Show first 3 meanings
        image_file = entry['image_file']
        
        # Check if image exists
        image_path = images_path / image_file
        image_exists = image_path.exists()
        
        print(f"{i+1:2d}. {kanji} ({entry['unicode']}): {', '.join(meanings)}")
        print(f"    Image: {image_file} {'✅' if image_exists else '❌'}")
        
        if image_exists:
            # Get image info
            try:
                img = Image.open(image_path)
                print(f"    Size: {img.size}, Mode: {img.mode}")
                
                # Check if it's black and white
                if img.mode == 'RGB':
                    # Convert to grayscale to check
                    gray = img.convert('L')
                    # Get unique values
                    unique_values = set(gray.getdata())
                    if len(unique_values) <= 2:  # Only black and white
                        print(f"    ✅ Pure black/white image")
                    else:
                        print(f"    ⚠️  Not pure black/white ({len(unique_values)} unique values)")
                
            except Exception as e:
                print(f"    ❌ Error reading image: {e}")
        
        print()
    
    # Statistics
    print("=== Dataset Statistics ===")
    total_meanings = sum(len(entry['meanings']) for entry in dataset)
    avg_meanings = total_meanings / len(dataset)
    print(f"Total Kanji: {len(dataset)}")
    print(f"Total meanings: {total_meanings}")
    print(f"Average meanings per Kanji: {avg_meanings:.1f}")
    
    # Check for common Kanji
    common_kanji = ['人', '大', '小', '山', '川', '日', '月', '火', '水', '木']
    found_common = []
    for kanji in common_kanji:
        for entry in dataset:
            if entry['kanji'] == kanji:
                found_common.append(kanji)
                break
    
    print(f"\nCommon Kanji found: {len(found_common)}/{len(common_kanji)}")
    for kanji in found_common:
        for entry in dataset:
            if entry['kanji'] == kanji:
                print(f"  {kanji}: {', '.join(entry['meanings'][:3])}")
                break

if __name__ == "__main__":
    verify_dataset() 