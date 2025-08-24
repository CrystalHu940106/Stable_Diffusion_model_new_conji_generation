#!/usr/bin/env python3
"""
Display and compare generated Kanji images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def display_generated_kanji():
    """Display all generated Kanji images in a grid"""
    
    # Concepts and guidance scales
    concepts = ["water", "future"]
    guidance_scales = [7.0, 9.0, 11.0]
    
    # Create figure
    fig, axes = plt.subplots(len(concepts), len(guidance_scales), figsize=(15, 10))
    fig.suptitle('🎌 Generated Kanji for "Water" and "Future" Concepts', fontsize=16, fontweight='bold')
    
    # Load and display images
    for i, concept in enumerate(concepts):
        for j, guidance in enumerate(guidance_scales):
            # Load image
            filename = f"kanji_{concept}_{guidance}.png"
            try:
                img = Image.open(filename)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'{concept.upper()} (Guidance: {guidance})', fontsize=12)
                axes[i, j].axis('off')
                
                # Add border
                rect = patches.Rectangle((0, 0), img.width, img.height, 
                                       linewidth=2, edgecolor='blue', facecolor='none')
                axes[i, j].add_patch(rect)
                
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f'Error loading\n{filename}', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'Error: {concept} (Guidance: {guidance})')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print file information
    print("📊 Generated Kanji Files:")
    print("=" * 40)
    for concept in concepts:
        print(f"\n🌊 {concept.upper()}:")
        for guidance in guidance_scales:
            filename = f"kanji_{concept}_{guidance}.png"
            try:
                img = Image.open(filename)
                print(f"   • Guidance {guidance}: {filename} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                print(f"   • Guidance {guidance}: {filename} (Error: {e})")

if __name__ == "__main__":
    display_generated_kanji()
