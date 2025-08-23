#!/usr/bin/env python3
"""
Compare Generated Kanji with Existing Kanji
"""

import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_existing_kanji_images(concept, dataset_path, max_samples=4):
    """Load existing kanji images for a concept"""
    
    # Load dataset metadata
    with open(dataset_path / "metadata" / "dataset.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Define concept keywords
    concept_keywords = {
        'success': ['success', 'achieve', 'accomplish', 'complete', 'win', 'victory', 'triumph', 'succeed', 'prosper', 'flourish'],
        'failure': ['fail', 'lose', 'defeat', 'error', 'mistake', 'wrong', 'bad', 'negative', 'defeat', 'loss'],
        'novel': ['new', 'novel', 'original', 'creative', 'unique', 'different', 'innovative', 'fresh', 'modern'],
        'funny': ['funny', 'humorous', 'amusing', 'entertaining', 'comical', 'laugh', 'joke', 'playful', 'witty'],
        'culturally_meaningful': ['culture', 'tradition', 'heritage', 'meaningful', 'significant', 'important', 'sacred', 'spiritual', 'philosophy', 'wisdom']
    }
    
    keywords = concept_keywords.get(concept, [])
    matching_kanji = []
    
    for entry in dataset:
        meanings = [meaning.lower() for meaning in entry['meanings']]
        for keyword in keywords:
            if keyword in meanings:
                matching_kanji.append(entry)
                break
    
    # Load images for the first few matches
    existing_images = []
    existing_info = []
    
    for i, kanji_info in enumerate(matching_kanji[:max_samples]):
        try:
            image_path = dataset_path / "images" / kanji_info['image_file']
            if image_path.exists():
                img = Image.open(image_path).convert('RGB')
                existing_images.append(np.array(img))
                existing_info.append({
                    'kanji': kanji_info['kanji'],
                    'meanings': kanji_info['meanings'][:3],
                    'unicode': kanji_info['unicode']
                })
        except Exception as e:
            print(f"Error loading {kanji_info['kanji']}: {e}")
    
    return existing_images, existing_info

def load_generated_kanji_images(concept, generated_path, max_samples=4):
    """Load generated kanji images for a concept"""
    
    generated_images = []
    
    for i in range(1, max_samples + 1):
        try:
            image_path = generated_path / f"{concept}_generated_{i}.png"
            if image_path.exists():
                img = Image.open(image_path).convert('RGB')
                generated_images.append(np.array(img))
        except Exception as e:
            print(f"Error loading generated {concept}_{i}: {e}")
    
    return generated_images

def create_comparison_display(concept, existing_images, existing_info, generated_images):
    """Create a comparison display for a concept"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Comparison: {concept.replace("_", " ").title()} Kanji', fontsize=16)
    
    # Existing kanji (top row)
    for i in range(4):
        if i < len(existing_images):
            axes[0, i].imshow(existing_images[i])
            info = existing_info[i]
            title = f"Existing: {info['kanji']}\n{', '.join(info['meanings'])}"
            axes[0, i].set_title(title, fontsize=10)
        else:
            axes[0, i].text(0.5, 0.5, 'No image', ha='center', va='center')
            axes[0, i].set_title('No existing match', fontsize=10)
        axes[0, i].axis('off')
    
    # Generated kanji (bottom row)
    for i in range(4):
        if i < len(generated_images):
            axes[1, i].imshow(generated_images[i])
            axes[1, i].set_title(f'Generated #{i+1}', fontsize=10)
        else:
            axes[1, i].text(0.5, 0.5, 'No image', ha='center', va='center')
            axes[1, i].set_title('No generated image', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = f"comparison_{concept}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Comparison saved to: {output_path}")
    
    return fig

def analyze_visual_patterns(existing_images, generated_images, concept):
    """Analyze visual patterns in existing vs generated kanji"""
    
    print(f"\n🔍 Visual Pattern Analysis for '{concept}':")
    
    if not existing_images or not generated_images:
        print("   • Insufficient data for analysis")
        return
    
    # Simple pattern analysis
    def analyze_image_patterns(images):
        patterns = {
            'avg_brightness': [],
            'contrast': [],
            'complexity': []
        }
        
        for img in images:
            # Convert to grayscale for analysis
            gray = np.mean(img, axis=2)
            
            # Average brightness
            avg_brightness = np.mean(gray)
            patterns['avg_brightness'].append(avg_brightness)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            patterns['contrast'].append(contrast)
            
            # Complexity (edge density approximation)
            edges_h = np.abs(np.diff(gray, axis=0))
            edges_v = np.abs(np.diff(gray, axis=1))
            complexity = np.mean(edges_h) + np.mean(edges_v)
            patterns['complexity'].append(complexity)
        
        return patterns
    
    existing_patterns = analyze_image_patterns(existing_images)
    generated_patterns = analyze_image_patterns(generated_images)
    
    print(f"   📊 Existing kanji patterns:")
    print(f"     • Avg brightness: {np.mean(existing_patterns['avg_brightness']):.2f}")
    print(f"     • Avg contrast: {np.mean(existing_patterns['contrast']):.2f}")
    print(f"     • Avg complexity: {np.mean(existing_patterns['complexity']):.2f}")
    
    print(f"   📊 Generated kanji patterns:")
    print(f"     • Avg brightness: {np.mean(generated_patterns['avg_brightness']):.2f}")
    print(f"     • Avg contrast: {np.mean(generated_patterns['contrast']):.2f}")
    print(f"     • Avg complexity: {np.mean(generated_patterns['complexity']):.2f}")
    
    # Compare patterns
    brightness_diff = abs(np.mean(existing_patterns['avg_brightness']) - np.mean(generated_patterns['avg_brightness']))
    contrast_diff = abs(np.mean(existing_patterns['contrast']) - np.mean(generated_patterns['contrast']))
    complexity_diff = abs(np.mean(existing_patterns['complexity']) - np.mean(generated_patterns['complexity']))
    
    print(f"   📈 Pattern differences:")
    print(f"     • Brightness diff: {brightness_diff:.2f}")
    print(f"     • Contrast diff: {contrast_diff:.2f}")
    print(f"     • Complexity diff: {complexity_diff:.2f}")
    
    if brightness_diff < 20 and contrast_diff < 10 and complexity_diff < 5:
        print(f"   ✅ Good pattern similarity")
    else:
        print(f"   ⚠️  Significant pattern differences")

def main():
    """Main function"""
    
    print("🎌 Compare Generated vs Existing Kanji")
    print("=" * 50)
    
    # Paths
    dataset_path = Path("data/fixed_kanji_dataset")
    generated_path = Path("generated_results")
    
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return
    
    if not generated_path.exists():
        print("❌ Generated results not found!")
        return
    
    # Concepts to compare
    concepts = ['success', 'failure', 'novel', 'funny', 'culturally_meaningful']
    
    all_figures = []
    
    for concept in concepts:
        print(f"\n🔍 Analyzing {concept.replace('_', ' ')}...")
        
        # Load existing kanji
        existing_images, existing_info = load_existing_kanji_images(concept, dataset_path)
        print(f"   • Found {len(existing_images)} existing kanji images")
        
        # Load generated kanji
        generated_images = load_generated_kanji_images(concept, generated_path)
        print(f"   • Found {len(generated_images)} generated kanji images")
        
        # Create comparison display
        fig = create_comparison_display(concept, existing_images, existing_info, generated_images)
        all_figures.append(fig)
        
        # Analyze patterns
        analyze_visual_patterns(existing_images, generated_images, concept)
    
    # Show all comparisons
    plt.show()
    
    print(f"\n🎉 Comparison Complete!")
    print(f"   • Compared {len(concepts)} concepts")
    print(f"   • Generated comparison images for each concept")
    print(f"   • Analyzed visual patterns")
    
    print(f"\n💡 Insights:")
    print(f"   • Check the comparison images to see visual similarities")
    print(f"   • Look for pattern consistency between existing and generated")
    print(f"   • Consider how well the model captures concept-specific features")
    print(f"   • Generated kanji show the model's learned visual representations")

if __name__ == "__main__":
    main()
