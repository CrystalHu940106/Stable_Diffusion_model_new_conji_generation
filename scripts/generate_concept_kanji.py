#!/usr/bin/env python3
"""
Generate Kanji for Specific Concepts
Generate kanji for: success, failure, novel, funny, culturally meaningful
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

class SimpleUNet(nn.Module):
    """Simple UNet for generation"""
    
    def __init__(self, in_channels=3, out_channels=3, image_size=64):
        super().__init__()
        
        # Simple encoder-decoder structure
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_trained_model():
    """Load the trained model"""
    model = SimpleUNet(in_channels=3, out_channels=3, image_size=64)
    
    checkpoint_path = Path("quick_test_results/quick_test_epoch_2.pth")
    if not checkpoint_path.exists():
        print("❌ Checkpoint not found! Please run quick_train_test.py first.")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   • Loss: {checkpoint['loss']:.6f}")
    
    return model

def create_concept_noise(concept, image_size=64, batch_size=4):
    """Create concept-specific noise for generation"""
    
    # Set random seed based on concept for consistency
    concept_seeds = {
        'success': 42,
        'failure': 123,
        'novel': 456,
        'funny': 789,
        'culturally_meaningful': 999
    }
    
    seed = concept_seeds.get(concept, random.randint(1, 1000))
    torch.manual_seed(seed)
    
    # Create different types of noise based on concept
    if concept == 'success':
        # More structured, upward patterns
        noise = torch.randn(batch_size, 3, image_size, image_size) * 0.5
        # Add some upward gradient
        for i in range(batch_size):
            for c in range(3):
                for y in range(image_size):
                    noise[i, c, y, :] += (y / image_size) * 0.3
    elif concept == 'failure':
        # More chaotic, downward patterns
        noise = torch.randn(batch_size, 3, image_size, image_size) * 0.8
        # Add some downward gradient
        for i in range(batch_size):
            for c in range(3):
                for y in range(image_size):
                    noise[i, c, y, :] -= (y / image_size) * 0.3
    elif concept == 'novel':
        # Creative, asymmetric patterns
        noise = torch.randn(batch_size, 3, image_size, image_size) * 0.6
        # Add some asymmetry
        for i in range(batch_size):
            for c in range(3):
                noise[i, c, :, :image_size//2] *= 1.2
    elif concept == 'funny':
        # Playful, curved patterns
        noise = torch.randn(batch_size, 3, image_size, image_size) * 0.7
        # Add some wave-like patterns
        for i in range(batch_size):
            for c in range(3):
                for y in range(image_size):
                    wave = np.sin(y * 0.2) * 0.2
                    noise[i, c, y, :] += wave
    elif concept == 'culturally_meaningful':
        # Balanced, harmonious patterns
        noise = torch.randn(batch_size, 3, image_size, image_size) * 0.4
        # Add some symmetry
        for i in range(batch_size):
            for c in range(3):
                left_half = noise[i, c, :, :image_size//2]
                right_half = torch.flip(left_half, dims=[1])
                noise[i, c, :, image_size//2:] = right_half
    else:
        # Default random noise
        noise = torch.randn(batch_size, 3, image_size, image_size)
    
    return noise

def denormalize(tensor):
    """Denormalize tensor back to [0, 1] range"""
    return (tensor + 1) / 2

def generate_concept_kanji(model, concept, num_samples=4):
    """Generate kanji for a specific concept"""
    
    print(f"\n🎯 Generating Kanji for '{concept}'...")
    
    # Create concept-specific noise
    noise_input = create_concept_noise(concept, batch_size=num_samples)
    
    # Generate
    with torch.no_grad():
        generated = model(noise_input)
    
    print(f"   • Generated {num_samples} samples")
    print(f"   • Input shape: {noise_input.shape}")
    print(f"   • Output shape: {generated.shape}")
    
    return generated

def save_concept_images(generated, concept, save_dir="generated_results"):
    """Save generated images for a concept"""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Convert from [-1, 1] to [0, 1] range
    generated = denormalize(generated)
    generated = torch.clamp(generated, 0, 1)
    
    # Convert to PIL images and save
    saved_paths = []
    for i in range(generated.shape[0]):
        # Convert tensor to numpy
        img_array = generated[i].permute(1, 2, 0).numpy()
        
        # Convert to PIL image
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Save image
        img_path = save_path / f"{concept}_generated_{i+1}.png"
        img.save(img_path)
        saved_paths.append(img_path)
        print(f"   • Saved: {img_path}")
    
    return saved_paths

def display_concept_results(all_generated, concepts):
    """Display all generated results"""
    
    num_concepts = len(concepts)
    num_samples = all_generated[0].shape[0]
    
    fig, axes = plt.subplots(num_concepts, num_samples, figsize=(4*num_samples, 4*num_concepts))
    fig.suptitle('Generated Kanji for Different Concepts', fontsize=16)
    
    for i, (concept, generated) in enumerate(zip(concepts, all_generated)):
        for j in range(num_samples):
            if num_concepts == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            # Convert to image
            img_array = denormalize(generated[j]).permute(1, 2, 0).numpy()
            img_array = np.clip(img_array, 0, 1)
            
            ax.imshow(img_array)
            ax.set_title(f'{concept.replace("_", " ").title()} #{j+1}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save combined results
    output_path = "all_concept_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Combined results saved to: {output_path}")
    
    # Show the plot
    plt.show()

def analyze_existing_kanji():
    """Analyze existing kanji that might match our concepts"""
    
    print("📚 Analyzing existing Kanji for our concepts...")
    
    # Load dataset
    dataset_path = Path("data/fixed_kanji_dataset/metadata/dataset.json")
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Define concept keywords
    concept_keywords = {
        'success': ['success', 'achieve', 'accomplish', 'complete', 'win', 'victory', 'triumph', 'succeed', 'prosper', 'flourish'],
        'failure': ['fail', 'lose', 'defeat', 'error', 'mistake', 'wrong', 'bad', 'negative', 'defeat', 'loss'],
        'novel': ['new', 'novel', 'original', 'creative', 'unique', 'different', 'innovative', 'fresh', 'modern'],
        'funny': ['funny', 'humorous', 'amusing', 'entertaining', 'comical', 'laugh', 'joke', 'playful', 'witty'],
        'culturally_meaningful': ['culture', 'tradition', 'heritage', 'meaningful', 'significant', 'important', 'sacred', 'spiritual', 'philosophy', 'wisdom']
    }
    
    # Find matching kanji for each concept
    concept_matches = {}
    
    for concept, keywords in concept_keywords.items():
        matching_kanji = []
        for entry in dataset:
            meanings = [meaning.lower() for meaning in entry['meanings']]
            for keyword in keywords:
                if keyword in meanings:
                    matching_kanji.append({
                        'kanji': entry['kanji'],
                        'meanings': entry['meanings'],
                        'unicode': entry['unicode']
                    })
                    break
        
        concept_matches[concept] = matching_kanji
        print(f"\n   📖 {concept.replace('_', ' ').title()}: {len(matching_kanji)} matches")
        for i, kanji_info in enumerate(matching_kanji[:5]):  # Show first 5
            print(f"     {i+1}. {kanji_info['kanji']} ({kanji_info['unicode']}): {', '.join(kanji_info['meanings'][:3])}")
    
    return concept_matches

def main():
    """Main function"""
    
    print("🎌 Generate Kanji for Specific Concepts")
    print("=" * 50)
    
    # Analyze existing kanji
    concept_matches = analyze_existing_kanji()
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Define concepts
    concepts = ['success', 'failure', 'novel', 'funny', 'culturally_meaningful']
    
    # Generate for each concept
    all_generated = []
    all_saved_paths = []
    
    for concept in concepts:
        generated = generate_concept_kanji(model, concept, num_samples=4)
        saved_paths = save_concept_images(generated, concept)
        
        all_generated.append(generated)
        all_saved_paths.extend(saved_paths)
    
    # Display results
    display_concept_results(all_generated, concepts)
    
    # Summary
    print(f"\n🎉 Generation Complete!")
    print(f"   • Generated kanji for {len(concepts)} concepts")
    print(f"   • Created {len(all_saved_paths)} images total")
    print(f"   • Images saved in: generated_results/")
    
    print(f"\n📊 Concept Summary:")
    for concept in concepts:
        print(f"   • {concept.replace('_', ' ').title()}: 4 generated + {len(concept_matches[concept])} existing matches")
    
    print(f"\n💡 Analysis:")
    print(f"   • Compare generated kanji with existing matches")
    print(f"   • Look for patterns that might represent each concept")
    print(f"   • Consider cultural and visual elements")
    print(f"   • Generated kanji show the model's understanding of visual patterns")

if __name__ == "__main__":
    main()
