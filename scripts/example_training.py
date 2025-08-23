#!/usr/bin/env python3
"""
Example: Using Kanji Dataset for Stable Diffusion Training

This script demonstrates how to load and prepare the Kanji dataset
for training a stable diffusion model.
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class KanjiDataset(Dataset):
    """Custom Dataset for Kanji characters"""
    
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load dataset metadata
        metadata_path = self.dataset_path / "metadata" / "dataset.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} Kanji entries")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Load image
        image_path = self.dataset_path / "images" / entry['image_file']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image and prompt
        return {
            'image': image,
            'prompt': entry['prompt'],
            'kanji': entry['kanji'],
            'meanings': entry['meanings']
        }

def create_transforms(image_size=128):
    """Create transforms for training - optimized for low resolution"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Normalize to [-1, 1] for stable diffusion
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def create_augmented_transforms(image_size=128):
    """Create transforms with data augmentation - optimized for low resolution"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(5),  # Small rotations
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight brightness/contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length prompts"""
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    kanji_chars = [item['kanji'] for item in batch]
    meanings = [item['meanings'] for item in batch]
    
    return {
        'image': images,
        'prompt': prompts,
        'kanji': kanji_chars,
        'meanings': meanings
    }

def example_training_setup():
    """Example of how to set up training with the dataset"""
    
    # Create dataset
    transform = create_transforms()
    dataset = KanjiDataset("kanji_dataset", transform=transform)
    
    # Create data loader with custom collate function - optimized for CPU training
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Reduced for CPU training
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=custom_collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Example: iterate through a few batches
    print("\n=== Example Batches ===")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Show first 3 batches
            break
            
        images = batch['image']
        prompts = batch['prompt']
        kanji_chars = batch['kanji']
        
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Sample prompts: {prompts[:3]}")
        print(f"  Sample Kanji: {kanji_chars[:3]}")
        print()

def example_validation_split():
    """Example of creating train/validation split"""
    
    # Load full dataset
    transform = create_transforms()
    full_dataset = KanjiDataset("kanji_dataset", transform=transform)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

def example_prompt_engineering():
    """Example of different prompt engineering approaches"""
    
    dataset = KanjiDataset("kanji_dataset")
    
    print("=== Prompt Engineering Examples ===")
    
    # Get a few examples
    examples = [dataset[i] for i in range(5)]
    
    for i, example in enumerate(examples):
        kanji = example['kanji']
        meanings = example['meanings']
        
        print(f"\n{i+1}. Kanji: {kanji}")
        print(f"   Meanings: {meanings}")
        
        # Different prompt formats
        prompts = [
            f"kanji character {kanji}: {', '.join(meanings)}",
            f"japanese kanji {kanji} meaning {meanings[0]}",
            f"black and white kanji character {kanji}",
            f"calligraphy kanji {kanji} with meaning {', '.join(meanings[:2])}",
            f"traditional japanese writing {kanji}"
        ]
        
        for j, prompt in enumerate(prompts):
            print(f"   Prompt {j+1}: {prompt}")

def example_simple_usage():
    """Simple example without DataLoader"""
    
    print("=== Simple Dataset Usage ===")
    
    # Create dataset without transforms for simple inspection
    dataset = KanjiDataset("kanji_dataset")
    
    # Show a few examples
    for i in range(5):
        example = dataset[i]
        print(f"\n{i+1}. Kanji: {example['kanji']}")
        print(f"   Prompt: {example['prompt']}")
        print(f"   Meanings: {example['meanings']}")
        print(f"   Image size: {example['image'].size}")

def main():
    """Main function demonstrating dataset usage"""
    
    print("=== Kanji Dataset Training Example ===\n")
    
    # Check if dataset exists
    if not Path("kanji_dataset").exists():
        print("❌ Dataset not found! Please run process_kanji_data.py first.")
        return
    
    # Example 1: Simple usage
    print("1. Simple Dataset Usage")
    example_simple_usage()
    
    # Example 2: Basic training setup
    print("\n2. Basic Training Setup")
    example_training_setup()
    
    # Example 3: Train/validation split
    print("\n3. Train/Validation Split")
    train_loader, val_loader = example_validation_split()
    
    # Example 4: Prompt engineering
    print("\n4. Prompt Engineering")
    example_prompt_engineering()
    
    print("\n=== Training Recommendations (CPU Optimized) ===")
    print("• Use batch size 2-4 for CPU training")
    print("• Learning rate: 1e-4 to 5e-4 (higher for faster convergence)")
    print("• Image resolution: 128x128 (faster than 256x256)")
    print("• Training epochs: 3-5 epochs should be sufficient")
    print("• Consider data augmentation for better generalization")
    print("• Monitor validation loss to prevent overfitting")
    print("• Use custom collate function for variable-length prompts")
    print("• Dataset size: 6,410 entries is sufficient for small model")

if __name__ == "__main__":
    main() 