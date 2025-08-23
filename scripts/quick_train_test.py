#!/usr/bin/env python3
"""
Quick Training Test for Kanji Diffusion Model
Ultra-fast training to validate the pipeline works correctly
"""

import json
import os
import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class QuickKanjiDataset(Dataset):
    """Quick test dataset for Kanji characters"""
    
    def __init__(self, dataset_path, transform=None, use_test_data=True):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load dataset metadata
        if use_test_data:
            metadata_path = self.dataset_path / "metadata" / "test_dataset.json"
        else:
            metadata_path = self.dataset_path / "metadata" / "dataset.json"
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} Kanji entries for quick test")
    
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
        
        return {
            'image': image,
            'prompt': entry['prompt'],
            'kanji': entry['kanji'],
            'meanings': entry['meanings']
        }

def create_quick_transforms(image_size=64):
    """Create transforms for quick training test"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def custom_collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    kanji_chars = [item['kanji'] for item in batch]
    
    return {
        'image': images,
        'prompt': prompts,
        'kanji': kanji_chars,
    }

class SimpleUNet(nn.Module):
    """Simple UNet for quick testing"""
    
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
            nn.Tanh(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def quick_training_test():
    """Run quick training test"""
    
    print("🚀 Starting Quick Training Test")
    print("=" * 40)
    
    # Configuration for quick test
    config = {
        'image_size': 64,
        'batch_size': 8,
        'learning_rate': 5e-4,
        'num_epochs': 2,
        'device': 'cpu',
        'save_dir': 'quick_test_results'
    }
    
    print(f"📊 Configuration:")
    print(f"   • Image size: {config['image_size']}x{config['image_size']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    print(f"   • Epochs: {config['num_epochs']}")
    print(f"   • Device: {config['device']}")
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Create dataset and dataloader
    transform = create_quick_transforms(config['image_size'])
    dataset = QuickKanjiDataset("data/fixed_kanji_dataset", transform=transform, use_test_data=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    print(f"\n📚 Dataset Info:")
    print(f"   • Total samples: {len(dataset)}")
    print(f"   • Batches per epoch: {len(dataloader)}")
    
    # Create model
    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        image_size=config['image_size']
    )
    model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n🏗️ Model Info:")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    print(f"\n🎯 Starting Training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(config['device'])
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.6f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        
        print(f"\n✅ Epoch {epoch+1} completed:")
        print(f"   • Average Loss: {avg_loss:.6f}")
        print(f"   • Time: {epoch_time:.1f} seconds")
        
        # Save checkpoint
        checkpoint_path = save_dir / f"quick_test_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"   • Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Quick Training Test Completed!")
    print(f"   • Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   • Final loss: {avg_loss:.6f}")
    print(f"   • Results saved in: {save_dir}")
    
    # Test generation
    print(f"\n🧪 Testing Generation...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(dataloader))
        sample_images = sample_batch['image'][:4].to(config['device'])
        
        # Generate
        generated = model(sample_images)
        
        print(f"   • Input shape: {sample_images.shape}")
        print(f"   • Output shape: {generated.shape}")
        print(f"   • Generation successful!")
    
    print(f"\n✅ Quick test validation complete!")
    print(f"   • Training pipeline works correctly")
    print(f"   • Model can process images")
    print(f"   • Checkpoints saved successfully")
    print(f"   • Ready for full training!")

def main():
    """Main function"""
    
    print("🎌 Quick Training Test for Kanji Diffusion")
    print("=" * 50)
    
    # Check if test dataset exists
    test_dataset_path = Path("data/fixed_kanji_dataset/metadata/test_dataset.json")
    if not test_dataset_path.exists():
        print("❌ Test dataset not found! Please run quick_test_config.py first.")
        return
    
    # Check if main dataset exists
    main_dataset_path = Path("data/fixed_kanji_dataset/metadata/dataset.json")
    if not main_dataset_path.exists():
        print("❌ Main dataset not found! Please run fix_kanji_dataset.py first.")
        return
    
    # Run quick training test
    quick_training_test()

if __name__ == "__main__":
    main()
