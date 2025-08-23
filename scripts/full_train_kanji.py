#!/usr/bin/env python3
"""
Full Kanji Training Script
Train the complete model on the full dataset
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
import matplotlib.pyplot as plt
from datetime import datetime

class KanjiDataset(Dataset):
    """Full Kanji dataset"""
    
    def __init__(self, dataset_path, transform=None, split='train', train_split=0.9):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load dataset metadata
        metadata_path = self.dataset_path / "metadata" / "dataset.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Split data
        total_size = len(self.data)
        train_size = int(total_size * train_split)
        
        if split == 'train':
            self.data = self.data[:train_size]
        else:  # validation
            self.data = self.data[train_size:]
        
        print(f"Loaded {len(self.data)} Kanji entries for {split}")
    
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

class AdvancedUNet(nn.Module):
    """Advanced UNet for better Kanji generation"""
    
    def __init__(self, in_channels=3, out_channels=3, image_size=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second level
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third level
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth level
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Fourth level
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Third level
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Second level
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final output
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_transforms(image_size=128):
    """Create transforms for training"""
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

def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"📂 Checkpoint loaded: {checkpoint_path}")
    print(f"   • Epoch: {epoch}")
    print(f"   • Loss: {loss:.6f}")
    
    return epoch, loss

def plot_training_progress(train_losses, val_losses, save_dir):
    """Plot training progress"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss Detail')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training progress saved: {plot_path}")
    
    plt.show()

def full_training():
    """Run full training on the complete dataset"""
    
    print("🎌 Full Kanji Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'image_size': 128,
        'batch_size': 4,
        'learning_rate': 2e-4,
        'num_epochs': 5,
        'device': 'cpu',
        'save_dir': 'full_training_results',
        'train_split': 0.9,
        'save_every': 1,
        'log_every': 100
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
    
    # Create datasets and dataloaders
    transform = create_transforms(config['image_size'])
    
    train_dataset = KanjiDataset("data/fixed_kanji_dataset", transform=transform, split='train', train_split=config['train_split'])
    val_dataset = KanjiDataset("data/fixed_kanji_dataset", transform=transform, split='val', train_split=config['train_split'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    print(f"\n📚 Dataset Info:")
    print(f"   • Training samples: {len(train_dataset)}")
    print(f"   • Validation samples: {len(val_dataset)}")
    print(f"   • Training batches: {len(train_loader)}")
    print(f"   • Validation batches: {len(val_loader)}")
    
    # Create model
    model = AdvancedUNet(
        in_channels=3,
        out_channels=3,
        image_size=config['image_size']
    )
    model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    print(f"\n🏗️ Model Info:")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    print(f"\n🎯 Starting Training...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(config['device'])
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Progress update
            if batch_idx % config['log_every'] == 0:
                print(f"   Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(config['device'])
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n✅ Epoch {epoch+1} completed:")
        print(f"   • Training Loss: {avg_train_loss:.6f}")
        print(f"   • Validation Loss: {avg_val_loss:.6f}")
        print(f"   • Time: {epoch_time:.1f} seconds")
        print(f"   • Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_name = f"full_training_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch+1, avg_val_loss, save_dir, checkpoint_name)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch+1, avg_val_loss, save_dir, "best_model.pth")
            print(f"   🏆 New best model saved!")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Full Training Completed!")
    print(f"   • Total time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
    print(f"   • Final training loss: {avg_train_loss:.6f}")
    print(f"   • Final validation loss: {avg_val_loss:.6f}")
    print(f"   • Best validation loss: {best_val_loss:.6f}")
    print(f"   • Results saved in: {save_dir}")
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses, save_dir)
    
    # Save training summary
    summary = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss
    }
    
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Training summary saved: {summary_path}")
    
    return model, save_dir

def main():
    """Main function"""
    
    print("🎌 Full Kanji Training Script")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("data/fixed_kanji_dataset")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please run fix_kanji_dataset.py first.")
        return
    
    # Run full training
    model, save_dir = full_training()
    
    print(f"\n🎯 Training Summary:")
    print(f"   • Model trained on full dataset")
    print(f"   • Checkpoints saved in: {save_dir}")
    print(f"   • Best model: {save_dir}/best_model.pth")
    print(f"   • Ready for generation!")

if __name__ == "__main__":
    main()
