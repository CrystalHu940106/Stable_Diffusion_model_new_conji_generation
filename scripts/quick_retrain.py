#!/usr/bin/env python3
"""
Quick retraining script to fix VAE architecture
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_diffusion_kanji import VAE, UNet2DConditionModel, DDPMScheduler
from PIL import Image

class KanjiDataset(Dataset):
    def __init__(self, dataset_path, transform=None, max_samples=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load metadata
        metadata_path = self.dataset_path / "metadata" / "dataset.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Limit samples if specified
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"📚 Loaded {len(self.data)} Kanji entries")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Load image
        image_path = self.dataset_path / "images" / entry['image_file']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def quick_retrain():
    """Quick retraining with fixed VAE architecture"""
    print("🚀 Quick Retraining with Fixed VAE Architecture")
    print("=" * 50)
    
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size = 4
    num_epochs = 5  # Quick training
    learning_rate = 1e-4
    
    print(f"🔧 Configuration:")
    print(f"   • Device: {device}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Learning Rate: {learning_rate}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    dataset_path = "data/fixed_kanji_dataset"
    dataset = KanjiDataset(dataset_path, transform=transform, max_samples=500)  # Reduced for quick training
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset: {len(dataset)} total, {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Initialize models with fixed architecture
    vae = VAE(hidden_dims=[128, 256, 512, 1024]).to(device)
    unet = UNet2DConditionModel(
        model_channels=256,
        num_res_blocks=3,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(8,),
        num_heads=16
    ).to(device)
    
    # Test VAE dimensions
    test_input = torch.randn(1, 3, 128, 128).to(device)
    latents, mu, logvar, kl_loss = vae.encode(test_input)
    reconstructed = vae.decode(latents)
    print(f"🧪 VAE Test: input {test_input.shape} → latents {latents.shape} → output {reconstructed.shape}")
    
    if reconstructed.shape[-2:] != test_input.shape[-2:]:
        print("❌ VAE output dimensions still incorrect!")
        return
    
    print("✅ VAE architecture is correct!")
    
    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Optimizers
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
    unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate/10)
    
    # Loss function
    mse_loss = nn.MSELoss()
    
    # Training loop
    print(f"\n🚀 Starting quick training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{num_epochs} {'='*20}")
        
        # Training
        vae.train()
        unet.train()
        total_loss = 0
        
        for batch_idx, images in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = images.to(device)
            
            # VAE reconstruction
            latents, mu, logvar, kl_loss = vae.encode(images)
            reconstructed = vae.decode(latents)
            recon_loss = mse_loss(reconstructed, images)
            
            # UNet noise prediction (simplified)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, torch.randn(images.shape[0], 77, 512).to(device))
            noise_loss = mse_loss(noise_pred, noise)
            
            # Total loss
            loss = recon_loss + 0.01 * kl_loss + noise_loss
            
            # Backward pass
            vae_optimizer.zero_grad()
            unet_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            unet_optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}: Loss={loss.item():.6f}, Recon={recon_loss.item():.6f}, Noise={noise_loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"📊 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': vae.state_dict(),
            'unet_state_dict': unet.state_dict(),
            'vae_optimizer_state_dict': vae_optimizer.state_dict(),
            'unet_optimizer_state_dict': unet_optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, f'fixed_vae_checkpoint_epoch_{epoch}.pth')
        print(f"💾 Checkpoint saved: fixed_vae_checkpoint_epoch_{epoch}.pth")
    
    print(f"\n🎉 Quick retraining completed!")
    print(f"💾 Final model saved as: fixed_vae_checkpoint_epoch_{num_epochs}.pth")

if __name__ == "__main__":
    quick_retrain()
