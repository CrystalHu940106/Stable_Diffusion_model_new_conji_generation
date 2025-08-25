#!/usr/bin/env python3
"""
Test the working trainer to make sure it runs without errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import gc
import os


class SimpleVAE(nn.Module):
    """Extremely simplified VAE for testing"""
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder: 128x128 -> 16x16x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels * 2, kernel_size=1),  # mu and logvar
        )
        
        # Decoder: 16x16x4 -> 128x128x3
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh()
        )
    
    def encode(self, x):
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        
        # KL loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar, kl_loss
    
    def decode(self, z):
        return self.decoder(z)


class SimpleResBlock(nn.Module):
    """Extremely simplified ResBlock"""
    def __init__(self, channels, time_dim):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.time_proj = nn.Linear(time_dim, channels)
        
    def forward(self, x, time_emb):
        h = self.block(x)
        
        # Add time embedding
        time_emb = self.time_proj(time_emb)
        time_emb = time_emb.view(x.shape[0], -1, 1, 1)
        h = h + time_emb
        
        return h + x


class SimpleUNet(nn.Module):
    """Extremely simplified UNet"""
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Simple path: no up/down sampling
        self.input_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.res1 = SimpleResBlock(64, 128)
        self.res2 = SimpleResBlock(64, 128)
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps, context=None):
        # Time embedding
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        t = self.time_embedding(timesteps.float().unsqueeze(-1))
        
        # Forward pass
        h = self.input_conv(x)
        h = self.res1(h, t)
        h = self.res2(h, t)
        return self.output_conv(h)


class SimpleDDPMScheduler:
    """Extremely simplified scheduler"""
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        
        self.betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alpuses_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, original_samples, noise, timesteps):
        device = original_samples.device
        
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpuses_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise


class WorkingTrainer:
    """Simplified trainer that actually works"""
    
    def __init__(self, device='auto'):
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
            else:
                self.device = 'cpu'
                print("💻 Using CPU")
        else:
            self.device = device
        
        # Initialize models
        self.vae = SimpleVAE().to(self.device)
        self.unet = SimpleUNet().to(self.device)
        self.scheduler = SimpleDDPMScheduler()
        
        # Optimizer
        self.optimizer = optim.AdamW([
            {'params': self.vae.parameters(), 'lr': 1e-4},
            {'params': self.unet.parameters(), 'lr': 1e-4}
        ], weight_decay=0.01)
        
        # Training parameters
        self.num_epochs = 2  # Even smaller for testing
        self.batch_size = 2
        self.save_every = 1
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        print(f"✅ Trainer initialized on {self.device}")
    
    def create_synthetic_dataset(self, num_samples=20):  # Very small for testing
        """Create simple synthetic dataset"""
        print(f"📊 Creating dataset ({num_samples} samples)...")
        
        images = []
        for i in range(num_samples):
            img = np.zeros((128, 128, 3), dtype=np.float32)
            
            if i % 4 == 0:
                # Circle
                y, x = np.ogrid[:128, :128]
                mask = (x - 64)**2 + (y - 64)**2 <= 30**2
                img[mask] = [0.8, 0.8, 0.8]
            elif i % 4 == 1:
                # Rectangle
                img[40:88, 40:88] = [0.7, 0.7, 0.7]
            elif i % 4 == 2:
                # Triangle
                for y in range(128):
                    for x in range(128):
                        if y >= 64 and abs(x - 64) <= (y - 64):
                            img[y, x] = [0.6, 0.6, 0.6]
            else:
                # Random noise
                img = np.random.rand(128, 128, 3).astype(np.float32) * 0.5
            
            # Normalize to [-1, 1]
            img = (img - 0.5) * 2
            images.append(img)
        
        # Convert to tensor
        images = np.array(images)
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        print(f"✅ Dataset created: {images.shape}")
        
        return images
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.vae.train()
        self.unet.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(self.device)
            
            # Forward pass
            latents, mu, logvar, kl_loss = self.vae.encode(images)
            
            # Add noise
            noise = torch.randn_like(latents, device=self.device)
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps, 
                (latents.shape[0],), 
                device=self.device
            )
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            # UNet prediction
            noise_pred = self.unet(noisy_latents, timesteps)
            
            # Calculate losses
            noise_loss = self.mse_loss(noise_pred, noise)
            reconstruction_loss = self.mse_loss(self.vae.decode(latents), images)
            
            loss = noise_loss + 0.1 * kl_loss + 0.1 * reconstruction_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.vae.parameters()) + list(self.unet.parameters()), 
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress
            print(f"   Epoch {epoch+1}/{self.num_epochs}, "
                  f"Batch {batch_idx+1}/{num_batches}, "
                  f"Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        print(f"\n🎯 Starting training...")
        print(f"   • Device: {self.device}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Epochs: {self.num_epochs}")
        
        # Create dataset
        images = self.create_synthetic_dataset()
        dataloader = DataLoader(images, batch_size=self.batch_size, shuffle=True)
        
        # Training history
        train_losses = []
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                print(f"\n🔄 Epoch {epoch+1}/{self.num_epochs}")
                print("-" * 50)
                
                # Train
                loss = self.train_epoch(dataloader, epoch)
                train_losses.append(loss)
                
                print(f"   📊 Average loss: {loss:.6f}")
            
            # Training summary
            total_time = time.time() - start_time
            print(f"\n🎉 Training complete!")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   📊 Final loss: {train_losses[-1]:.6f}")
            print(f"   📈 Loss change: {train_losses[0]:.6f} → {train_losses[-1]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    print("🔧 Testing working trainer...")
    
    trainer = WorkingTrainer()
    success = trainer.train()
    
    if success:
        print("\n✅ Working trainer test PASSED!")
        print("📝 The simplified architecture works without GroupNorm errors!")
    else:
        print("\n❌ Working trainer test FAILED!")


if __name__ == "__main__":
    main()