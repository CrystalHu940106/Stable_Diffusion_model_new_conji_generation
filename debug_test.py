#!/usr/bin/env python3
"""
Simple test script to debug the GroupNorm issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, original_samples, noise, timesteps):
        device = original_samples.device
        
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise


def test_models():
    """Test the simplified models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧪 Testing on device: {device}")
    
    # Create models
    vae = SimpleVAE().to(device)
    unet = SimpleUNet().to(device)
    scheduler = SimpleDDPMScheduler()
    
    # Create synthetic data
    batch_size = 4
    images = torch.randn(batch_size, 3, 128, 128, device=device)
    
    print("📊 Testing VAE...")
    try:
        # Test VAE encoding
        latents, mu, logvar, kl_loss = vae.encode(images)
        print(f"   ✅ VAE encode: {images.shape} -> {latents.shape}")
        print(f"   ✅ KL loss: {kl_loss:.6f}")
        
        # Test VAE decoding
        reconstructed = vae.decode(latents)
        print(f"   ✅ VAE decode: {latents.shape} -> {reconstructed.shape}")
        
    except Exception as e:
        print(f"   ❌ VAE failed: {e}")
        return False
    
    print("🔮 Testing UNet...")
    try:
        # Test UNet
        noise = torch.randn_like(latents, device=device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        noise_pred = unet(noisy_latents, timesteps)
        print(f"   ✅ UNet: {noisy_latents.shape} -> {noise_pred.shape}")
        
    except Exception as e:
        print(f"   ❌ UNet failed: {e}")
        return False
    
    print("🎯 Testing complete training step...")
    try:
        # Test complete training step
        mse_loss = nn.MSELoss()
        
        # Forward pass
        latents, mu, logvar, kl_loss = vae.encode(images)
        noise = torch.randn_like(latents, device=device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps)
        reconstructed = vae.decode(latents)
        
        # Calculate losses
        noise_loss = mse_loss(noise_pred, noise)
        reconstruction_loss = mse_loss(reconstructed, images)
        total_loss = noise_loss + 0.1 * kl_loss + 0.1 * reconstruction_loss
        
        print(f"   ✅ Noise loss: {noise_loss:.6f}")
        print(f"   ✅ Reconstruction loss: {reconstruction_loss:.6f}")
        print(f"   ✅ Total loss: {total_loss:.6f}")
        
        # Test backward pass
        total_loss.backward()
        print("   ✅ Backward pass successful")
        
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True


if __name__ == "__main__":
    success = test_models()
    if success:
        print("✅ Models work correctly - ready for training!")
    else:
        print("❌ Models have issues - need debugging")