#!/usr/bin/env python3
"""
Complete Stable Diffusion Training Script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import json
from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts.stable_diffusion_kanji import VAE, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline

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
        
        # Create prompt from meanings
        prompt = f"kanji character meaning: {', '.join(entry['meanings'][:3])}"
        
        return {
            'image': image,
            'prompt': prompt,
            'kanji': entry['kanji']
        }

class StableDiffusionTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize models with larger capacity
        self.vae = VAE(hidden_dims=[128, 256, 512, 1024]).to(device)
        self.unet = UNet2DConditionModel(
            model_channels=256,  # Increased from 128
            num_res_blocks=3,    # Increased from 2
            channel_mult=(1, 2, 4, 8, 16),  # More levels
            num_heads=16         # Increased from 8
        ).to(device)
        
        # Initialize scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # CLIP text encoder
        from transformers import CLIPTokenizer, CLIPTextModel
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.text_encoder.eval()
        
        # Optimizers with different learning rates
        self.vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=1e-4, weight_decay=1e-6)
        self.unet_optimizer = torch.optim.AdamW(self.unet.parameters(), lr=1e-5, weight_decay=1e-6)
        
        # Learning rate schedulers
        self.vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.vae_optimizer, T_max=100)
        self.unet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.unet_optimizer, T_max=100)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        print(f"🚀 Trainer initialized on {device}")
        print(f"   VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        print(f"   UNet parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
    
    def encode_text(self, prompts):
        """Encode text prompts to embeddings"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        tokens = self.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
        
        return text_embeddings
    
    def train_step(self, images, prompts, timesteps):
        """Single training step with improved loss calculation"""
        batch_size = images.shape[0]
        
        # Encode images to latent space with KL loss
        latents, mu, logvar, kl_loss = self.vae.encode(images)
        
        # Encode text prompts
        text_embeddings = self.encode_text(prompts)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise using UNet
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)
        
        # Calculate losses
        noise_loss = self.mse_loss(noise_pred, noise)
        
        # VAE reconstruction loss
        reconstructed = self.vae.decode(latents)
        recon_loss = self.mse_loss(reconstructed, images)
        
        # Total loss with KL divergence
        total_loss = noise_loss + 0.1 * recon_loss + 0.01 * kl_loss
        
        # Backward pass
        self.vae_optimizer.zero_grad()
        self.unet_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        
        self.vae_optimizer.step()
        self.unet_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'noise_loss': noise_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.vae.train()
        self.unet.train()
        
        total_loss = 0
        total_noise_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            prompts = batch['prompt']
            
            # Generate random timesteps
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],), device=self.device)
            
            # Training step
            losses = self.train_step(images, prompts, timesteps)
            
            # Update metrics
            total_loss += losses['total_loss']
            total_noise_loss += losses['noise_loss']
            total_recon_loss += losses['recon_loss']
            total_kl_loss += losses['kl_loss']
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss']:.6f}",
                'Noise': f"{losses['noise_loss']:.6f}",
                'Recon': f"{losses['recon_loss']:.6f}",
                'KL': f"{losses['kl_loss']:.6f}"
            })
        
        # Update learning rates
        self.vae_scheduler.step()
        self.unet_scheduler.step()
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_noise_loss = total_noise_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        print(f"\n📊 Epoch {epoch} Training Results:")
        print(f"   • Average Total Loss: {avg_loss:.6f}")
        print(f"   • Average Noise Loss: {avg_noise_loss:.6f}")
        print(f"   • Average Recon Loss: {avg_recon_loss:.6f}")
        print(f"   • Average KL Loss: {avg_kl_loss:.6f}")
        print(f"   • VAE LR: {self.vae_scheduler.get_last_lr()[0]:.2e}")
        print(f"   • UNet LR: {self.unet_scheduler.get_last_lr()[0]:.2e}")
        
        return {
            'train_loss': avg_loss,
            'noise_loss': avg_noise_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate(self, dataloader):
        """Validate the model"""
        self.vae.eval()
        self.unet.eval()
        
        total_loss = 0
        total_noise_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                prompts = batch['prompt']
                
                # Generate random timesteps
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],), device=self.device)
                
                # Forward pass
                latents, mu, logvar, kl_loss = self.vae.encode(images)
                text_embeddings = self.encode_text(prompts)
                
                noise = torch.randn_like(latents)
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)
                
                # Calculate losses
                noise_loss = self.mse_loss(noise_pred, noise)
                reconstructed = self.vae.decode(latents)
                recon_loss = self.mse_loss(reconstructed, images)
                total_loss_val = noise_loss + 0.1 * recon_loss + 0.01 * kl_loss
                
                total_loss += total_loss_val.item()
                total_noise_loss += noise_loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_noise_loss = total_noise_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        print(f"\n📊 Validation Results:")
        print(f"   • Average Total Loss: {avg_loss:.6f}")
        print(f"   • Average Noise Loss: {avg_noise_loss:.6f}")
        print(f"   • Average Recon Loss: {avg_recon_loss:.6f}")
        print(f"   • Average KL Loss: {avg_kl_loss:.6f}")
        
        return {
            'val_loss': avg_loss,
            'noise_loss': avg_noise_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def test_diffusion_components(self):
        """Test all diffusion components before training"""
        print("🧪 Testing diffusion components...")
        
        # Test VAE
        test_image = torch.randn(2, 3, 128, 128).to(self.device)
        try:
            latents, mu, logvar, kl_loss = self.vae.encode(test_image)
            reconstructed = self.vae.decode(latents)
            print(f"✅ VAE: input {test_image.shape} → latents {latents.shape} → output {reconstructed.shape}")
        except Exception as e:
            print(f"❌ VAE test failed: {e}")
            return False
        
        # Test UNet
        test_latents = torch.randn(2, 4, 16, 16).to(self.device)
        test_timesteps = torch.randint(0, 1000, (2,)).to(self.device)
        test_context = torch.randn(2, 77, 512).to(self.device)
        
        try:
            output = self.unet(test_latents, test_timesteps, test_context)
            print(f"✅ UNet: input {test_latents.shape} → output {output.shape}")
        except Exception as e:
            print(f"❌ UNet test failed: {e}")
            return False
        
        # Test scheduler
        try:
            noise = torch.randn_like(test_latents)
            noisy = self.scheduler.add_noise(test_latents, noise, test_timesteps)
            denoised = self.scheduler.step(noise, test_timesteps, noisy)
            print(f"✅ Scheduler: noise addition and denoising successful")
        except Exception as e:
            print(f"❌ Scheduler test failed: {e}")
            return False
        
        print("🎉 All component tests passed!")
        return True
    
    def save_checkpoint(self, epoch, metrics, filename=None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'unet_optimizer_state_dict': self.unet_optimizer.state_dict(),
            'vae_scheduler_state_dict': self.vae_scheduler.state_dict(),
            'unet_scheduler_state_dict': self.unet_scheduler.state_dict(),
            'metrics': metrics,
            'scheduler_config': {
                'num_train_timesteps': self.scheduler.num_train_timesteps,
                'beta_start': self.scheduler.beta_start,
                'beta_end': self.scheduler.beta_end
            }
        }
        
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.unet_optimizer.load_state_dict(checkpoint['unet_optimizer_state_dict'])
        self.vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
        self.unet_scheduler.load_state_dict(checkpoint['unet_scheduler_state_dict'])
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']

def main():
    """Main training function"""
    print("🎌 Stable Diffusion Kanji Training")
    print("=" * 50)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16  # Increased from 2
    num_epochs = 10
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
    dataset = KanjiDataset(dataset_path, transform=transform, max_samples=1000)  # Limit for testing
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset: {len(dataset)} total, {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Initialize trainer
    trainer = StableDiffusionTrainer(device=device)
    
    # Test components before training
    if not trainer.test_diffusion_components():
        print("❌ Component tests failed. Exiting.")
        return
    
    # Training loop
    best_val_loss = float('inf')
    train_metrics = []
    val_metrics = []
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{num_epochs} {'='*20}")
        
        # Train
        train_results = trainer.train_epoch(train_loader, epoch)
        train_metrics.append(train_results)
        
        # Validate
        val_results = trainer.validate(val_loader)
        val_metrics.append(val_results)
        
        # Save best model
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            trainer.save_checkpoint(epoch, val_results, "best_model.pth")
        
        # Save regular checkpoint
        if epoch % 5 == 0:
            trainer.save_checkpoint(epoch, train_results)
    
    print(f"\n🎉 Training complete!")
    print(f"   • Best validation loss: {best_val_loss:.6f}")
    
    # Save final model
    trainer.save_checkpoint(num_epochs, train_results[-1], "final_model.pth")
    
    # Plot training curves
    try:
        epochs = range(1, len(train_metrics) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, [m['train_loss'] for m in train_metrics], 'b-', label='Train')
        plt.plot(epochs, [m['val_loss'] for m in val_metrics], 'r-', label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Noise loss plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, [m['noise_loss'] for m in train_metrics], 'b-', label='Train')
        plt.plot(epochs, [m['noise_loss'] for m in val_metrics], 'r-', label='Validation')
        plt.title('Noise Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # KL loss plot
        plt.subplot(1, 3, 3)
        plt.plot(epochs, [m['kl_loss'] for m in train_metrics], 'b-', label='Train')
        plt.plot(epochs, [m['kl_loss'] for m in val_metrics], 'r-', label='Validation')
        plt.title('KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Training curves saved as 'training_curves.png'")
        
    except Exception as e:
        print(f"⚠️  Could not plot training curves: {e}")

if __name__ == "__main__":
    main()
