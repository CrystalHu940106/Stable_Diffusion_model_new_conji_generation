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
from stable_diffusion_kanji import VAE, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline

def get_optimal_batch_size(device):
    """智can选择最优批处理大小"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_memory > 8:
            return 16
        elif gpu_memory > 4:
            return 8
        else:
            return 4
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 4
    else:
        return 2

class EMAModel:
    """指数移动平均model"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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
            channel_mult=(1, 2, 4, 8),  # Reduced to match VAE latent space
            attention_resolutions=(8,),  # Only at 8x8 resolution
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
        
        # EMA models for better quality
        self.vae_ema = EMAModel(self.vae)
        self.unet_ema = EMAModel(self.unet)
        
        # Mixed precision training (GPU only)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Gradient accumulation
        self.accumulation_steps = 4
        
        print(f"🚀 Trainer initialized on {device}")
        print(f"   VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        print(f"   UNet parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"   Mixed Precision: {'✅' if self.use_amp else '❌'}")
        print(f"   Gradient Accumulation: {self.accumulation_steps} steps")
    
    def encode_text(self, prompts):
        """Encode text prompts to embeddings"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        tokens = self.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
        
        return text_embeddings
    
    def train_step(self, images, prompts, timesteps, step_idx):
        """Single training step with mixed precision and gradient accumulation"""
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
        
        # Scale loss for gradient accumulation
        total_loss = total_loss / self.accumulation_steps
        
        # Backward pass with mixed precision
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Gradient accumulation
        if (step_idx + 1) % self.accumulation_steps == 0:
            if self.use_amp:
                # Gradient clipping
                self.scaler.unscale_(self.vae_optimizer)
                self.scaler.unscale_(self.unet_optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.vae_optimizer)
                self.scaler.step(self.unet_optimizer)
                self.scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.vae_optimizer.step()
                self.unet_optimizer.step()
            
            # Zero gradients
            self.vae_optimizer.zero_grad()
            self.unet_optimizer.zero_grad()
            
            # Update EMA models
            self.vae_ema.update()
            self.unet_ema.update()
        
        return {
            'total_loss': total_loss.item() * self.accumulation_steps,
            'noise_loss': noise_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with optimized settings"""
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
            losses = self.train_step(images, prompts, timesteps, batch_idx)
            
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
        """Validate the model using EMA models"""
        self.vae_ema.apply_shadow()
        self.unet_ema.apply_shadow()
        
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
        
        # Restore original models
        self.vae_ema.restore()
        self.unet_ema.restore()
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_noise_loss = total_noise_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        print(f"\n📊 Validation Results (EMA):")
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
        """Save model checkpoint with EMA models"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'vae_ema_state_dict': self.vae_ema.shadow,
            'unet_ema_state_dict': self.unet_ema.shadow,
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
        """Load model checkpoint with EMA models"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.vae_ema.shadow = checkpoint['vae_ema_state_dict']
        self.unet_ema.shadow = checkpoint['unet_ema_state_dict']
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.unet_optimizer.load_state_dict(checkpoint['unet_optimizer_state_dict'])
        self.vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
        self.unet_scheduler.load_state_dict(checkpoint['unet_scheduler_state_dict'])
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']

def main():
    """Main training function with performance optimizations"""
    print("🎌 Stable Diffusion Kanji Training - Performance Optimized")
    print("=" * 60)
    
    # Configuration with performance optimizations
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    batch_size = get_optimal_batch_size(device)
    num_epochs = 25  # Increased from 10 for better quality
    learning_rate = 1e-4
    validation_frequency = 5  # Reduce validation frequency for speed
    
    print(f"🔧 Configuration:")
    print(f"   • Device: {device}")
    print(f"   • Batch Size: {batch_size} (auto-optimized)")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Learning Rate: {learning_rate}")
    print(f"   • Validation Frequency: Every {validation_frequency} epochs")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    dataset_path = "data/fixed_kanji_dataset"
    dataset = KanjiDataset(dataset_path, transform=transform, max_samples=2000)  # Increased for better training
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"📊 Dataset: {len(dataset)} total, {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Initialize trainer
    trainer = StableDiffusionTrainer(device=device)
    
    # Test components before training
    if not trainer.test_diffusion_components():
        print("❌ Component tests failed. Exiting.")
        return
    
    # Training loop with performance optimizations
    best_val_loss = float('inf')
    train_metrics = []
    val_metrics = []
    
    print(f"\n🚀 Starting optimized training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{num_epochs} {'='*20}")
        
        # Train
        train_results = trainer.train_epoch(train_loader, epoch)
        train_metrics.append(train_results)
        
        # Validate less frequently for speed
        if epoch % validation_frequency == 0 or epoch == num_epochs:
            val_results = trainer.validate(val_loader)
            val_metrics.append(val_results)
            
            # Save best model
            if val_results['val_loss'] < best_val_loss:
                best_val_loss = val_results['val_loss']
                trainer.save_checkpoint(epoch, val_results, "best_model.pth")
        
        # Save regular checkpoint
        if epoch % 10 == 0:
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
        if val_metrics:
            val_epochs = [i for i in epochs if i % validation_frequency == 0 or i == num_epochs]
            plt.plot(val_epochs, [m['val_loss'] for m in val_metrics], 'r-', label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Noise loss plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, [m['noise_loss'] for m in train_metrics], 'b-', label='Train')
        if val_metrics:
            plt.plot(val_epochs, [m['noise_loss'] for m in val_metrics], 'r-', label='Validation')
        plt.title('Noise Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # KL loss plot
        plt.subplot(1, 3, 3)
        plt.plot(epochs, [m['kl_loss'] for m in train_metrics], 'b-', label='Train')
        if val_metrics:
            plt.plot(val_epochs, [m['kl_loss'] for m in val_metrics], 'r-', label='Validation')
        plt.title('KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves_optimized.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Training curves saved as 'training_curves_optimized.png'")
        
    except Exception as e:
        print(f"⚠️  Could not plot training curves: {e}")

if __name__ == "__main__":
    main()
