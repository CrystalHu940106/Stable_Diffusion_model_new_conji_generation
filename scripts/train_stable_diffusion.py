#!/usr/bin/env python3
"""
Complete Stable Diffusion Training Script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from PIL import Image
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our Stable Diffusion components
from stable_diffusion_kanji import (
    VAE, UNet2DConditionModel, DDPMScheduler, 
    StableDiffusionPipeline, create_kanji_dataset
)

class StableDiffusionTrainer:
    """Complete trainer for Stable Diffusion"""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # Initialize models
        self.vae = VAE().to(self.device)
        self.unet = UNet2DConditionModel().to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze text encoder and VAE during training
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config['num_train_timesteps']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )
        
        print("✅ Stable Diffusion Trainer initialized")
        print(f"   • UNet parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"   • VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        print(f"   • Text encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
    
    def encode_text(self, text):
        """Encode text to embeddings"""
        tokens = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=77,
            return_tensors="pt"
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
        
        return text_embeddings
    
    def encode_image(self, image):
        """Encode image to latent space"""
        return self.vae.encode(image)
    
    def train_step(self, batch):
        """Single training step"""
        images = batch['image'].to(self.device)
        prompts = batch['prompt']
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.encode_image(images)
            # Scale latents
            latents = latents * 0.18215
        
        # Encode text
        text_embeddings = self.encode_text(prompts)
        
        # Sample random timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise, reduction="none")
        loss = loss.mean(dim=[1, 2, 3]).mean()
        
        return loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.unet.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self.train_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validation step"""
        self.unet.eval()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                loss = self.train_step(batch)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, train_loss, val_loss, save_dir):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = save_dir / f"stable_diffusion_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        print(f"   • Epoch: {checkpoint['epoch']}")
        print(f"   • Train loss: {checkpoint['train_loss']:.6f}")
        print(f"   • Val loss: {checkpoint['val_loss']:.6f}")
        
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
    
    def plot_training_progress(self, train_losses, val_losses, save_dir):
        """Plot training progress"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Stable Diffusion Training Progress')
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
        
        plot_path = save_dir / "stable_diffusion_training_progress.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training progress saved: {plot_path}")
        
        plt.show()

def create_transforms(image_size=128):
    """Create transforms for training"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def train_stable_diffusion():
    """Main training function"""
    
    print("🎌 Stable Diffusion Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'image_size': 128,
        'batch_size': 2,  # Smaller batch for memory
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'num_train_timesteps': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'stable_diffusion_results',
        'train_split': 0.9,
        'save_every': 2,
        'log_every': 50
    }
    
    print(f"📊 Configuration:")
    print(f"   • Image size: {config['image_size']}x{config['image_size']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    print(f"   • Epochs: {config['num_epochs']}")
    print(f"   • Device: {config['device']}")
    print(f"   • Timesteps: {config['num_train_timesteps']}")
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Create datasets
    transform = create_transforms(config['image_size'])
    KanjiDataset = create_kanji_dataset()
    
    train_dataset = KanjiDataset("data/fixed_kanji_dataset", transform=transform)
    
    # Split into train/validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    print(f"\n📚 Dataset Info:")
    print(f"   • Training samples: {len(train_dataset)}")
    print(f"   • Validation samples: {len(val_dataset)}")
    print(f"   • Training batches: {len(train_loader)}")
    print(f"   • Validation batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = StableDiffusionTrainer(config)
    
    # Training loop
    print(f"\n🎯 Starting Stable Diffusion Training...")
    start_time = datetime.now()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_start = datetime.now()
        
        # Training
        train_loss = trainer.train_epoch(train_loader, epoch)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = trainer.validate(val_loader)
        val_losses.append(val_loss)
        
        epoch_time = datetime.now() - epoch_start
        
        print(f"\n✅ Epoch {epoch+1} completed:")
        print(f"   • Training Loss: {train_loss:.6f}")
        print(f"   • Validation Loss: {val_loss:.6f}")
        print(f"   • Time: {epoch_time}")
        print(f"   • Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            trainer.save_checkpoint(epoch, train_loss, val_loss, save_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, train_loss, val_loss, save_dir)
            trainer.save_checkpoint(epoch, train_loss, val_loss, save_dir, "best_model.pth")
            print(f"   🏆 New best model saved!")
    
    total_time = datetime.now() - start_time
    
    print(f"\n🎉 Stable Diffusion Training Completed!")
    print(f"   • Total time: {total_time}")
    print(f"   • Final training loss: {train_loss:.6f}")
    print(f"   • Final validation loss: {val_loss:.6f}")
    print(f"   • Best validation loss: {best_val_loss:.6f}")
    print(f"   • Results saved in: {save_dir}")
    
    # Plot training progress
    trainer.plot_training_progress(train_losses, val_losses, save_dir)
    
    # Save training summary
    summary = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': str(total_time),
        'final_train_loss': train_loss,
        'final_val_loss': val_loss
    }
    
    summary_path = save_dir / "stable_diffusion_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📋 Training summary saved: {summary_path}")
    
    return trainer, save_dir

def test_generation(trainer, save_dir):
    """Test generation after training"""
    
    print(f"\n🧪 Testing Generation...")
    
    # Create pipeline for generation
    pipeline = StableDiffusionPipeline(trainer.device)
    pipeline.vae = trainer.vae
    pipeline.unet = trainer.unet
    pipeline.text_encoder = trainer.text_encoder
    pipeline.tokenizer = trainer.tokenizer
    
    # Test prompts
    test_prompts = [
        "kanji character meaning: success, achieve, accomplish",
        "kanji character meaning: failure, lose, defeat",
        "kanji character meaning: novel, new, creative",
        "kanji character meaning: funny, humorous, amusing",
        "kanji character meaning: culture, tradition, heritage",
        "kanji character meaning: technology, innovation, future",
        "kanji character meaning: love, heart, emotion",
        "kanji character meaning: strength, power, energy"
    ]
    
    for prompt in test_prompts:
        try:
            generated = pipeline.generate(prompt, num_inference_steps=50)
            print(f"✅ Generated: {prompt}")
            
            # Save result
            output_path = save_dir / f"generated_{prompt.split(':')[1].strip().replace(', ', '_')[:30]}.png"
            generated_image = transforms.ToPILImage()(generated.squeeze(0))
            generated_image.save(output_path)
            print(f"💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error generating '{prompt}': {e}")
    
    print(f"🎉 Generation test complete!")

def main():
    """Main function"""
    
    print("🎌 Stable Diffusion Training Script")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("data/fixed_kanji_dataset")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please run fix_kanji_dataset.py first.")
        return
    
    # Run training
    trainer, save_dir = train_stable_diffusion()
    
    # Test generation
    test_generation(trainer, save_dir)
    
    print(f"\n🎯 Training Summary:")
    print(f"   • Stable Diffusion model trained successfully")
    print(f"   • Checkpoints saved in: {save_dir}")
    print(f"   • Best model: {save_dir}/best_model.pth")
    print(f"   • Ready for text-to-kanji generation!")

if __name__ == "__main__":
    main()
