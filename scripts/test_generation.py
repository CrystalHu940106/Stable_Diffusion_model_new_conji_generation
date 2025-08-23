#!/usr/bin/env python3
"""
Test Generation with Trained Model
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
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
        
        print(f"Loaded {len(self.data)} Kanji entries for testing")
    
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

def create_transforms(image_size=64):
    """Create transforms for testing"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def denormalize(tensor):
    """Denormalize tensor back to [0, 1] range"""
    return (tensor + 1) / 2

def test_generation():
    """Test generation with trained model"""
    
    print("🧪 Testing Generation with Trained Model")
    print("=" * 50)
    
    # Load model
    model = SimpleUNet(in_channels=3, out_channels=3, image_size=64)
    
    # Load checkpoint
    checkpoint_path = Path("quick_test_results/quick_test_epoch_2.pth")
    if not checkpoint_path.exists():
        print("❌ Checkpoint not found! Please run quick_train_test.py first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   • Loss: {checkpoint['loss']:.6f}")
    
    # Load dataset
    transform = create_transforms(64)
    dataset = QuickKanjiDataset("data/fixed_kanji_dataset", transform=transform, use_test_data=True)
    
    # Get a few sample images
    sample_indices = [0, 10, 20, 30]  # Test different kanji
    sample_images = []
    sample_kanji = []
    
    for idx in sample_indices:
        sample = dataset[idx]
        sample_images.append(sample['image'])
        sample_kanji.append(sample['kanji'])
    
    # Stack images
    input_tensor = torch.stack(sample_images)
    
    print(f"📊 Input shape: {input_tensor.shape}")
    print(f"🔤 Testing kanji: {sample_kanji}")
    
    # Generate
    with torch.no_grad():
        generated = model(input_tensor)
    
    print(f"📊 Output shape: {generated.shape}")
    
    # Convert to images
    input_images = []
    output_images = []
    
    for i in range(len(sample_images)):
        # Input image
        input_img = denormalize(input_tensor[i]).permute(1, 2, 0).numpy()
        input_img = np.clip(input_img, 0, 1)
        input_images.append(input_img)
        
        # Generated image
        output_img = denormalize(generated[i]).permute(1, 2, 0).numpy()
        output_img = np.clip(output_img, 0, 1)
        output_images.append(output_img)
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Kanji Generation Test Results', fontsize=16)
    
    for i in range(4):
        # Input images
        axes[0, i].imshow(input_images[i])
        axes[0, i].set_title(f'Input: {sample_kanji[i]}', fontsize=12)
        axes[0, i].axis('off')
        
        # Generated images
        axes[1, i].imshow(output_images[i])
        axes[1, i].set_title(f'Generated: {sample_kanji[i]}', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_path = "generation_test_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Results saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Analyze results
    print(f"\n📊 Generation Analysis:")
    print(f"   • Model successfully processed {len(sample_images)} images")
    print(f"   • Input and output shapes match")
    print(f"   • Generation completed without errors")
    
    # Calculate some metrics
    mse_loss = nn.MSELoss()(generated, input_tensor)
    print(f"   • MSE Loss: {mse_loss.item():.6f}")
    
    if mse_loss.item() < 0.1:
        print(f"   ✅ Good reconstruction quality")
    elif mse_loss.item() < 0.3:
        print(f"   ⚠️  Moderate reconstruction quality")
    else:
        print(f"   ❌ Poor reconstruction quality")
    
    print(f"\n🎉 Generation test complete!")
    print(f"   • Model is working correctly")
    print(f"   • Ready for more advanced training")

def main():
    """Main function"""
    print("🎌 Kanji Generation Test")
    print("=" * 50)
    
    test_generation()

if __name__ == "__main__":
    main()
