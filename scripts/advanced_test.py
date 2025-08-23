#!/usr/bin/env python3
"""
Advanced Testing Script for Kanji Diffusion Model
Test different generation strategies and parameters
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

class SimpleUNet(nn.Module):
    """Simple UNet for testing generation"""
    
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

def load_model(model_path):
    """Load trained model"""
    model = SimpleUNet()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def create_different_noise_patterns(image_size=64, num_samples=4):
    """Create different types of noise for generation"""
    
    patterns = {}
    
    # 1. Random noise
    patterns['random'] = torch.randn(num_samples, 3, image_size, image_size)
    
    # 2. Structured noise (more organized)
    x = torch.linspace(-1, 1, image_size)
    y = torch.linspace(-1, 1, image_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    structured = torch.stack([X, Y, torch.zeros_like(X)], dim=0)
    structured = structured.unsqueeze(0).repeat(num_samples, 1, 1, 1)
    patterns['structured'] = structured
    
    # 3. Low-frequency noise
    low_freq = torch.randn(num_samples, 3, image_size//4, image_size//4)
    low_freq = torch.nn.functional.interpolate(low_freq, size=(image_size, image_size), mode='bilinear')
    patterns['low_frequency'] = low_freq
    
    # 4. High-frequency noise
    high_freq = torch.randn(num_samples, 3, image_size, image_size) * 0.5
    patterns['high_frequency'] = high_freq
    
    return patterns

def generate_with_different_strategies(model, save_dir="advanced_results"):
    """Generate using different strategies"""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"\n🎨 Testing Different Generation Strategies...")
    
    # Create different noise patterns
    noise_patterns = create_different_noise_patterns()
    
    for pattern_name, noise_input in noise_patterns.items():
        print(f"\n   Testing {pattern_name} noise pattern...")
        
        # Generate
        with torch.no_grad():
            generated = model(noise_input)
        
        # Save results
        generated = (generated + 1) / 2
        generated = torch.clamp(generated, 0, 1)
        
        for i in range(generated.shape[0]):
            img_array = generated[i].permute(1, 2, 0).numpy()
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            
            img_path = save_path / f"success_{pattern_name}_{i+1}.png"
            img.save(img_path)
            print(f"     • Saved: {img_path}")

def test_interpolation_between_kanji(model, dataset_path="kanji_dataset"):
    """Test interpolation between existing Kanji"""
    
    print(f"\n🔄 Testing Interpolation Between Kanji...")
    
    # Load some existing Kanji
    with open(f"{dataset_path}/metadata/dataset.json", 'r') as f:
        dataset = json.load(f)
    
    # Find some success-related Kanji
    success_kanji = []
    for entry in dataset:
        meanings = [m.lower() for m in entry['meanings']]
        if any(word in meanings for word in ['success', 'achieve', 'complete', 'finish']):
            success_kanji.append(entry)
            if len(success_kanji) >= 4:
                break
    
    if len(success_kanji) < 2:
        print("   ❌ Not enough success-related Kanji found for interpolation")
        return
    
    print(f"   • Found {len(success_kanji)} success-related Kanji for interpolation")
    
    # Load and process Kanji images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    kanji_images = []
    for kanji_info in success_kanji[:2]:  # Use first 2
        img_path = f"{dataset_path}/images/{kanji_info['image_file']}"
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        kanji_images.append(img_tensor)
        print(f"     • Loaded: {kanji_info['kanji']} ({', '.join(kanji_info['meanings'][:2])})")
    
    # Interpolate between them
    save_path = Path("advanced_results")
    save_path.mkdir(exist_ok=True)
    
    for i in range(5):  # 5 interpolation steps
        alpha = i / 4.0
        interpolated = (1 - alpha) * kanji_images[0] + alpha * kanji_images[1]
        
        # Generate from interpolated
        with torch.no_grad():
            generated = model(interpolated)
        
        # Save
        generated = (generated + 1) / 2
        generated = torch.clamp(generated, 0, 1)
        
        img_array = generated[0].permute(1, 2, 0).numpy()
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        img_path = save_path / f"interpolation_step_{i+1}.png"
        img.save(img_path)
        print(f"     • Interpolation {i+1}/5: {img_path}")

def test_model_analysis():
    """Analyze model behavior and capabilities"""
    
    print(f"\n🔍 Model Analysis...")
    
    # Load model
    model_path = "quick_test_results/quick_test_epoch_2.pth"
    model = load_model(model_path)
    
    # Analyze model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    
    # Test with different input sizes
    print(f"   • Testing model with different inputs...")
    
    test_inputs = [
        torch.randn(1, 3, 64, 64),
        torch.randn(1, 3, 32, 32),  # Smaller
        torch.randn(1, 3, 128, 128),  # Larger
    ]
    
    for i, test_input in enumerate(test_inputs):
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"     • Input {test_input.shape} -> Output {output.shape} ✅")
        except Exception as e:
            print(f"     • Input {test_input.shape} -> Error: {e} ❌")

def main():
    """Main function"""
    
    print("🎌 Advanced Kanji Generation Testing")
    print("=" * 50)
    
    # Check if model exists
    model_path = Path("quick_test_results/quick_test_epoch_2.pth")
    if not model_path.exists():
        print("❌ Trained model not found! Please run quick_train_test.py first.")
        return
    
    # Load model
    model = load_model(model_path)
    print(f"✅ Model loaded successfully")
    
    # Run different tests
    generate_with_different_strategies(model)
    test_interpolation_between_kanji(model)
    test_model_analysis()
    
    print(f"\n🎉 Advanced testing completed!")
    print(f"   • Check advanced_results/ directory for generated images")
    print(f"   • Different noise patterns tested")
    print(f"   • Interpolation between Kanji tested")
    print(f"   • Model analysis completed")
    
    print(f"\n📊 Test Results Summary:")
    print(f"   • Generated images with 4 different noise patterns")
    print(f"   • Tested interpolation between existing Kanji")
    print(f"   • Analyzed model capabilities and parameters")
    print(f"   • All results saved in advanced_results/ directory")

if __name__ == "__main__":
    main()
