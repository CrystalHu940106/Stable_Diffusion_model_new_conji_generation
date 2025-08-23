#!/usr/bin/env python3
"""
Debug Generation Issues
Analyze why generated images are mostly white and provide solutions
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import json

def analyze_model_output():
    """Analyze what the model is actually outputting"""
    
    print("🔍 Debugging Generation Issues")
    print("=" * 50)
    
    # Load the trained model
    model_path = "quick_test_results/quick_test_epoch_2.pth"
    if not Path(model_path).exists():
        print("❌ Model not found!")
        return
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✅ Model loaded: {model_path}")
    print(f"   • Final loss: {checkpoint.get('loss', 'Unknown'):.2e}")
    
    # Create test input
    test_input = torch.randn(1, 3, 64, 64)
    print(f"   • Test input shape: {test_input.shape}")
    print(f"   • Test input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # Load model and generate
    from quick_train_test import SimpleUNet
    model = SimpleUNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   • Model output shape: {output.shape}")
    print(f"   • Model output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   • Model output mean: {output.mean():.3f}")
    print(f"   • Model output std: {output.std():.3f}")
    
    # Analyze the issue
    print(f"\n🎯 Problem Analysis:")
    
    if output.min() > 0.9:
        print(f"   ❌ Issue: Output is too bright (all values > 0.9)")
        print(f"   💡 Cause: Model learned to output white/light colors")
    elif output.max() < 0.1:
        print(f"   ❌ Issue: Output is too dark (all values < 0.1)")
        print(f"   💡 Cause: Model learned to output black/dark colors")
    elif output.std() < 0.01:
        print(f"   ❌ Issue: Output has no variation (std < 0.01)")
        print(f"   💡 Cause: Model output is uniform/constant")
    else:
        print(f"   ✅ Output range looks reasonable")
    
    return output

def analyze_training_data():
    """Analyze the training data to understand the issue"""
    
    print(f"\n📚 Training Data Analysis:")
    
    # Load dataset
    dataset_path = Path("kanji_dataset/metadata/test_dataset.json")
    if not dataset_path.exists():
        print("❌ Test dataset not found!")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"   • Test dataset size: {len(dataset)} samples")
    
    # Check a few sample images
    from quick_train_test import QuickKanjiDataset, create_quick_transforms
    
    transform = create_quick_transforms()
    test_dataset = QuickKanjiDataset("kanji_dataset", transform=transform, use_test_data=True)
    
    print(f"\n   📸 Sample Training Images:")
    for i in range(3):
        sample = test_dataset[i]
        img_tensor = sample['image']
        
        print(f"     {i+1}. {sample['kanji']}: {', '.join(sample['meanings'][:2])}")
        print(f"        • Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"        • Tensor mean: {img_tensor.mean():.3f}")
        print(f"        • Tensor std: {img_tensor.std():.3f}")

def provide_solutions():
    """Provide solutions to fix the generation issues"""
    
    print(f"\n💡 Solutions to Fix Generation Issues:")
    
    print(f"\n🔧 Immediate Fixes:")
    print(f"   1. **Invert Output**: Convert white output to black strokes")
    print(f"   2. **Adjust Threshold**: Apply threshold to create binary images")
    print(f"   3. **Change Loss Function**: Use different loss for better learning")
    print(f"   4. **Modify Training Data**: Invert training images (black background)")
    
    print(f"\n🚀 Long-term Solutions:")
    print(f"   1. **Full Training**: Train on complete dataset (6,410 samples)")
    print(f"   2. **Better Architecture**: Use Stable Diffusion instead of simple UNet")
    print(f"   3. **Proper Diffusion**: Implement actual diffusion process")
    print(f"   4. **Text Conditioning**: Add CLIP text encoder for better control")
    print(f"   5. **Higher Resolution**: Use 128x128 or 256x256 resolution")
    
    print(f"\n⚡ Quick Test Solutions:")
    print(f"   1. **Invert Colors**: 1 - output to get black strokes")
    print(f"   2. **Threshold**: Apply threshold to create binary images")
    print(f"   3. **Different Input**: Use structured noise instead of random")
    print(f"   4. **Post-processing**: Apply morphological operations")

def create_fixed_generation():
    """Create fixed generation with color inversion"""
    
    print(f"\n🎨 Creating Fixed Generation...")
    
    # Load model
    model_path = "quick_test_results/quick_test_epoch_2.pth"
    from quick_train_test import SimpleUNet
    model = SimpleUNet()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create different inputs
    inputs = {
        'random': torch.randn(4, 3, 64, 64),
        'structured': torch.randn(4, 3, 64, 64) * 0.5 + 0.5,  # More structured
        'low_freq': torch.randn(4, 3, 16, 16).repeat(1, 1, 4, 4),  # Low frequency
    }
    
    # Create output directory
    output_dir = Path("fixed_generation")
    output_dir.mkdir(exist_ok=True)
    
    for input_name, input_tensor in inputs.items():
        print(f"   Testing {input_name} input...")
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Apply fixes
        fixes = {
            'original': output,
            'inverted': 1 - output,  # Invert colors
            'threshold': (output > 0.5).float(),  # Binary threshold
            'enhanced': torch.clamp(output * 2 - 0.5, 0, 1),  # Enhance contrast
        }
        
        for fix_name, fixed_output in fixes.items():
            # Convert to image
            for i in range(4):
                img_array = fixed_output[i].permute(1, 2, 0).numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                filename = f"fixed_{input_name}_{fix_name}_{i+1}.png"
                img.save(output_dir / filename)
                print(f"     • Saved: {filename}")
    
    print(f"\n✅ Fixed generation completed!")
    print(f"   • Results saved in: {output_dir}")
    print(f"   • Try different fixes: original, inverted, threshold, enhanced")

def main():
    """Main function"""
    
    # Analyze model output
    output = analyze_model_output()
    
    # Analyze training data
    analyze_training_data()
    
    # Provide solutions
    provide_solutions()
    
    # Create fixed generation
    create_fixed_generation()
    
    print(f"\n🎉 Debug Analysis Complete!")
    print(f"   • Problem identified: Model outputs mostly white")
    print(f"   • Solutions provided")
    print(f"   • Fixed generation created")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Check fixed_generation/ folder for improved results")
    print(f"   2. Consider proceeding with full training")
    print(f"   3. Use better architecture (Stable Diffusion)")
    print(f"   4. Implement proper diffusion process")

if __name__ == "__main__":
    main()
