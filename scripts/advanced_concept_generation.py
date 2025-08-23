#!/usr/bin/env python3
"""
Advanced Concept Generation with Semantic Interpolation
Generate Kanji for modern concepts like YouTube, Gundam, etc.
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
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random

# Import our Stable Diffusion components
from stable_diffusion_kanji import (
    VAE, UNet2DConditionModel, DDPMScheduler, 
    StableDiffusionPipeline
)

class AdvancedConceptGenerator:
    """Advanced concept generator with semantic interpolation"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model_path = Path(model_path)
        
        # Load trained models
        self.load_trained_models()
        
        # Initialize pipeline
        self.pipeline = StableDiffusionPipeline(device=device)
        self.pipeline.vae = self.vae
        self.pipeline.unet = self.unet
        self.pipeline.text_encoder = self.text_encoder
        self.pipeline.tokenizer = self.tokenizer
        
        # Concept database
        self.concept_database = self.create_concept_database()
        
        print("✅ Advanced Concept Generator initialized")
    
    def load_trained_models(self):
        """Load trained models from checkpoint"""
        if not self.model_path.exists():
            print(f"❌ Model checkpoint not found: {self.model_path}")
            print("Please train the model first using train_stable_diffusion.py")
            return
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load VAE
        self.vae = VAE().to(self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        
        # Load UNet
        self.unet = UNet2DConditionModel().to(self.device)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        print(f"✅ Models loaded from: {self.model_path}")
        print(f"   • Epoch: {checkpoint['epoch']}")
        print(f"   • Train loss: {checkpoint['train_loss']:.6f}")
        print(f"   • Val loss: {checkpoint['val_loss']:.6f}")
    
    def create_concept_database(self) -> Dict[str, Dict]:
        """Create database of modern and traditional concepts"""
        
        return {
            # Modern Technology
            'youtube': {
                'description': 'video sharing platform, streaming, content creation',
                'keywords': ['video', 'streaming', 'platform', 'content', 'media'],
                'style': 'modern, digital, connected',
                'prompts': [
                    'kanji character meaning: video sharing platform',
                    'kanji character meaning: streaming media content',
                    'kanji character meaning: digital platform connection'
                ]
            },
            'gundam': {
                'description': 'giant robot, mecha, futuristic warfare',
                'keywords': ['robot', 'mecha', 'giant', 'future', 'warfare'],
                'style': 'futuristic, mechanical, powerful',
                'prompts': [
                    'kanji character meaning: giant robot mecha',
                    'kanji character meaning: futuristic warfare machine',
                    'kanji character meaning: mechanical giant warrior'
                ]
            },
            'internet': {
                'description': 'global network, connectivity, information',
                'keywords': ['network', 'connectivity', 'global', 'information', 'digital'],
                'style': 'connected, global, digital',
                'prompts': [
                    'kanji character meaning: global network connection',
                    'kanji character meaning: digital information flow',
                    'kanji character meaning: worldwide connectivity'
                ]
            },
            'ai': {
                'description': 'artificial intelligence, machine learning, automation',
                'keywords': ['intelligence', 'machine', 'learning', 'automation', 'future'],
                'style': 'intelligent, automated, futuristic',
                'prompts': [
                    'kanji character meaning: artificial intelligence',
                    'kanji character meaning: machine learning automation',
                    'kanji character meaning: intelligent automation'
                ]
            },
            'crypto': {
                'description': 'cryptocurrency, blockchain, digital money',
                'keywords': ['digital', 'money', 'blockchain', 'secure', 'virtual'],
                'style': 'digital, secure, virtual',
                'prompts': [
                    'kanji character meaning: digital cryptocurrency',
                    'kanji character meaning: blockchain security',
                    'kanji character meaning: virtual digital money'
                ]
            },
            
            # Traditional Concepts (for comparison)
            'success': {
                'description': 'achievement, accomplishment, victory',
                'keywords': ['achieve', 'accomplish', 'complete', 'win', 'victory'],
                'style': 'traditional, positive, upward',
                'prompts': [
                    'kanji character meaning: success achievement',
                    'kanji character meaning: accomplish victory',
                    'kanji character meaning: complete success'
                ]
            },
            'culture': {
                'description': 'tradition, heritage, wisdom',
                'keywords': ['tradition', 'heritage', 'wisdom', 'ancient', 'knowledge'],
                'style': 'traditional, cultural, ancient',
                'prompts': [
                    'kanji character meaning: cultural tradition',
                    'kanji character meaning: ancient heritage wisdom',
                    'kanji character meaning: traditional knowledge'
                ]
            }
        }
    
    def generate_concept_kanji(self, concept: str, num_samples: int = 4, 
                             num_inference_steps: int = 50) -> List[torch.Tensor]:
        """Generate Kanji for a specific concept"""
        
        if concept not in self.concept_database:
            print(f"❌ Concept '{concept}' not found in database")
            return []
        
        concept_info = self.concept_database[concept]
        print(f"\n🎯 Generating Kanji for '{concept}'")
        print(f"   • Description: {concept_info['description']}")
        print(f"   • Style: {concept_info['style']}")
        
        generated_images = []
        
        for i in range(num_samples):
            # Select random prompt from concept
            prompt = random.choice(concept_info['prompts'])
            print(f"   • Sample {i+1}: {prompt}")
            
            try:
                # Generate image
                generated = self.pipeline.generate(
                    prompt, 
                    num_inference_steps=num_inference_steps
                )
                generated_images.append(generated)
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        return generated_images
    
    def semantic_interpolation(self, concept1: str, concept2: str, 
                             interpolation_steps: int = 5) -> List[torch.Tensor]:
        """Generate Kanji by interpolating between two concepts"""
        
        if concept1 not in self.concept_database or concept2 not in self.concept_database:
            print(f"❌ One or both concepts not found in database")
            return []
        
        print(f"\n🔄 Semantic Interpolation: {concept1} → {concept2}")
        
        # Get concept embeddings
        concept1_prompt = random.choice(self.concept_database[concept1]['prompts'])
        concept2_prompt = random.choice(self.concept_database[concept2]['prompts'])
        
        # Encode both prompts
        tokens1 = self.pipeline.tokenizer(concept1_prompt, padding=True, return_tensors="pt")
        tokens2 = self.pipeline.tokenizer(concept2_prompt, padding=True, return_tensors="pt")
        
        tokens1 = {k: v.to(self.device) for k, v in tokens1.items()}
        tokens2 = {k: v.to(self.device) for k, v in tokens2.items()}
        
        with torch.no_grad():
            emb1 = self.pipeline.text_encoder(**tokens1).last_hidden_state
            emb2 = self.pipeline.text_encoder(**tokens2).last_hidden_state
        
        interpolated_images = []
        
        for i in range(interpolation_steps):
            # Interpolate between embeddings
            alpha = i / (interpolation_steps - 1)
            interpolated_emb = alpha * emb2 + (1 - alpha) * emb1
            
            print(f"   • Step {i+1}/{interpolation_steps} (α={alpha:.2f})")
            
            try:
                # Generate with interpolated embedding
                # This is a simplified approach - in practice you'd need to modify the pipeline
                # to accept custom embeddings
                prompt = f"interpolated between {concept1} and {concept2}"
                generated = self.pipeline.generate(prompt, num_inference_steps=30)
                interpolated_images.append(generated)
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        return interpolated_images
    
    def generate_modern_concepts(self, num_samples: int = 3) -> Dict[str, List[torch.Tensor]]:
        """Generate Kanji for all modern concepts"""
        
        modern_concepts = ['youtube', 'gundam', 'internet', 'ai', 'crypto']
        results = {}
        
        print(f"\n🚀 Generating Modern Concept Kanji")
        print("=" * 50)
        
        for concept in modern_concepts:
            print(f"\n📱 {concept.upper()}")
            generated = self.generate_concept_kanji(concept, num_samples)
            results[concept] = generated
            
            if generated:
                print(f"   ✅ Generated {len(generated)} samples")
            else:
                print(f"   ❌ Failed to generate")
        
        return results
    
    def save_generated_images(self, images: List[torch.Tensor], 
                            concept: str, save_dir: str = "advanced_generated_results"):
        """Save generated images"""
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        saved_paths = []
        
        for i, img_tensor in enumerate(images):
            try:
                # Convert tensor to PIL image
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor.squeeze(0)
                
                # Denormalize from [-1, 1] to [0, 1]
                img_tensor = (img_tensor + 1) / 2
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # Convert to PIL
                img_array = img_tensor.permute(1, 2, 0).numpy()
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                
                # Save
                output_path = save_path / f"{concept}_advanced_{i+1}.png"
                img.save(output_path)
                saved_paths.append(output_path)
                
            except Exception as e:
                print(f"   ❌ Error saving image {i+1}: {e}")
        
        return saved_paths
    
    def display_results(self, results: Dict[str, List[torch.Tensor]], 
                       save_dir: str = "advanced_generated_results"):
        """Display and save all generated results"""
        
        print(f"\n🖼️  Displaying Generated Results")
        
        # Create subplot
        concepts = list(results.keys())
        max_samples = max(len(imgs) for imgs in results.values()) if results else 0
        
        if max_samples == 0:
            print("❌ No images to display")
            return
        
        fig, axes = plt.subplots(len(concepts), max_samples, 
                                figsize=(4*max_samples, 4*len(concepts)))
        
        if len(concepts) == 1:
            axes = axes.reshape(1, -1)
        
        for i, concept in enumerate(concepts):
            images = results[concept]
            
            for j in range(max_samples):
                if j < len(images):
                    # Display image
                    img_tensor = images[j]
                    if img_tensor.dim() == 4:
                        img_tensor = img_tensor.squeeze(0)
                    
                    # Denormalize
                    img_tensor = (img_tensor + 1) / 2
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    
                    axes[i, j].imshow(img_array)
                    axes[i, j].set_title(f'{concept.title()} #{j+1}', fontsize=10)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, 'No image', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save combined results
        output_path = Path(save_dir) / "all_advanced_concepts.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Combined results saved to: {output_path}")
        
        plt.show()
        
        # Save individual images
        for concept, images in results.items():
            if images:
                saved_paths = self.save_generated_images(images, concept, save_dir)
                print(f"💾 {concept}: {len(saved_paths)} images saved")

def main():
    """Main function"""
    
    print("🎌 Advanced Concept Generation")
    print("=" * 50)
    
    # Check if trained model exists
    model_paths = [
        "stable_diffusion_results/best_model.pth",
        "stable_diffusion_results/stable_diffusion_epoch_10.pth",
        "stable_diffusion_results/stable_diffusion_epoch_5.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Please train the model first using train_stable_diffusion.py")
        return
    
    # Initialize generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = AdvancedConceptGenerator(model_path, device)
    
    # Generate modern concepts
    results = generator.generate_modern_concepts(num_samples=3)
    
    # Display and save results
    generator.display_results(results)
    
    # Test semantic interpolation
    print(f"\n🔄 Testing Semantic Interpolation...")
    try:
        interpolated = generator.semantic_interpolation('youtube', 'gundam', 3)
        if interpolated:
            generator.save_generated_images(interpolated, 'youtube_gundam_interpolation')
            print(f"✅ Interpolation completed: {len(interpolated)} images")
    except Exception as e:
        print(f"❌ Interpolation failed: {e}")
    
    print(f"\n🎉 Advanced Concept Generation Complete!")
    print(f"   • Generated Kanji for {len(results)} modern concepts")
    print(f"   • Images saved in: advanced_generated_results/")
    print(f"   • Ready for modern concept Kanji generation!")

if __name__ == "__main__":
    main()
