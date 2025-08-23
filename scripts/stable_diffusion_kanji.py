#!/usr/bin/env python3
"""
Complete Stable Diffusion Implementation for Kanji Generation
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
from typing import Optional, Union, Tuple
import math

class VAE(nn.Module):
    """Variational Autoencoder for image compression"""
    
    def __init__(self, in_channels=3, latent_channels=4, hidden_size=128):
        super().__init__()
        
        # Encoder - 3 downsampling steps: 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, 3, stride=2, padding=1),      # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=2, padding=1),      # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=2, padding=1),      # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(hidden_size, latent_channels, 3, padding=1),            # 16 -> 16
        )
        
        # Decoder - 3 upsampling steps: 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_size, 3, padding=1),            # 16 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, in_channels, 3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class CrossAttention(nn.Module):
    """Cross attention for text conditioning"""
    
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(context_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(context_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, query_dim)
    
    def forward(self, x, context):
        h = self.heads
        
        # Debug: print input shapes
        print(f"   🔍 CrossAttention: x={x.shape}, context={context.shape}")
        
        # Ensure context has correct dimensions
        if context.shape[-1] != self.to_k.in_features:
            print(f"   ⚠️  Context dimension mismatch: expected {self.to_k.in_features}, got {context.shape[-1]}")
            # Project context to correct dimension if needed
            if not hasattr(self, 'context_proj'):
                self.context_proj = nn.Linear(context.shape[-1], self.to_k.in_features).to(context.device)
            context = self.context_proj(context)
            print(f"   ✅ Context projected to: {context.shape}")
        
        q = self.to_q(x).view(x.shape[0], -1, h, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(context.shape[0], -1, h, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(context.shape[0], -1, h, self.dim_head).transpose(1, 2)
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], -1, h * self.dim_head)
        return self.to_out(out)

class ResBlock(nn.Module):
    """Residual block with cross attention"""
    
    def __init__(self, channels, context_dim, time_dim, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )
        
        # Main conv layers
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # Cross attention - ensure dimensions match
        self.cross_attn = CrossAttention(channels, context_dim)
        
        # Normalization
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context, time_emb):
        h = x
        
        # Time embedding - ensure correct shape and channels
        time_emb = self.time_mlp(time_emb)  # (B, channels)
        
        # Reshape time embedding to match spatial dimensions
        batch_size, channels, height, width = h.shape
        time_emb = time_emb.view(batch_size, channels, 1, 1).expand(-1, -1, height, width)
        
        # First conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + time_emb
        
        # Cross attention - reshape for attention
        batch_size, channels, height, width = h.shape
        h_flat = h.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Ensure context has correct shape
        if context.dim() == 2:
            context = context.unsqueeze(0)  # Add batch dimension if missing
        
        # Apply cross attention
        h_flat = self.cross_attn(h_flat, context)
        h = h_flat.transpose(1, 2).view(batch_size, channels, height, width)
        
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Residual connection
        return x + self.dropout(h)

class UNet2DConditionModel(nn.Module):
    """UNet with cross attention for text conditioning - optimized for 16x16 latent space"""
    
    def __init__(self, in_channels=4, context_dim=512, time_dim=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)
        
        # Downsampling - 16 -> 8 -> 4
        self.down1 = nn.ModuleList([
            ResBlock(128, context_dim, time_dim),
            ResBlock(128, context_dim, time_dim),
            nn.Conv2d(128, 256, 3, stride=2, padding=1)  # 16 -> 8
        ])
        
        self.down2 = nn.ModuleList([
            ResBlock(256, context_dim, time_dim),
            ResBlock(256, context_dim, time_dim),
            nn.Conv2d(256, 512, 3, stride=2, padding=1)  # 8 -> 4
        ])
        
        # Middle
        self.middle = nn.ModuleList([
            ResBlock(512, context_dim, time_dim),
            ResBlock(512, context_dim, time_dim)
        ])
        
        # Upsampling - 4 -> 8 -> 16
        self.up2 = nn.ModuleList([
            ResBlock(512, context_dim, time_dim),
            ResBlock(512, context_dim, time_dim),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)  # 4 -> 8
        ])
        
        self.up1 = nn.ModuleList([
            ResBlock(256, context_dim, time_dim),
            ResBlock(256, context_dim, time_dim),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  # 8 -> 16
        ])
        
        # Output
        self.conv_out = nn.Conv2d(128, in_channels, 3, padding=1)
    
    def forward(self, x, timesteps, context):
        # Debug: print input shapes
        print(f"   🔍 UNet forward: x={x.shape}, timesteps={timesteps.shape}, context={context.shape}")
        
        # Time embedding - ensure correct shape
        t = self.time_embedding(timesteps.unsqueeze(-1).float())
        print(f"   🔍 Time embedding: {t.shape}")
        
        # Ensure time embedding has correct batch dimension
        if t.dim() == 1:
            t = t.unsqueeze(0)  # Add batch dimension if missing
        
        # Ensure time embedding has correct batch size to match input
        batch_size = x.shape[0]
        if t.shape[0] != batch_size:
            # Expand time embedding to match batch size
            if t.shape[0] == 1:
                t = t.expand(batch_size, -1)
            else:
                # Repeat time embedding to match batch size
                t = t.repeat(batch_size // t.shape[0], 1)
        
        print(f"   🔍 Time embedding after reshape: {t.shape}")
        
        # Initial conv
        h = self.conv_in(x)
        print(f"   🔍 After conv_in: {h.shape}")
        
        # Downsampling
        h1 = h  # 16x16
        for i, layer in enumerate(self.down1[:-1]):
            h1 = layer(h1, context, t)
            print(f"   🔍 After down1[{i}]: {h1.shape}")
        h1 = self.down1[-1](h1)  # 8x8
        print(f"   🔍 After down1[-1]: {h1.shape}")
        
        h2 = h1  # 8x8
        for i, layer in enumerate(self.down2[:-1]):
            h2 = layer(h2, context, t)
            print(f"   🔍 After down2[{i}]: {h2.shape}")
        h2 = self.down2[-1](h2)  # 4x4
        print(f"   🔍 After down2[-1]: {h2.shape}")
        
        # Middle
        for i, layer in enumerate(self.middle):
            h2 = layer(h2, context, t)  # 4x4
            print(f"   🔍 After middle[{i}]: {h2.shape}")
        
        # Upsampling
        for i, layer in enumerate(self.up2[:-1]):
            h2 = layer(h2, context, t)
            print(f"   🔍 After up2[{i}]: {h2.shape}")
        h2 = self.up2[-1](h2)  # 8x8
        print(f"   🔍 After up2[-1]: {h2.shape}")
        
        # Skip connection - ensure dimensions match
        if h1.shape != h2.shape:
            # Resize h1 to match h2 if needed
            h1_resized = F.interpolate(h1, size=h2.shape[2:], mode='bilinear', align_corners=False)
            print(f"   🔍 Skip connection: h1 resized from {h1.shape} to {h1_resized.shape}")
        else:
            h1_resized = h1
        
        h = h1_resized + h2  # Skip connection at 8x8
        print(f"   🔍 After skip connection 1: {h.shape}")
        
        for i, layer in enumerate(self.up1[:-1]):
            h = layer(h, context, t)
            print(f"   🔍 After up1[{i}]: {h.shape}")
        h = self.up1[-1](h)  # 16x16
        print(f"   🔍 After up1[-1]: {h.shape}")
        
        # Final skip connection - ensure dimensions match
        if x.shape != h.shape:
            # Resize x to match h if needed
            x_resized = F.interpolate(x, size=h.shape[2:], mode='bilinear', align_corners=False)
            print(f"   🔍 Final skip: x resized from {x.shape} to {x_resized.shape}")
            
            # Also ensure channel dimensions match
            if x_resized.shape[1] != h.shape[1]:
                # Project x to match h's channels
                if not hasattr(self, 'final_proj'):
                    self.final_proj = nn.Conv2d(x_resized.shape[1], h.shape[1], 1).to(x_resized.device)
                x_resized = self.final_proj(x_resized)
                print(f"   🔍 Final skip: x projected to channels {x_resized.shape}")
        else:
            x_resized = x
        
        h = x_resized + h  # Skip connection at 16x16
        print(f"   🔍 After final skip: {h.shape}")
        
        # Output
        output = self.conv_out(h)
        print(f"   🔍 Final output: {output.shape}")
        return output

class DDPMScheduler:
    """DDPM noise scheduler with proper timestep management"""
    
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-calculate values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For inference
        self.timesteps = None
    
    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference"""
        self.num_inference_steps = num_inference_steps
        
        # Create timestep schedule
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).long()
        timesteps = timesteps.flip(0)  # Reverse order for denoising
        
        self.timesteps = timesteps
        print(f"   📊 Set {num_inference_steps} inference timesteps: {timesteps.tolist()}")
    
    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples according to timestep"""
        # Ensure timesteps are within bounds
        timesteps = torch.clamp(timesteps, 0, self.num_train_timesteps - 1)
        
        # Get noise schedule values for these timesteps
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        # Add noise: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        noisy_samples = sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise
        
        return noisy_samples
    
    def step(self, model_output, timestep, sample):
        """Single denoising step using predicted noise"""
        # Ensure timestep is within bounds
        if timestep >= self.num_train_timesteps:
            timestep = self.num_train_timesteps - 1
        
        # Get current alpha values
        alpha = self.alphas[timestep]
        alpha_prev = self.alphas[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        
        # Convert to same device as sample
        alpha = alpha.to(sample.device)
        alpha_prev = alpha_prev.to(sample.device)
        
        # Predicted noise from model
        predicted_noise = model_output
        
        # Denoising step: x_{t-1} = (x_t - sqrt(1-α_t) * ε_pred) / sqrt(α_t)
        x_prev = (sample - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
        
        return x_prev

class StableDiffusionPipeline:
    """Complete Stable Diffusion pipeline"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Initialize components
        self.vae = VAE().to(device)
        self.unet = UNet2DConditionModel().to(device)
        self.scheduler = DDPMScheduler()
        
        # Text encoder (simplified CLIP)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move text encoder to device
        self.text_encoder.to(device)
        
        print("✅ Stable Diffusion Pipeline initialized")
    
    def encode_text(self, text):
        """Encode text to embeddings with better error handling"""
        try:
            # Handle both single strings and lists of strings
            if isinstance(text, str):
                text = [text]
            
            # Tokenize with proper padding and truncation
            tokens = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=77,  # CLIP standard length
                return_tensors="pt"
            )
            
            # Move to device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Encode with gradient computation disabled
            with torch.no_grad():
                text_embeddings = self.text_encoder(**tokens).last_hidden_state
            
            # Ensure proper shape: (batch_size, 77, 768)
            if text_embeddings.dim() == 2:
                text_embeddings = text_embeddings.unsqueeze(0)
            
            print(f"   ✅ Text encoded: {len(text)} prompts → {text_embeddings.shape}")
            return text_embeddings
            
        except Exception as e:
            print(f"   ❌ Text encoding error: {e}")
            # Return default embedding if encoding fails
            batch_size = len(text) if isinstance(text, list) else 1
            default_embedding = torch.zeros(batch_size, 77, 768, device=self.device)
            return default_embedding
    
    def encode_image(self, image):
        """Encode image to latent space"""
        return self.vae.encode(image)
    
    def decode_latent(self, latent):
        """Decode latent to image space"""
        return self.vae.decode(latent)
    
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
        """Generate image from text prompt using proper diffusion process with enhanced capabilities"""
        print(f"🎯 Generating: '{prompt}'")
        
        # Set seed for reproducible generation
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        
        # Enhanced text prompt parsing
        parsed_prompt = self._parse_kanji_prompt(prompt)
        print(f"   📝 Parsed prompt: {parsed_prompt}")
        
        # Encode text prompt with semantic understanding
        text_embeddings = self.encode_text(parsed_prompt)
        
        # Initialize noise in latent space - match VAE output size (16x16)
        batch_size = 1
        latent_height = 16  # VAE downsampling factor: 128 -> 64 -> 32 -> 16
        latent_width = 16
        
        latents = torch.randn(
            batch_size, 4, latent_height, latent_width,
            device=self.device
        )
        
        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Enhanced denoising loop with classifier-free guidance
        for i, t in enumerate(self.scheduler.timesteps):
            # Ensure timestep is on correct device
            t = t.to(self.device)
            
            # Generate with and without text conditioning for guidance
            if guidance_scale > 1.0:
                # Unconditional generation (no text)
                noise_pred_uncond = self.unet(latents, t, None)
                
                # Conditional generation (with text)
                noise_pred_cond = self.unet(latents, t, text_embeddings)
                
                # Classifier-free guidance: interpolate between conditional and unconditional
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard conditional generation
                noise_pred = self.unet(latents, t, text_embeddings)
            
            # Denoise step using predicted noise
            latents = self.scheduler.step(noise_pred, t, latents)
            
            # Progress update
            if i % max(1, num_inference_steps // 10) == 0:
                print(f"   Step {i+1}/{num_inference_steps} (t={t.item()})")
        
        # Decode from latent space to image space
        image = self.decode_latent(latents)
        
        print(f"   ✅ Generation complete: {image.shape}")
        return image
    
    def _parse_kanji_prompt(self, prompt):
        """Enhanced prompt parsing for Kanji generation"""
        # Convert to lowercase for consistent processing
        prompt = prompt.lower()
        
        # Extract key concepts and meanings
        kanji_keywords = {
            'success': ['成功', 'achievement', 'accomplish', 'victory', 'win'],
            'failure': ['失敗', 'defeat', 'lose', 'mistake', 'error'],
            'novel': ['新穎', 'creative', 'innovative', 'original', 'unique'],
            'funny': ['面白い', 'humorous', 'amusing', 'entertaining', 'comical'],
            'culture': ['文化', 'tradition', 'heritage', 'custom', 'society'],
            'technology': ['技術', 'tech', 'digital', 'modern', 'innovation'],
            'nature': ['自然', 'nature', 'environment', 'organic', 'earth'],
            'emotion': ['感情', 'emotion', 'feeling', 'mood', 'sentiment']
        }
        
        # Enhanced prompt construction
        enhanced_prompt = prompt
        
        # Add Kanji-specific context if not present
        if 'kanji' not in prompt and '漢字' not in prompt:
            enhanced_prompt = f"kanji character meaning: {prompt}"
        
        # Add style and quality modifiers
        style_modifiers = [
            "traditional calligraphy style",
            "clean and modern design",
            "balanced stroke composition",
            "cultural authenticity"
        ]
        
        enhanced_prompt += f", {', '.join(style_modifiers)}"
        
        return enhanced_prompt
    
    def generate_concept_kanji(self, concept, style="traditional", num_variations=3):
        """Generate Kanji for specific modern concepts"""
        print(f"🎨 Generating Kanji for concept: {concept}")
        
        # Concept-specific prompt engineering
        concept_prompts = {
            'youtube': f"kanji character representing video sharing platform, streaming content, digital entertainment, {style} style",
            'gundam': f"kanji character representing giant robot mecha, futuristic warfare, technological advancement, {style} style",
            'ai': f"kanji character representing artificial intelligence, machine learning, digital consciousness, {style} style",
            'crypto': f"kanji character representing digital cryptocurrency, blockchain technology, decentralized finance, {style} style",
            'internet': f"kanji character representing global network, digital connectivity, information age, {style} style",
            'social_media': f"kanji character representing social networking, digital communication, online community, {style} style",
            'gaming': f"kanji character representing video games, interactive entertainment, digital play, {style} style",
            'streaming': f"kanji character representing live broadcasting, real-time content, digital media, {style} style"
        }
        
        if concept.lower() in concept_prompts:
            prompt = concept_prompts[concept.lower()]
        else:
            # Generic concept prompt
            prompt = f"kanji character representing {concept}, modern concept, {style} style"
        
        # Generate multiple variations
        variations = []
        for i in range(num_variations):
            seed = torch.randint(0, 1000000, (1,)).item()
            variation = self.generate(prompt, num_inference_steps=30, seed=seed)
            variations.append(variation)
            print(f"   ✅ Variation {i+1} generated with seed {seed}")
        
        return variations
    
    def semantic_interpolation(self, prompt1, prompt2, num_steps=5, interpolation_type="linear"):
        """Generate images by interpolating between two text prompts in CLIP space"""
        print(f"🔄 Semantic interpolation: '{prompt1}' → '{prompt2}'")
        
        # Encode both prompts
        embeddings1 = self.encode_text(prompt1)
        embeddings2 = self.encode_text(prompt2)
        
        # Generate interpolation weights
        if interpolation_type == "linear":
            weights = torch.linspace(0, 1, num_steps)
        elif interpolation_type == "ease_in_out":
            weights = torch.tensor([0, 0.25, 0.5, 0.75, 1.0])
        else:
            weights = torch.linspace(0, 1, num_steps)
        
        interpolated_images = []
        
        for i, weight in enumerate(weights):
            print(f"   🔄 Interpolation step {i+1}/{num_steps} (weight: {weight:.2f})")
            
            # Interpolate embeddings
            interpolated_embedding = (1 - weight) * embeddings1 + weight * embeddings2
            
            # Generate image with interpolated embedding
            # We need to modify the generation to use custom embeddings
            image = self._generate_with_custom_embedding(interpolated_embedding)
            interpolated_images.append(image)
        
        return interpolated_images
    
    def _generate_with_custom_embedding(self, custom_embedding, num_inference_steps=30):
        """Generate image using custom text embedding (for interpolation)"""
        # Initialize noise
        batch_size = 1
        latent_height = 16
        latent_width = 16
        
        latents = torch.randn(
            batch_size, 4, latent_height, latent_width,
            device=self.device
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop with custom embedding
        for t in self.scheduler.timesteps:
            t = t.to(self.device)
            
            # Predict noise using custom embedding
            noise_pred = self.unet(latents, t, custom_embedding)
            
            # Denoise step
            latents = self.scheduler.step(noise_pred, t, latents)
        
        # Decode to image space
        image = self.decode_latent(latents)
        return image
    
    def generate_creative_kanji(self, base_concept, modifiers=None, num_generations=1):
        """Generate creative Kanji with concept modifiers"""
        print(f"🎭 Generating creative Kanji for: {base_concept}")
        
        if modifiers is None:
            modifiers = ["modern", "traditional", "artistic", "minimalist"]
        
        creative_prompts = []
        for modifier in modifiers:
            prompt = f"kanji character meaning: {base_concept}, {modifier} style, creative interpretation"
            creative_prompts.append(prompt)
        
        generated_images = []
        for i, prompt in enumerate(creative_prompts):
            print(f"   🎨 Style {i+1}: {modifier}")
            image = self.generate(prompt, num_inference_steps=40, guidance_scale=8.0)
            generated_images.append(image)
        
        return generated_images
    
    def train_step(self, images, prompts, timesteps):
        """Training step"""
        # Encode images to latent space
        latents = self.encode_image(images)
        
        # Encode text
        text_embeddings = self.encode_text(prompts)
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

def create_kanji_dataset():
    """Create dataset for training"""
    class KanjiDataset(Dataset):
        def __init__(self, dataset_path, transform=None):
            self.dataset_path = Path(dataset_path)
            self.transform = transform
            
            # Load metadata
            metadata_path = self.dataset_path / "metadata" / "dataset.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            print(f"Loaded {len(self.data)} Kanji entries")
        
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
    
    return KanjiDataset

def main():
    """Main function to test the pipeline"""
    print("🎌 Stable Diffusion Kanji Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    pipeline = StableDiffusionPipeline(device=device)
    
    # Test generation
    test_prompts = [
        "kanji character meaning: success, achieve, accomplish",
        "kanji character meaning: failure, lose, defeat",
        "kanji character meaning: novel, new, creative",
        "kanji character meaning: funny, humorous, amusing",
        "kanji character meaning: culture, tradition, heritage"
    ]
    
    for prompt in test_prompts:
        try:
            generated = pipeline.generate(prompt, num_inference_steps=20)
            print(f"✅ Generated: {prompt}")
            
            # Save result
            output_path = f"generated_{prompt.split(':')[1].strip().replace(', ', '_')[:20]}.png"
            generated_image = transforms.ToPILImage()(generated.squeeze(0))
            generated_image.save(output_path)
            print(f"💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error generating '{prompt}': {e}")
    
    print("\n🎉 Pipeline test complete!")

if __name__ == "__main__":
    main()
