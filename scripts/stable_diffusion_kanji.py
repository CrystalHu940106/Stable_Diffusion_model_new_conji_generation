#!/usr/bin/env python3
"""
Complete Stable Diffusion Implementation for Kanji Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import math
from typing import Optional, Union, Tuple
import numpy as np

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder
        encoder_layers = []
        in_ch = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_ch = h_dim
        
        # Final encoding layer
        encoder_layers.extend([
            nn.Conv2d(hidden_dims[-1], latent_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_channels * 2)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder - ensure exact 128x128 output
        decoder_layers = []
        in_ch = latent_channels
        
        # We need exactly 4 upsampling layers: 8->16->32->64->128
        # Use the hidden_dims in reverse order for channels
        hidden_dims_rev = hidden_dims[::-1]
        
        for i, h_dim in enumerate(hidden_dims_rev):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_ch = h_dim
        
        # Final layer to get to 3 channels
        decoder_layers.extend([
            nn.Conv2d(in_ch, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Pre-define projection layers for different input sizes
        self.input_projections = nn.ModuleDict({
            '128': nn.Conv2d(3, 3, 1),
            '64': nn.Conv2d(3, 3, 1),
            '32': nn.Conv2d(3, 3, 1)
        })
        
    def encode(self, x):
        # Ensure input is 128x128
        if x.shape[-1] != 128:
            target_size = 128
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
            if str(target_size) in self.input_projections:
                x = self.input_projections[str(target_size)](x)
        
        # Encode to latent space
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar, kl_loss
    
    def decode(self, z):
        return self.decoder(z)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        # Pre-define all projection layers
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        h = self.heads
        
        if context is None:
            context = x
        
        # Reshape x for attention: (B, C, H, W) -> (B, H*W, C)
        batch_size, channels, height, width = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply attention
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.reshape(*t.shape[:2], h, -1).transpose(1, 2), (q, k, v))
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(*x_flat.shape[:2], -1)
        
        # Apply output projection
        out = self.to_out(out)
        
        # Reshape back to spatial dimensions: (B, H*W, C) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        
        return out

class ResBlock(nn.Module):
    def __init__(self, channels, time_dim, dropout=0.1):
        super().__init__()
        self.channels = channels
        
        # Calculate appropriate group size for GroupNorm
        # GroupNorm requires that the number of groups divides the number of channels
        if channels >= 32:
            num_groups = 32
        elif channels >= 16:
            num_groups = 16
        elif channels >= 8:
            num_groups = 8
        elif channels >= 4:
            num_groups = 4
        else:
            num_groups = 1
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        # Pre-define time embedding projection
        self.time_proj = nn.Linear(time_dim, channels)
        
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Use pre-defined time projection
        time_emb = self.time_proj(time_emb)
        time_emb = time_emb.view(x.shape[0], -1, 1, 1)
        h = h + time_emb
        
        h = self.block2(h)
        return h + x

class UNet2DConditionModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, model_channels=128, num_res_blocks=2, 
                 attention_resolutions=(8, 16), dropout=0.1, channel_mult=(1, 2, 4), 
                 conv_resample=True, num_heads=8, context_dim=512):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.context_dim = context_dim
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling blocks - simplified architecture
        input_block_chans = [model_channels]
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            # Add ResBlock
            self.input_blocks.append(
                nn.ModuleList([ResBlock(ch, time_embed_dim, dropout)])
            )
            input_block_chans.append(ch)
            
            # Add CrossAttention if at attention resolution
            if level in attention_resolutions:
                self.input_blocks.append(
                    nn.ModuleList([CrossAttention(ch, context_dim, num_heads)])
                )
                input_block_chans.append(ch)
            
            # Downsample
            if level < len(channel_mult) - 1:
                ch = mult * model_channels
                self.input_blocks.append(
                    nn.ModuleList([nn.Conv2d(input_block_chans[-1], ch, 3, stride=2, padding=1)])
                )
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim, dropout),
            CrossAttention(ch, context_dim, num_heads),
            ResBlock(ch, time_embed_dim, dropout)
        ])
        
        # Output blocks - simplified
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # Upsample
            if level < len(channel_mult) - 1:
                self.output_blocks.append(
                    nn.ModuleList([nn.ConvTranspose2d(ch, ch//2, 4, stride=2, padding=1)])
                )
                ch = ch // 2
            
            # Add ResBlock
            self.output_blocks.append(
                nn.ModuleList([ResBlock(ch, time_embed_dim, dropout)])
            )
            
            # Add CrossAttention if at attention resolution
            if level in attention_resolutions:
                self.output_blocks.append(
                    nn.ModuleList([CrossAttention(ch, context_dim, num_heads)])
                )
        
        # Output projection
        # Calculate appropriate group size for GroupNorm
        if ch >= 32:
            num_groups = 32
        elif ch >= 16:
            num_groups = 16
        elif ch >= 8:
            num_groups = 8
        elif ch >= 4:
            num_groups = 4
        else:
            num_groups = 1
        
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, timesteps, context=None):
        # Time embedding
        t = self.time_embedding(timesteps.unsqueeze(-1).float())
        if t.dim() == 1:
            t = t.unsqueeze(0)
        
        # Ensure time embedding matches batch size
        batch_size = x.shape[0]
        if t.shape[0] != batch_size:
            if t.shape[0] == 1:
                t = t.expand(batch_size, -1)
            else:
                t = t.repeat(batch_size // t.shape[0], 1)
        
        # Input blocks
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t)
                    elif isinstance(layer, CrossAttention):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            else:
                h = module(h)
        
        # Middle block
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, t)
            elif isinstance(module, CrossAttention):
                h = module(h, context)
            else:
                h = module(h)
        
        # Output blocks
        for module in self.output_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t)
                    elif isinstance(layer, CrossAttention):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            else:
                h = module(h)
        
        return self.out(h)

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Pre-compute values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
    def add_noise(self, original_samples, noise, timesteps):
        # Ensure timesteps are within bounds
        timesteps = torch.clamp(timesteps, 0, self.num_train_timesteps - 1)
        
        # Move timesteps to CPU for indexing, then back to device
        device = timesteps.device
        timesteps_cpu = timesteps.cpu()
        
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps_cpu].view(-1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].view(-1, 1, 1, 1).to(device)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise
    
    def step(self, model_output, timestep, sample):
        # Ensure timestep is within bounds
        timestep = torch.clamp(timestep, 0, self.num_train_timesteps - 1)
        
        # Move timestep to CPU for indexing, then back to device
        device = timestep.device
        timestep_cpu = timestep.cpu()
        
        # Simple DDPM step
        alpha = self.alphas_cumprod[timestep_cpu].to(device)
        alpha_prev = self.alphas_cumprod_prev[timestep_cpu].to(device)
        
        # Ensure alpha tensors have correct shape
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
        if alpha_prev.dim() == 0:
            alpha_prev = alpha_prev.unsqueeze(0)
        
        # Ensure alpha tensors have correct spatial dimensions
        alpha = alpha.view(-1, 1, 1, 1)
        alpha_prev = alpha_prev.view(-1, 1, 1, 1)
        
        # Predict x0
        pred_original_sample = (sample - torch.sqrt(1 - alpha) * model_output) / torch.sqrt(alpha)
        
        # Predict previous sample
        pred_sample_direction = torch.sqrt(1 - alpha_prev) * model_output
        pred_prev_sample = torch.sqrt(alpha_prev) * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample
    
    @property
    def timesteps(self):
        return torch.arange(self.num_train_timesteps)
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).flip(0)
        return timesteps

class StableDiffusionPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize components
        self.vae = VAE().to(device)
        self.unet = UNet2DConditionModel().to(device)
        self.scheduler = DDPMScheduler()
        
        # CLIP text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.vae.eval()
        
    def _encode_prompt(self, prompt):
        # Tokenize and encode text
        tokens = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
        return text_embeddings
    
    def _parse_kanji_prompt(self, prompt):
        # Enhanced prompt parsing for Kanji generation
        base_prompt = f"kanji character representing {prompt}, traditional calligraphy style, black ink on white paper, high contrast, detailed strokes"
        return base_prompt
    
    def generate(self, prompt, height=128, width=128, num_inference_steps=50, 
                guidance_scale=9.0, seed=None):
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Encode prompt
        text_embeddings = self._encode_prompt(self._parse_kanji_prompt(prompt))
        
        # Initialize latents
        latent_height = height // 8
        latent_width = width // 8
        latents = torch.randn(1, 4, latent_height, latent_width, device=self.device)
        
        # Set timesteps
        timesteps = self.scheduler.set_timesteps(num_inference_steps)
        timesteps = timesteps.to(self.device)
        
        # Enhanced denoising loop with better guidance
        for i, t in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = torch.cat([latents] * 2)
            t_expanded = t.expand(2)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t_expanded, text_embeddings)
            
            # Perform enhanced guidance with optimal scale
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # Clamp guidance scale for stability
            guidance_scale = torch.clamp(torch.tensor(guidance_scale), min=1.0, max=15.0)
            
            # Enhanced classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents)
            
            # Optional: Add noise for better exploration
            if i < len(timesteps) - 1:
                next_timestep = timesteps[i + 1]
                latents = self.scheduler.add_noise(latents, torch.randn_like(latents), next_timestep)
        
        # Decode latents
        with torch.no_grad():
            image = self.vae.decode(latents)
        
        return image
    
    def generate_concept_kanji(self, concept, style="traditional", guidance_scale=9.0):
        # Generate Kanji for modern concepts with optimized guidance
        style_prompts = {
            "traditional": "traditional calligraphy, brush strokes, ink wash, authentic",
            "modern": "modern typography, clean lines, minimalist, contemporary",
            "artistic": "artistic interpretation, creative design, unique style, expressive",
            "professional": "professional design, corporate style, clean and clear",
            "creative": "creative interpretation, innovative design, artistic flair"
        }
        
        style_desc = style_prompts.get(style, style_prompts["traditional"])
        prompt = f"kanji character for {concept}, {style_desc}, high quality, detailed, well-balanced composition"
        
        return self.generate(prompt, guidance_scale=guidance_scale)
    
    def semantic_interpolation(self, prompt1, prompt2, num_steps=5, guidance_scale=9.0):
        # Generate images by interpolating between two prompts with enhanced quality
        embeddings1 = self._encode_prompt(prompt1)
        embeddings2 = self._encode_prompt(prompt2)
        
        images = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated_embeddings = (1 - alpha) * embeddings1 + alpha * embeddings2
            
            # Generate with interpolated embeddings and optimal guidance
            image = self._generate_with_custom_embedding(interpolated_embeddings, guidance_scale)
            images.append(image)
        
        return images
    
    def _generate_with_custom_embedding(self, text_embeddings, guidance_scale=9.0):
        # Helper method for custom embeddings with enhanced quality
        latents = torch.randn(1, 4, 16, 16, device=self.device)
        
        # Set timesteps for generation
        num_inference_steps = 50
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            with torch.no_grad():
                # Generate with and without conditioning
                latent_model_input = torch.cat([latents] * 2)
                t_expanded = t.expand(2)
                
                noise_pred = self.unet(latent_model_input, t_expanded, text_embeddings)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                
                # Apply guidance
                guidance_scale = torch.clamp(torch.tensor(guidance_scale), min=1.0, max=15.0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Denoise step
                latents = self.scheduler.step(noise_pred, t, latents)
        
        return self.vae.decode(latents)
    
    def generate_creative_kanji(self, base_concept, modifiers=None, guidance_scale=10.0):
        # Generate creative Kanji with concept modifiers and enhanced quality
        if modifiers is None:
            modifiers = ["elegant", "powerful", "mystical", "balanced", "harmonious"]
        
        prompt = f"kanji character for {base_concept}, {' '.join(modifiers)}, artistic interpretation, high quality, detailed strokes"
        return self.generate(prompt, guidance_scale=guidance_scale)
    
    def generate_high_quality_kanji(self, prompt, num_variations=3, guidance_scales=[7.0, 9.0, 12.0]):
        # Generate multiple high-quality variations with different guidance scales
        variations = []
        
        for i in range(num_variations):
            seed = torch.randint(0, 1000000, (1,)).item()
            guidance_scale = guidance_scales[i % len(guidance_scales)]
            
            variation = self.generate(
                prompt, 
                num_inference_steps=75,  # More steps for higher quality
                guidance_scale=guidance_scale,
                seed=seed
            )
            variations.append({
                'image': variation,
                'seed': seed,
                'guidance_scale': guidance_scale
            })
        
        return variations
