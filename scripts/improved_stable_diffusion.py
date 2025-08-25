#!/usr/bin/env python3
"""
改进ofStable Diffusionimplementation
借鉴官方CompVis/stable-diffusionof最佳实践
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import math
from typing import Optional, Union, Tuple
import numpy as np

class ImprovedVAE(nn.Module):
    """
    改进ofVAEimplementation，借鉴官方架构
    """
    def __init__(self, in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder - usingmore深of网络
        encoder_layers = []
        in_ch = in_channels
        for h_dim in hidden_dims:
            # 计算合适ofGroupNorm组数
            num_groups = min(32, h_dim)
            while h_dim % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups, h_dim),  # usingGroupNormreplaceBatchNorm
                nn.SiLU()  # usingSiLUreplaceLeakyReLU
            ])
            in_ch = h_dim
        
        # Final encoding layer
        final_channels = latent_channels * 2
        num_groups = min(32, final_channels)
        while final_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        encoder_layers.extend([
            nn.Conv2d(hidden_dims[-1], final_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, final_channels)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder - ensure精确of128x128output
        decoder_layers = []
        in_ch = latent_channels
        
        # usinghidden_dimsof反序进行on采样
        hidden_dims_rev = hidden_dims[::-1]
        
        for i, h_dim in enumerate(hidden_dims_rev):
            # 计算合适ofGroupNorm组数
            num_groups = min(32, h_dim)
            while h_dim % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups, h_dim),
                nn.SiLU()
            ])
            in_ch = h_dim
        
        # 最终output层
        decoder_layers.extend([
            nn.Conv2d(in_ch, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        # ensure输入是128x128
        if x.shape[-1] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        
        # 编码to潜in空间
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar, kl_loss
    
    def decode(self, z):
        return self.decoder(z)

class ImprovedCrossAttention(nn.Module):
    """
    改进of交叉注意力implementation，借鉴官方version
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        context = context if context is not None else x
        
        # 保存原始输入形状
        original_shape = x.shape
        
        # 处理4D输入 (B, C, H, W) -> (B, H*W, C)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # 处理on下文
        if context.dim() == 4:
            B, C, H, W = context.shape
            context = context.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 重塑为多头注意力
        q = q.view(q.shape[0], -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, out.shape[-1] * self.heads)
        
        # 应用output投影
        out = self.to_out(out)
        
        # convert回4D格式 (B, H*W, C) -> (B, C, H, W)
        if len(original_shape) == 4:
            B, C, H, W = original_shape
            out = out.transpose(1, 2).view(B, C, H, W)
        
        return out

class ImprovedResBlock(nn.Module):
    """
    改进of残差块，借鉴官方implementation
    """
    def __init__(self, channels, time_dim, dropout=0.0):
        super().__init__()
        
        # 动态计算GroupNormof组数，ensurechannelscan被num_groups整除
        if channels >= 32:
            num_groups = min(32, channels // (channels // 32))
        elif channels >= 16:
            num_groups = min(16, channels // (channels // 16))
        elif channels >= 8:
            num_groups = min(8, channels // (channels // 8))
        elif channels >= 4:
            num_groups = min(4, channels // (channels // 4))
        else:
            num_groups = 1
        
        # ensurenum_groupscan整除channels
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
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
        
        # 时间嵌入投影
        self.time_proj = nn.Linear(time_dim, channels)
        
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # 时间嵌入处理
        time_emb = self.time_proj(time_emb)
        time_emb = time_emb.view(x.shape[0], -1, 1, 1)
        h = h + time_emb
        
        h = self.block2(h)
        return h + x

class ImprovedUNet2DConditionModel(nn.Module):
    """
    简化ofUNetimplementation，avoid复杂of跳跃连接
    """
    def __init__(self, in_channels=4, out_channels=4, model_channels=128, num_res_blocks=2, 
                 attention_resolutions=(8, 16), dropout=0.1, channel_mult=(1, 2, 4, 8), 
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
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 编码器块
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            # ResBlock
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ImprovedResBlock(ch, time_embed_dim, dropout))
            
            # CrossAttention
            if level in attention_resolutions:
                self.encoder_blocks.append(ImprovedCrossAttention(ch, context_dim, num_heads, dropout=dropout))
            
            # 下采样
            if level < len(channel_mult) - 1:
                ch = mult * model_channels
                self.encoder_blocks.append(nn.Conv2d(self.encoder_blocks[-1].block1[0].num_features, ch, 3, stride=2, padding=1))
        
        # 中间块
        self.middle_block = nn.ModuleList([
            ImprovedResBlock(ch, time_embed_dim, dropout),
            ImprovedCrossAttention(ch, context_dim, num_heads, dropout=dropout),
            ImprovedResBlock(ch, time_embed_dim, dropout)
        ])
        
        # decode器块
        self.decoder_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # on采样
            if level < len(channel_mult) - 1:
                ch = ch // 2
                self.decoder_blocks.append(nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1))
            
            # ResBlock
            for _ in range(num_res_blocks + 1):
                self.decoder_blocks.append(ImprovedResBlock(ch, time_embed_dim, dropout))
            
            # CrossAttention
            if level in attention_resolutions:
                self.decoder_blocks.append(ImprovedCrossAttention(ch, context_dim, num_heads, dropout=dropout))
        
        # output投影
        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, timesteps, context=None):
        # 时间嵌入
        t = self.time_embedding(timesteps.unsqueeze(-1).float())
        if t.dim() == 1:
            t = t.unsqueeze(0)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 编码器
        for module in self.encoder_blocks:
            if isinstance(module, ImprovedCrossAttention):
                h = module(h, context)
            elif isinstance(module, ImprovedResBlock):
                h = module(h, t)
            else:
                h = module(h)
        
        # 中间块
        for module in self.middle_block:
            if isinstance(module, ImprovedCrossAttention):
                h = module(h, context)
            else:
                h = module(h, t)
        
        # output块
        for module in self.output_blocks:
            if isinstance(module, nn.ModuleList):
                # 处理ModuleList中of模块
                for submodule in module:
                    if isinstance(submodule, ImprovedCrossAttention):
                        h = submodule(h, context)
                    elif isinstance(submodule, ImprovedResBlock):
                        h = submodule(h, t)
                    else:
                        h = submodule(h)
            else:
                # 直接处理单个模块
                if isinstance(module, ImprovedCrossAttention):
                    h = module(h, context)
                elif isinstance(module, ImprovedResBlock):
                    h = module(h, t)
                else:
                    h = module(h)
            
            # 跳跃连接
            if hs:
                skip_h = hs.pop()
                # 简单of跳跃连接，不进行通道调整
                h = torch.cat([h, skip_h], dim=1)
        
        return self.out(h)

class ImprovedDDPMScheduler:
    """
    改进ofDDPMscheduling器，借鉴官方implementation
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # 线性noisescheduling
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 计算noisepredictionofcoefficient
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, original_samples, noise, timesteps):
        """addnoiseto原始样本"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise
    
    def step(self, model_output, timestep, sample):
        """去噪步骤"""
        alpha = self.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod_prev[timestep].view(-1, 1, 1, 1)
        
        # predictionx0
        pred_original_sample = (sample - torch.sqrt(1 - alpha) * model_output) / torch.sqrt(alpha)
        
        # prediction前a样本
        pred_sample_direction = torch.sqrt(1 - alpha_prev) * model_output
        pred_prev_sample = torch.sqrt(alpha_prev) * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample
    
    def set_timesteps(self, num_inference_steps):
        """set推理时间步"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).flip(0)
        return timesteps

class ImprovedStableDiffusionPipeline:
    """
    改进ofStable Diffusion Pipeline，借鉴官方最佳实践
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # initialization组件
        self.vae = ImprovedVAE().to(device)
        self.unet = ImprovedUNet2DConditionModel(
            in_channels=4,
            out_channels=4,
            model_channels=128,
            channel_mult=(1, 2, 4, 8),
            attention_resolutions=(8, 16),
            context_dim=512
        ).to(device)
        self.scheduler = ImprovedDDPMScheduler()
        
        # CLIP文本编码器
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        # set为评估模式
        self.text_encoder.eval()
        self.vae.eval()
        
    def _encode_prompt(self, prompt):
        """编码文本提示"""
        tokens = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
        return text_embeddings
    
    def _parse_kanji_prompt(self, prompt):
        """解析汉字提示，usingmore详细of描述"""
        base_prompt = f"kanji character representing {prompt}, traditional calligraphy style, black ink on white paper, high contrast, detailed strokes, clear lines, professional quality, artistic interpretation"
        return base_prompt
    
    def generate(self, prompt, height=128, width=128, num_inference_steps=50, 
                guidance_scale=7.5, seed=None):
        """generationimage，using官方推荐of参数"""
        
        # setrandom seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # 编码提示
        text_embeddings = self._encode_prompt(self._parse_kanji_prompt(prompt))
        
        # initialization潜in变量
        latent_height = height // 8
        latent_width = width // 8
        latents = torch.randn(1, 4, latent_height, latent_width, device=self.device)
        
        # set时间步
        timesteps = self.scheduler.set_timesteps(num_inference_steps)
        timesteps = timesteps.to(self.device)
        
        # 改进of去噪循环
        for i, t in enumerate(timesteps):
            # 扩展潜in变量用于批处理
            latent_model_input = torch.cat([latents] * 2)
            t_expanded = t.expand(2)
            
            # predictionnoise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t_expanded, text_embeddings)
            
            # executedclassifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # using官方推荐ofguidance scale
            guidance_scale = torch.clamp(torch.tensor(guidance_scale), min=1.0, max=20.0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 计算前a样本
            latents = self.scheduler.step(noise_pred, t, latents)
        
        # decode潜in变量
        with torch.no_grad():
            image = self.vae.decode(latents)
        
        return image

if __name__ == "__main__":
    print("🎌 改进的Stable Diffusion实现")
    print("=" * 50)
    
    # testmodel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    try:
        pipeline = ImprovedStableDiffusionPipeline(device=device)
        print("✅ 模型初始化成功")
        
        # testgeneration
        print("🌊 测试生成...")
        result = pipeline.generate(
            "water",
            height=128,
            width=128,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42
        )
        print("✅ 生成完成")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
