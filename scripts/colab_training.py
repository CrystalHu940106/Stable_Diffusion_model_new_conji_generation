#!/usr/bin/env python3
"""
Google Colab优化的Stable Diffusion训练脚本
专门为Colab GPU环境优化，包含自动检测和性能优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gc

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_stable_diffusion import (
    ImprovedStableDiffusionPipeline,
    ImprovedVAE,
    ImprovedUNet2DConditionModel,
    ImprovedDDPMScheduler
)

class ColabOptimizedTrainer:
    """
    Colab优化的训练器
    """
    def __init__(self, device='auto'):
        # 自动检测设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 检测到CUDA设备: {torch.cuda.get_device_name()}")
                print(f"   • GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                print(f"   • CUDA版本: {torch.version.cuda}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                print("🍎 检测到Apple Silicon (MPS)")
            else:
                self.device = 'cpu'
                print("💻 使用CPU训练")
        else:
            self.device = device
        
        # 初始化模型
        self.vae = ImprovedVAE().to(self.device)
        self.unet = ImprovedUNet2DConditionModel(
            in_channels=4,
            out_channels=4,
            model_channels=128,
            channel_mult=(1, 2, 4, 8),
            attention_resolutions=(8, 16),
            context_dim=512
        ).to(self.device)
        self.scheduler = ImprovedDDPMScheduler()
        
        # 优化器设置
        self.optimizer = optim.AdamW([
            {'params': self.vae.parameters(), 'lr': 1e-4},
            {'params': self.unet.parameters(), 'lr': 1e-4}
        ], weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 训练参数
        self.num_epochs = 50
        self.batch_size = 8  # Colab GPU内存优化
        self.gradient_accumulation_steps = 4
        self.save_every = 5
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        print(f"✅ 模型初始化完成，使用设备: {self.device}")
    
    def create_synthetic_dataset(self, num_samples=1000):
        """
        创建合成数据集用于演示
        在实际使用中，这里应该加载真实的汉字数据
        """
        print(f"📊 创建合成数据集 ({num_samples} 样本)...")
        
        # 创建128x128的合成图像
        images = []
        for i in range(num_samples):
            # 创建简单的几何图案作为训练数据
            img = np.zeros((128, 128, 3), dtype=np.float32)
            
            # 添加一些随机几何形状
            if i % 4 == 0:
                # 圆形
                y, x = np.ogrid[:128, :128]
                mask = (x - 64)**2 + (y - 64)**2 <= 30**2
                img[mask] = [0.8, 0.8, 0.8]
            elif i % 4 == 1:
                # 矩形
                img[40:88, 40:88] = [0.7, 0.7, 0.7]
            elif i % 4 == 2:
                # 三角形
                for y in range(128):
                    for x in range(128):
                        if y >= 64 and abs(x - 64) <= (y - 64):
                            img[y, x] = [0.6, 0.6, 0.6]
            else:
                # 随机噪声
                img = np.random.rand(128, 128, 3).astype(np.float32) * 0.5
            
            # 归一化到[-1, 1]
            img = (img - 0.5) * 2
            images.append(img)
        
        # 转换为tensor
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        print(f"✅ 数据集创建完成: {images.shape}")
        
        return images
    
    def train_epoch(self, dataloader, epoch):
        """
        训练一个epoch
        """
        self.vae.train()
        self.unet.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(self.device)
            
            # 梯度累积
            with autocast():
                # VAE编码
                latents, mu, logvar, kl_loss = self.vae.encode(images)
                
                # 添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, 
                                       (latents.shape[0],), device=self.device)
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # UNet预测噪声
                noise_pred = self.unet(noisy_latents, timesteps)
                
                # 计算损失
                noise_loss = self.mse_loss(noise_pred, noise)
                reconstruction_loss = self.mse_loss(self.vae.decode(latents), images)
                
                # 总损失
                loss = noise_loss + 0.1 * kl_loss + 0.1 * reconstruction_loss
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.vae.parameters()) + list(self.unet.parameters()), 
                    max_norm=1.0
                )
                
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # 进度显示
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.num_epochs}, "
                      f"Batch {batch_idx+1}/{num_batches}, "
                      f"Loss: {loss.item():.6f}")
        
        # 学习率调度
        self.scheduler_lr.step()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, loss, save_dir="colab_checkpoints"):
        """
        保存检查点
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler_lr.state_dict(),
            'loss': loss,
            'device': self.device
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if epoch == 0 or loss < getattr(self, 'best_loss', float('inf')):
            self.best_loss = loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"🏆 最佳模型已保存: {best_model_path}")
    
    def train(self):
        """
        主训练循环
        """
        print(f"\n🎯 开始训练...")
        print(f"   • 设备: {self.device}")
        print(f"   • 批次大小: {self.batch_size}")
        print(f"   • 梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"   • 总epochs: {self.num_epochs}")
        print(f"   • 混合精度: {'启用' if self.device == 'cuda' else '禁用'}")
        
        # 创建数据集
        images = self.create_synthetic_dataset()
        dataloader = DataLoader(images, batch_size=self.batch_size, shuffle=True)
        
        # 训练历史
        train_losses = []
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                epoch_start = time.time()
                
                print(f"\n🔄 Epoch {epoch+1}/{self.num_epochs}")
                print("-" * 50)
                
                # 训练
                loss = self.train_epoch(dataloader, epoch)
                train_losses.append(loss)
                
                epoch_time = time.time() - epoch_start
                print(f"   ⏱️  Epoch耗时: {epoch_time:.2f}秒")
                print(f"   📊 平均损失: {loss:.6f}")
                print(f"   📈 学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # 保存检查点
                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(epoch, loss)
                
                # 内存清理 (Colab优化)
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 显示GPU内存使用情况
                if self.device == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"   🧠 GPU内存: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        
        except KeyboardInterrupt:
            print(f"\n⚠️  训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 保存最终模型
            final_loss = train_losses[-1] if train_losses else float('inf')
            self.save_checkpoint(len(train_losses) - 1, final_loss)
            
            # 训练总结
            total_time = time.time() - start_time
            print(f"\n🎉 训练完成!")
            print(f"   ⏱️  总耗时: {total_time:.2f}秒")
            print(f"   📊 最终损失: {final_loss:.6f}")
            print(f"   📈 损失变化: {train_losses[0]:.6f} → {final_loss:.6f}")
            
            # 绘制损失曲线
            self.plot_training_curve(train_losses)
    
    def plot_training_curve(self, losses):
        """
        绘制训练损失曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2, label='训练损失')
        plt.title('Colab训练损失曲线', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('损失', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图片
        plot_path = 'colab_training_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 训练曲线已保存: {plot_path}")
        plt.show()
    
    def test_generation(self, prompt="water"):
        """
        测试生成功能
        """
        print(f"\n🧪 测试生成: {prompt}")
        
        try:
            # 创建pipeline
            pipeline = ImprovedStableDiffusionPipeline(device=self.device)
            
            # 加载训练好的权重
            if hasattr(self, 'best_loss'):
                checkpoint_path = 'colab_checkpoints/best_model.pth'
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    pipeline.vae.load_state_dict(checkpoint['vae_state_dict'])
                    pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
                    print(f"✅ 已加载最佳模型权重")
            
            # 生成图像
            print(f"🌊 生成中...")
            result = pipeline.generate(
                prompt,
                height=128,
                width=128,
                num_inference_steps=50,
                guidance_scale=7.5,
                seed=42
            )
            
            # 保存结果
            if isinstance(result, torch.Tensor):
                result = (result + 1) / 2
                result = torch.clamp(result, 0, 1)
                img_array = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
                pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
            else:
                pil_image = result
            
            output_path = f'colab_generated_{prompt}.png'
            pil_image.save(output_path)
            print(f"✅ 生成完成，已保存: {output_path}")
            
            # 显示图像
            plt.figure(figsize=(6, 6))
            plt.imshow(pil_image, cmap='gray')
            plt.title(f'Colab生成: {prompt}', fontsize=14)
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"❌ 生成测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    主函数
    """
    print("🚀 Google Colab优化的Stable Diffusion训练器")
    print("=" * 60)
    
    # 检查Colab环境
    is_colab = 'COLAB_GPU' in os.environ
    if is_colab:
        print("✅ 检测到Google Colab环境")
        print(f"   • GPU类型: {os.environ.get('COLAB_GPU', 'Unknown')}")
        print(f"   • 运行时类型: {os.environ.get('COLAB_RUNTIME_TYPE', 'Unknown')}")
    else:
        print("💻 本地环境运行")
    
    # 创建训练器
    trainer = ColabOptimizedTrainer(device='auto')
    
    # 开始训练
    trainer.train()
    
    # 测试生成
    trainer.test_generation("water")

if __name__ == "__main__":
    main()
