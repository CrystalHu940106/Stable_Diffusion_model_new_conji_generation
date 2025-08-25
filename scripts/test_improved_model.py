#!/usr/bin/env python3
"""
test改进后ofStable Diffusionmodel
比较改进前后of性can差异
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_diffusion_kanji import StableDiffusionPipeline as OriginalPipeline
from improved_stable_diffusion import ImprovedStableDiffusionPipeline as ImprovedPipeline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

def test_model_comparison():
    """比较原始model和改进modelof性can"""
    
    print("🎌 模型性能对比测试")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # test概念
    concepts = ["water", "future"]
    
    # test原始model
    print(f"\n🔍 测试原始模型...")
    try:
        original_pipeline = OriginalPipeline(device=device)
        print("✅ 原始模型初始化成功")
        
        for concept in concepts:
            print(f"\n🌊 原始模型生成 '{concept}'...")
            start_time = time.time()
            
            try:
                result = original_pipeline.generate(
                    concept,
                    height=128,
                    width=128,
                    num_inference_steps=50,
                    guidance_scale=12.0,
                    seed=42
                )
                
                generation_time = time.time() - start_time
                print(f"   ⏱️  生成时间: {generation_time:.2f}秒")
                
                # convert为PILimage
                if isinstance(result, torch.Tensor):
                    result = (result + 1) / 2
                    result = torch.clamp(result, 0, 1)
                    img_array = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
                else:
                    pil_image = result
                
                # 保存结果
                output_path = f"original_{concept}.png"
                pil_image.save(output_path)
                print(f"   💾 已保存: {output_path}")
                
                # 分析imagequality
                img_array = np.array(pil_image.convert('L'))
                print(f"   📊 图像统计:")
                print(f"      • 尺寸: {img_array.shape}")
                print(f"      • 最小值: {img_array.min()}")
                print(f"      • 最大值: {img_array.max()}")
                print(f"      • 平均值: {img_array.mean():.2f}")
                print(f"      • 标准差: {img_array.std():.2f}")
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                
    except Exception as e:
        print(f"❌ 原始模型初始化失败: {e}")
    
    # test改进model
    print(f"\n🔍 测试改进模型...")
    try:
        improved_pipeline = ImprovedPipeline(device=device)
        print("✅ 改进模型初始化成功")
        
        for concept in concepts:
            print(f"\n🌊 改进模型生成 '{concept}'...")
            start_time = time.time()
            
            try:
                result = improved_pipeline.generate(
                    concept,
                    height=128,
                    width=128,
                    num_inference_steps=50,
                    guidance_scale=7.5,  # using官方推荐ofguidance scale
                    seed=42
                )
                
                generation_time = time.time() - start_time
                print(f"   ⏱️  生成时间: {generation_time:.2f}秒")
                
                # convert为PILimage
                if isinstance(result, torch.Tensor):
                    result = (result + 1) / 2
                    result = torch.clamp(result, 0, 1)
                    img_array = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
                else:
                    pil_image = result
                
                # 保存结果
                output_path = f"improved_{concept}.png"
                pil_image.save(output_path)
                print(f"   💾 已保存: {output_path}")
                
                # 分析imagequality
                img_array = np.array(pil_image.convert('L'))
                print(f"   📊 图像统计:")
                print(f"      • 尺寸: {img_array.shape}")
                print(f"      • 最小值: {img_array.min()}")
                print(f"      • 最大值: {img_array.max()}")
                print(f"      • 平均值: {img_array.mean():.2f}")
                print(f"      • 标准差: {img_array.std():.2f}")
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                
    except Exception as e:
        print(f"❌ 改进模型初始化失败: {e}")
    
    # generation对比图
    print(f"\n🎨 生成对比图...")
    try:
        concepts = ["water", "future"]
        fig, axes = plt.subplots(len(concepts), 2, figsize=(12, 10))
        
        for i, concept in enumerate(concepts):
            # 原始model结果
            original_file = f"original_{concept}.png"
            if os.path.exists(original_file):
                original_img = Image.open(original_file)
                axes[i, 0].imshow(original_img, cmap='gray')
                axes[i, 0].set_title(f'{concept} - 原始模型')
                axes[i, 0].axis('off')
            else:
                axes[i, 0].text(0.5, 0.5, f'{concept}\n原始模型\n生成失败', 
                              ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_title(f'{concept} - 原始模型')
                axes[i, 0].axis('off')
            
            # 改进model结果
            improved_file = f"improved_{concept}.png"
            if os.path.exists(improved_file):
                improved_img = Image.open(improved_file)
                axes[i, 1].imshow(improved_img, cmap='gray')
                axes[i, 1].set_title(f'{concept} - 改进模型')
                axes[i, 1].axis('off')
            else:
                axes[i, 1].text(0.5, 0.5, f'{concept}\n改进模型\n生成失败', 
                              ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].set_title(f'{concept} - 改进模型')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        comparison_path = 'model_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图已保存: {comparison_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ 生成对比图失败: {e}")
    
    print(f"\n🎉 模型对比测试完成！")
    print(f"📁 生成的文件:")
    for concept in concepts:
        print(f"   • original_{concept}.png - 原始模型结果")
        print(f"   • improved_{concept}.png - 改进模型结果")
    print(f"   • model_comparison.png - 对比图")

def analyze_improvements():
    """分析改进点"""
    
    print(f"\n🔍 改进点分析")
    print("=" * 50)
    
    improvements = [
        "🏗️  架构改进:",
        "   • 使用GroupNorm替代BatchNorm，提高训练稳定性",
        "   • 使用SiLU激活函数，替代LeakyReLU",
        "   • 更深的网络结构，增加模型容量",
        "",
        "🎯 训练策略改进:",
        "   • 借鉴官方的时间嵌入网络设计",
        "   • 改进的交叉注意力机制",
        "   • 更好的残差块设计",
        "",
        "⚙️  推理优化:",
        "   • 使用官方推荐的guidance scale (7.5)",
        "   • 改进的DDPM调度器",
        "   • 更稳定的去噪过程",
        "",
        "📝 提示工程:",
        "   • 更详细的汉字描述",
        "   • 专业质量的艺术风格描述",
        "   • 强调对比度和清晰度"
    ]
    
    for improvement in improvements:
        print(improvement)

if __name__ == "__main__":
    test_model_comparison()
    analyze_improvements()
