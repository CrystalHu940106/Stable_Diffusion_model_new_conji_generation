#!/usr/bin/env python3
"""
批量转换Python脚本为Jupyter notebook
扫描整个项目包，生成能在Colab和Kaggle上运行的notebook
"""

import os
import subprocess
import sys
from pathlib import Path

def batch_convert_to_notebooks():
    """批量转换Python文件为Jupyter notebook"""
    
    # 扫描scripts目录
    scripts_dir = './scripts'
    
    if not os.path.exists(scripts_dir):
        print(f"❌ 目录不存在: {scripts_dir}")
        return
    
    print(f"🔍 扫描目录: {scripts_dir}")
    
    # 重点转换的训练相关脚本
    priority_files = [
        'colab_training.py',
        'train_stable_diffusion.py', 
        'improved_stable_diffusion.py',
        'test_improved_model.py'
    ]
    
    converted_files = []
    
    # 首先转换优先级文件
    for filename in priority_files:
        py_path = os.path.join(scripts_dir, filename)
        if os.path.exists(py_path):
            ipynb_path = py_path.replace('.py', '.ipynb')
            
            print(f"🔄 转换: {filename}")
            try:
                # 使用ipynb-py-convert转换
                result = subprocess.run([
                    '/Users/hu.crystal/Library/Python/3.9/bin/ipynb-py-convert', 
                    py_path, ipynb_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ 成功转换: {ipynb_path}")
                    converted_files.append(ipynb_path)
                else:
                    print(f"❌ 转换失败: {filename}")
                    print(f"错误: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ 转换错误 {filename}: {e}")
    
    # 然后转换其他Python文件
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.py') and filename not in priority_files:
            py_path = os.path.join(scripts_dir, filename)
            ipynb_path = py_path.replace('.py', '.ipynb')
            
            print(f"🔄 转换: {filename}")
            try:
                result = subprocess.run([
                    '/Users/hu.crystal/Library/Python/3.9/bin/ipynb-py-convert', 
                    py_path, ipynb_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ 成功转换: {ipynb_path}")
                    converted_files.append(ipynb_path)
                else:
                    print(f"❌ 转换失败: {filename}")
                    
            except Exception as e:
                print(f"❌ 转换错误 {filename}: {e}")
    
    print(f"\n🎉 转换完成！共转换 {len(converted_files)} 个文件:")
    for file in converted_files:
        print(f"   📓 {file}")
    
    return converted_files

def create_complete_colab_notebook():
    """创建一个完整的Colab训练notebook"""
    
    print("\n🚀 创建完整的Colab/Kaggle训练notebook...")
    
    # 读取核心脚本内容
    scripts_to_include = [
        'scripts/improved_stable_diffusion.py',
        'scripts/colab_training.py'
    ]
    
    notebook_cells = []
    
    # 添加标题和说明
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Complete Stable Diffusion Kanji Generation - Colab/Kaggle\n",
            "\n",
            "**Single file training notebook** - Upload to Colab/Kaggle and start training immediately!\n",
            "\n",
            "## 🎯 Features\n",
            "- ✅ **Complete Training Pipeline**: VAE + UNet + DDPM\n",
            "- 🚀 **GPU Optimized**: Auto CUDA/MPS detection\n",
            "- 💾 **Auto-save**: Checkpoints every 5 epochs\n",
            "- 📊 **Real-time Monitoring**: Loss curves and GPU stats\n",
            "- 🔄 **Resume Training**: Continue from any checkpoint\n",
            "- 🎌 **Kanji Generation**: Text-to-Kanji capabilities\n",
            "\n",
            "## 🚀 Quick Start\n",
            "1. Upload this notebook to Colab/Kaggle\n",
            "2. Select GPU runtime\n",
            "3. Run all cells\n",
            "4. Start training!\n",
            "\n",
            "**Expected Training Time**:\n",
            "- Colab Free (T4): 50 epochs in 2-3 hours\n",
            "- Colab Pro (V100/P100): 50 epochs in 1-1.5 hours\n",
            "- Kaggle (P100): 50 epochs in 1-2 hours"
        ]
    })
    
    # 添加依赖安装
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📦 Install Dependencies"]
    })
    
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required packages\n",
            "!pip install transformers pillow matplotlib scikit-image opencv-python tqdm\n",
            "!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "\n",
            "print(\"✅ Dependencies installed successfully!\")"
        ]
    })
    
    # 添加GPU检查
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🔧 Check GPU and Environment"]
    })
    
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import os\n",
            "\n",
            "# Check environment\n",
            "is_colab = 'COLAB_GPU' in os.environ\n",
            "is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ\n",
            "\n",
            "print(f\"🌐 Environment: {'Colab' if is_colab else 'Kaggle' if is_kaggle else 'Local'}\")\n",
            "print(f\"PyTorch: {torch.__version__}\")\n",
            "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
            "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
            "    print(\"🍎 Apple Silicon (MPS) available\")\n",
            "else:\n",
            "    print(\"⚠️ Using CPU (will be slow!)\")"
        ]
    })
    
    # 读取并添加模型实现代码
    for script_path in scripts_to_include:
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 提取主要代码部分（去掉import和main部分）
            lines = content.split('\n')
            code_lines = []
            skip_main = False
            
            for line in lines:
                if line.strip().startswith('if __name__'):
                    skip_main = True
                if not skip_main and not line.strip().startswith('#!'):
                    code_lines.append(line)
            
            clean_code = '\n'.join(code_lines)
            
            notebook_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## 🏗️ {os.path.basename(script_path)} Implementation"]
            })
            
            notebook_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [clean_code]
            })
    
    # 添加训练启动代码
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🚀 Start Training"]
    })
    
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create trainer and start training\n",
            "if 'ColabOptimizedTrainer' in globals():\n",
            "    trainer = ColabOptimizedTrainer()\n",
            "    trainer.train()\n",
            "else:\n",
            "    print(\"⚠️ Trainer class not found. Please run the model implementation cells first.\")"
        ]
    })
    
    # 添加结果下载
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📥 Download Results"]
    })
    
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Download training results\n",
            "from google.colab import files\n",
            "import zipfile\n",
            "\n",
            "def download_results():\n",
            "    print(\"📥 Preparing results for download...\")\n",
            "    \n",
            "    # Create results zip\n",
            "    with zipfile.ZipFile('training_results.zip', 'w') as zipf:\n",
            "        # Add checkpoints\n",
            "        if os.path.exists('checkpoints'):\n",
            "            for root, dirs, files in os.walk('checkpoints'):\n",
            "                for file in files:\n",
            "                    file_path = os.path.join(root, file)\n",
            "                    zipf.write(file_path, os.path.relpath(file_path, '.'))\n",
            "        \n",
            "        # Add training curves\n",
            "        for img_file in ['training_curve.png', 'loss_curve.png']:\n",
            "            if os.path.exists(img_file):\n",
            "                zipf.write(img_file)\n",
            "        \n",
            "        # Add generated images\n",
            "        for i in range(10):\n",
            "            img_file = f'generated_{i}.png'\n",
            "            if os.path.exists(img_file):\n",
            "                zipf.write(img_file)\n",
            "    \n",
            "    print(\"✅ Results packaged: training_results.zip\")\n",
            "    \n",
            "    # Download\n",
            "    try:\n",
            "        files.download('training_results.zip')\n",
            "        print(\"📥 Results downloaded successfully!\")\n",
            "    except:\n",
            "        print(\"⚠️ Download failed (not in Colab)\")\n",
            "        print(\"📁 Files are saved in the current directory\")\n",
            "\n",
            "# Download results\n",
            "download_results()"
        ]
    })
    
    # 创建完整的notebook
    complete_notebook = {
        "cells": notebook_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # 保存notebook
    import json
    notebook_path = 'complete_colab_kaggle_training.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(complete_notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ 完整的Colab/Kaggle notebook已创建: {notebook_path}")
    return notebook_path

if __name__ == "__main__":
    print("🚀 开始批量转换Python脚本为Jupyter notebook...")
    
    # 批量转换现有脚本
    converted_files = batch_convert_to_notebooks()
    
    # 创建完整的Colab训练notebook
    complete_notebook = create_complete_colab_notebook()
    
    print(f"\n🎉 所有转换完成！")
    print(f"📓 转换的notebook文件: {len(converted_files)} 个")
    print(f"🚀 完整的Colab/Kaggle训练notebook: {complete_notebook}")
    print(f"\n📋 使用说明:")
    print(f"   1. 上传 {complete_notebook} 到 Google Colab 或 Kaggle")
    print(f"   2. 选择 GPU 运行时")
    print(f"   3. 运行所有单元格开始训练")
    print(f"   4. 训练完成后下载结果")
