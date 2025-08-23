# Kanji Diffusion Project Structure

## 📁 项目文件夹结构

```
Question2/
├── 📁 data/                    # 数据文件夹
│   ├── kanjidic2.xml          # 原始汉字字典数据
│   ├── kanjivg-20220427.xml   # 原始汉字SVG数据
│   └── fixed_kanji_dataset/   # 修复后的汉字数据集
│       ├── images/            # 6,410个汉字图像 (64x64 PNG)
│       └── metadata/          # 汉字元数据 (JSON格式)
│
├── 📁 scripts/                 # 脚本文件夹
│   ├── 🛠️ 数据处理脚本
│   │   ├── process_kanji_data.py      # 原始数据处理脚本
│   │   ├── fix_kanji_dataset.py       # 修复版数据处理脚本
│   │   ├── dataset_summary.py         # 数据集统计脚本
│   │   └── verify_dataset.py          # 数据集验证脚本
│   │
│   ├── 🎯 训练脚本
│   │   ├── quick_train_test.py        # 快速训练测试脚本
│   │   ├── full_train_kanji.py        # 完整训练脚本
│   │   └── example_training.py        # 训练示例脚本
│   │
│   ├── 🧪 测试脚本
│   │   ├── test_generation.py         # 基础生成测试
│   │   ├── advanced_test.py           # 高级生成测试
│   │   └── debug_generation.py        # 调试生成问题
│   │
│   ├── 📊 分析脚本
│   │   ├── view_original_kanji.py     # 查看原始汉字
│   │   ├── analyze_fixed_images.py    # 分析修复图像
│   │   ├── analyze_images.py          # 图像分析
│   │   └── simple_image_viewer.py     # 简单图像查看器
│   │
│   └── 🎌 演示脚本
│       └── demonstration.py           # 项目演示脚本
│
├── 📁 configs/                 # 配置文件
│   ├── optimized_training_config.py   # 优化训练配置
│   └── quick_test_config.py           # 快速测试配置
│
├── 📁 results/                 # 结果文件夹
│   ├── test_analysis.py               # 测试分析结果
│   ├── test_summary.json              # 测试总结
│   ├── generated_results/             # 基础生成结果
│   ├── advanced_results/              # 高级生成结果
│   ├── fixed_generation/              # 修复生成结果
│   └── quick_test_results/            # 快速测试结果
│
├── 📁 models/                  # 模型文件夹 (训练后保存)
│   └── (训练后的模型文件将保存在这里)
│
├── 📁 docs/                    # 文档文件夹
│   ├── PROJECT_SUMMARY.md      # 项目总结
│   └── README.md               # 项目说明
│
└── 📄 PROJECT_STRUCTURE.md     # 项目结构说明 (本文件)
```

## 🎯 各文件夹功能说明

### 📁 data/ - 数据文件夹
- **原始数据**: KANJIDIC2和KanjiVG的XML文件
- **处理数据**: 修复后的汉字数据集，包含6,410个汉字
- **图像格式**: 64x64 PNG，黑字白底
- **元数据**: JSON格式，包含汉字、含义、Unicode等信息

### 📁 scripts/ - 脚本文件夹
- **数据处理**: 从原始XML提取汉字和SVG，转换为图像
- **训练脚本**: 快速测试和完整训练的模型训练脚本
- **测试脚本**: 生成测试、调试和验证脚本
- **分析脚本**: 图像质量分析和数据集统计脚本

### 📁 configs/ - 配置文件
- **训练配置**: 优化后的训练参数和设置
- **测试配置**: 快速测试的配置参数

### 📁 results/ - 结果文件夹
- **生成结果**: 所有测试生成的图像和结果
- **分析结果**: 测试分析和统计结果
- **模型结果**: 训练过程中的检查点和日志

### 📁 models/ - 模型文件夹
- **保存位置**: 训练完成的模型文件
- **检查点**: 训练过程中的模型检查点

### 📁 docs/ - 文档文件夹
- **项目文档**: 项目说明、总结和使用指南
- **技术文档**: 技术细节和实现说明

## 🚀 使用流程

### 1. 数据处理
```bash
cd scripts/
python3 fix_kanji_dataset.py
```

### 2. 快速测试
```bash
cd scripts/
python3 quick_train_test.py
```

### 3. 完整训练
```bash
cd scripts/
python3 full_train_kanji.py
```

### 4. 生成测试
```bash
cd scripts/
python3 test_generation.py
```

### 5. 结果分析
```bash
cd scripts/
python3 analyze_fixed_images.py
```

## 📊 数据集信息

- **总汉字数**: 6,410个
- **图像格式**: 64x64 PNG
- **颜色**: 黑字白底
- **元数据**: 16,692个含义 (平均2.6个/汉字)
- **Unicode范围**: U+342C 到 U+20B9F

## 🎯 项目目标

1. **数据处理**: 从原始XML创建高质量的汉字图像数据集
2. **模型训练**: 训练Stable Diffusion模型生成汉字
3. **汉字生成**: 根据英文描述生成新的汉字字符
4. **质量验证**: 确保生成的汉字具有文化真实性

## 📈 当前状态

- ✅ **数据处理**: 完成，6,410个汉字图像
- ✅ **快速测试**: 完成，训练流程验证
- 🔄 **完整训练**: 准备就绪，等待开始
- 🎯 **汉字生成**: 目标功能，需要完整训练后实现
