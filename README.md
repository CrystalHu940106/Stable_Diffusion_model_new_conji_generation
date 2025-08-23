# 🎌 Kanji Diffusion Model - New Conji Generation

A complete pipeline for training and generating novel Kanji characters using diffusion models, built with PyTorch.

## 🌟 Features

- **Complete Training Pipeline**: From dataset preparation to model training
- **Advanced UNet Architecture**: Optimized for Kanji generation
- **Dataset Management**: Automatic processing of KANJIDIC2 and KanjiVG data
- **Multiple Training Modes**: Quick test, full training, and concept-specific generation
- **Visualization Tools**: Comprehensive analysis and comparison scripts
- **CPU Optimized**: Designed for efficient training on CPU resources

## 📁 Project Structure

```
Question2/
├── configs/                 # Training configurations
├── data/                    # Dataset storage
├── docs/                    # Documentation
├── scripts/                 # Core scripts
├── models/                  # Model storage
└── results/                 # Training results
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/CrystalHu940106/Stable_Diffusion_model_new_conji_generation.git
cd Stable_Diffusion_model_new_conji_generation

# Install dependencies
pip install torch torchvision pillow matplotlib numpy
```

### 2. Prepare Dataset

```bash
# Build the Kanji dataset
python3 scripts/fix_kanji_dataset.py
```

### 3. Quick Training Test

```bash
# Run a quick training test
python3 scripts/quick_train_test.py
```

### 4. Full Training

```bash
# Train the complete model
python3 scripts/full_train_kanji.py
```

### 5. Generate Concept Kanji

```bash
# Generate Kanji for specific concepts
python3 scripts/generate_concept_kanji.py
```

## 🔧 Core Scripts

- **`fix_kanji_dataset.py`**: Builds the complete Kanji dataset from KANJIDIC2 and KanjiVG
- **`quick_train_test.py`**: Quick validation of the training pipeline
- **`full_train_kanji.py`**: Complete training on the full dataset
- **`generate_concept_kanji.py`**: Generate Kanji for specific concepts
- **`compare_generated_kanji.py`**: Compare generated vs. existing Kanji

## 📊 Dataset

- **Source**: KANJIDIC2 (meanings) + KanjiVG (stroke data)
- **Size**: 6,410 Kanji characters
- **Format**: 128x128 PNG images with black strokes on white background
- **Quality**: High-quality vector-to-raster conversion

## 🏗️ Model Architecture

- **Type**: Advanced UNet with BatchNorm
- **Input/Output**: 3-channel RGB images
- **Resolution**: 128x128 pixels (configurable)
- **Optimization**: AdamW optimizer with learning rate scheduling

## ⚙️ Configuration

Training parameters can be customized in `configs/optimized_training_config.py`:

- Image resolution: 64x64 to 256x256
- Batch size: 2-8 (CPU optimized)
- Learning rate: 2e-4 (default)
- Epochs: 3-10 (configurable)

## 📈 Training Results

The model generates:
- **Concept-specific Kanji**: Success, failure, novel, funny, culturally meaningful
- **Visual patterns**: Learned representations of Kanji stroke structures
- **Quality metrics**: Training/validation loss tracking

## 🔍 Analysis Tools

- **Image Quality Analysis**: Pixel distribution, contrast, complexity
- **Training Progress**: Loss curves and convergence analysis
- **Generation Comparison**: Side-by-side analysis of generated vs. existing Kanji

## 🎯 Use Cases

- **Educational**: Teaching Kanji stroke patterns
- **Creative**: Generating novel character designs
- **Research**: Studying AI understanding of written language
- **Cultural**: Exploring AI interpretation of cultural symbols

## 🛠️ Technical Details

- **Framework**: PyTorch
- **Image Processing**: PIL, NumPy
- **Data Loading**: Custom Dataset classes with DataLoader
- **Optimization**: Gradient clipping, learning rate scheduling
- **Checkpointing**: Automatic model saving and loading

## 📝 Requirements

- Python 3.7+
- PyTorch 1.8+
- PIL (Pillow)
- NumPy
- Matplotlib
- rsvg-convert (for SVG processing)

## 🤝 Contributing

This project is open for contributions! Areas for improvement:
- Model architecture enhancements
- Additional training strategies
- More concept-specific generation
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- KANJIDIC2 project for Kanji meanings
- KanjiVG project for stroke data
- PyTorch community for the deep learning framework

## 📞 Contact

For questions or contributions, please open an issue on GitHub.

---

**Note**: This project demonstrates AI's potential in understanding and generating written language, particularly in the context of Japanese Kanji characters. While the current model is a foundation, it opens exciting possibilities for future research in AI-generated writing systems.
