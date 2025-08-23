# Kanji Dataset for Stable Diffusion Training

This project creates a dataset of Kanji characters with their English meanings and corresponding pixel images, suitable for training stable diffusion models to generate Kanji characters.

## Overview

The dataset contains:
- **6,410 Kanji characters** with English meanings
- **16,692 total meanings** (average 2.6 meanings per Kanji)
- **64x64 pixel images** with pure black strokes on white background
- **No stroke order numbers** - clean character images only

## Data Sources

The dataset is built from two primary sources:

1. **KANJIDIC2** (`kanjidic2.xml.gz`)
   - Source: https://www.edrdg.org/kanjidic/kanjidic2.xml.gz
   - Contains Kanji definitions and English meanings
   - Extracted 10,383 Kanji with meanings

2. **KanjiVG** (`kanjivg-20220427.xml.gz`)
   - Source: https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz
   - Contains SVG stroke data for Kanji characters
   - Extracted 6,761 SVG entries

## Dataset Structure

```
kanji_dataset/
├── images/           # PNG images (64x64 pixels)
│   ├── 4e85.png     # Kanji: 人 (person)
│   ├── 5927.png     # Kanji: 大 (large, big)
│   └── ...
├── metadata/
│   ├── dataset.json  # Complete dataset in JSON format
│   ├── 4e85.json    # Individual Kanji metadata
│   ├── 5927.json    # Individual Kanji metadata
│   └── ...
```

## Image Specifications

- **Format**: PNG
- **Size**: 64x64 pixels
- **Colors**: Pure black strokes (#000000) on white background (#FFFFFF)
- **No stroke order numbers**: Clean character images only
- **Quality**: High-contrast, suitable for machine learning

## Dataset Format

Each entry in the dataset contains:

```json
{
  "kanji": "人",
  "unicode": "4e85",
  "meanings": ["person"],
  "image_file": "4e85.png",
  "prompt": "kanji character 人: person"
}
```

### Fields:
- `kanji`: The actual Kanji character
- `unicode`: Unicode code point in hexadecimal
- `meanings`: Array of English meanings
- `image_file`: Filename of the corresponding PNG image
- `prompt`: Formatted prompt for stable diffusion training

## Usage for Stable Diffusion Training

### 1. Basic Training Setup

```python
import json
from pathlib import Path

# Load dataset
with open('kanji_dataset/metadata/dataset.json', 'r') as f:
    dataset = json.load(f)

# Example training pairs
for entry in dataset:
    image_path = f"kanji_dataset/images/{entry['image_file']}"
    prompt = entry['prompt']
    # Use for training...
```

### 2. Training Configuration

Recommended settings for stable diffusion training:

- **Image size**: 64x64 pixels (as provided)
- **Batch size**: 8-16 (depending on GPU memory)
- **Learning rate**: 1e-5 to 1e-4
- **Training steps**: 1000-5000 per epoch
- **Prompt format**: "kanji character {kanji}: {meanings}"

### 3. Data Augmentation

Consider these augmentations for better generalization:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomRotation(5),  # Small rotations
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight brightness/contrast
])
```

## Quality Assurance

The dataset has been verified to ensure:

✅ **6,410 Kanji characters** with valid meanings  
✅ **6,410 corresponding images** (100% coverage)  
✅ **Pure black/white images** (no grayscale artifacts)  
✅ **64x64 pixel resolution** (consistent size)  
✅ **No stroke order numbers** (clean characters)  
✅ **Common Kanji included** (人, 大, 小, 山, 川, 日, 月, 火, 水, 木)  

## Statistics

- **Total Kanji**: 6,410
- **Total meanings**: 16,692
- **Average meanings per Kanji**: 2.6
- **Image format**: PNG, 64x64 pixels
- **Color depth**: RGB (black/white only)

## Common Kanji Examples

| Kanji | Meanings | Unicode |
|-------|----------|---------|
| 人 | person | 4e85 |
| 大 | large, big | 5927 |
| 小 | little, small | 5c0f |
| 山 | mountain | 5c71 |
| 川 | stream, river | 5ddd |
| 日 | day, sun, Japan | 65e5 |
| 月 | month, moon | 6708 |
| 火 | fire | 706b |
| 水 | water | 6c34 |
| 木 | tree, wood | 6728 |

## Building the Dataset

To rebuild the dataset from source:

```bash
# Install dependencies
brew install cairo librsvg
python3 -m pip install pillow

# Run the dataset builder
python3 process_kanji_data.py

# Verify the dataset
python3 verify_dataset.py
```

## Requirements

- Python 3.7+
- Pillow (PIL)
- Cairo graphics library
- librsvg (for SVG conversion)

## License

This dataset is derived from:
- KANJIDIC2 (Creative Commons Attribution-ShareAlike 3.0)
- KanjiVG (Creative Commons Attribution-ShareAlike 3.0)

The processed dataset maintains the same license terms.

## Citation

If you use this dataset in your research, please cite:

```
Kanji Dataset for Stable Diffusion Training
Built from KANJIDIC2 and KanjiVG data
https://www.edrdg.org/kanjidic/
https://kanjivg.tagaini.net/
```

## Applications

This dataset is suitable for:

1. **Stable Diffusion Training**: Generate Kanji characters from text descriptions
2. **OCR Training**: Recognize handwritten Kanji
3. **Font Generation**: Create new Kanji fonts
4. **Educational Tools**: Interactive Kanji learning applications
5. **Research**: Machine learning studies on character recognition

## Troubleshooting

### Common Issues

1. **Cairo library not found**: Install with `brew install cairo`
2. **rsvg-convert not found**: Install with `brew install librsvg`
3. **Image quality issues**: Ensure proper thresholding in the conversion process

### Performance Tips

- Use GPU acceleration for training
- Consider data augmentation for better generalization
- Monitor training loss to prevent overfitting
- Use validation split for model evaluation 