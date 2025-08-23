#!/usr/bin/env python3
"""
Fixed Kanji Dataset Builder
Ensure proper black strokes on white background
"""

import xml.etree.ElementTree as ET
import os
import re
from PIL import Image, ImageDraw
import io
import json
from pathlib import Path
import subprocess
import tempfile
import shutil

class FixedKanjiDatasetBuilder:
    def __init__(self, kanjidic_path, kanjivg_path, output_dir="fixed_kanji_dataset"):
        self.kanjidic_path = kanjidic_path
        self.kanjivg_path = kanjivg_path
        self.output_dir = Path(output_dir)
        
        # Remove existing directory and recreate
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        self.kanji_data = {}
        self.svg_data = {}
        
    def extract_kanji_meanings(self):
        """Extract Kanji characters and their English meanings from KANJIDIC2"""
        print("Extracting Kanji meanings from KANJIDIC2...")
        
        # Parse the XML file
        tree = ET.parse(self.kanjidic_path)
        root = tree.getroot()
        
        # Define the namespace
        namespace = {'k': 'http://www.edrdg.org/kanjidic/kanjidic2'}
        
        for character in root.findall('.//character'):
            literal = character.find('literal')
            if literal is None:
                continue
                
            kanji_char = literal.text
            
            # Extract English meanings
            meanings = []
            reading_meaning = character.find('.//reading_meaning')
            if reading_meaning is not None:
                for meaning in reading_meaning.findall('.//meaning'):
                    # Only include English meanings (no language attribute or m_lang="en")
                    if meaning.get('m_lang') is None or meaning.get('m_lang') == 'en':
                        meanings.append(meaning.text)
            
            if meanings:
                self.kanji_data[kanji_char] = {
                    'meanings': meanings,
                    'unicode': None
                }
                
                # Extract Unicode code point
                codepoint = character.find('.//codepoint/cp_value[@cp_type="ucs"]')
                if codepoint is not None:
                    self.kanji_data[kanji_char]['unicode'] = codepoint.text
        
        print(f"Extracted {len(self.kanji_data)} Kanji with meanings")
        
    def extract_svg_data(self):
        """Extract SVG data from KanjiVG"""
        print("Extracting SVG data from KanjiVG...")
        
        # Parse the XML file
        tree = ET.parse(self.kanjivg_path)
        root = tree.getroot()
        
        # Define the namespace
        namespace = {'kvg': 'http://kanjivg.tagaini.net'}
        
        for kanji in root.findall('.//kanji'):
            kanji_id = kanji.get('id')
            
            # Extract the character from the ID (format: kvg:kanji_XXXXX)
            match = re.search(r'kvg:kanji_([0-9a-f]+)', kanji_id)
            if match:
                unicode_hex = match.group(1)
                try:
                    # Convert hex to Unicode character
                    unicode_int = int(unicode_hex, 16)
                    kanji_char = chr(unicode_int)
                    
                    # Get the SVG content
                    g_element = kanji.find('.//g')
                    if g_element is not None:
                        # Convert to string
                        svg_content = ET.tostring(g_element, encoding='unicode')
                        
                        # Create full SVG document
                        full_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 109 109">
  <g style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
    {svg_content}
  </g>
</svg>'''
                        
                        self.svg_data[kanji_char] = full_svg
                        
                except (ValueError, UnicodeEncodeError):
                    continue
        
        print(f"Extracted {len(self.svg_data)} SVG entries")
    
    def convert_svg_to_image_fixed(self, svg_content, size=64):
        """Convert SVG to PIL Image with proper black strokes - FIXED VERSION"""
        try:
            # Create temporary SVG file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(svg_content)
                svg_file = f.name
            
            # Create temporary PNG file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                png_file = f.name
            
            # Convert using rsvg-convert with better parameters
            cmd = [
                'rsvg-convert', 
                '-w', str(size), 
                '-h', str(size), 
                '--background-color', 'white',
                svg_file, 
                '-o', png_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load the PNG file
                image = Image.open(png_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Create a new white background image
                result_img = Image.new('RGB', (size, size), (255, 255, 255))
                
                # Convert to grayscale and threshold to get pure black/white
                gray = image.convert('L')
                
                # Apply threshold to get black strokes
                # Any pixel darker than 200 becomes black, rest becomes white
                threshold = 200
                for y in range(size):
                    for x in range(size):
                        pixel_value = gray.getpixel((x, y))
                        if pixel_value < threshold:
                            result_img.putpixel((x, y), (0, 0, 0))  # Black stroke
                        else:
                            result_img.putpixel((x, y), (255, 255, 255))  # White background
                
                # Clean up temporary files
                os.unlink(svg_file)
                os.unlink(png_file)
                
                return result_img
            else:
                print(f"rsvg-convert error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error converting SVG to image: {e}")
            return None
    
    def build_dataset(self):
        """Build the complete dataset"""
        print("Building dataset...")
        
        # Extract data
        self.extract_kanji_meanings()
        self.extract_svg_data()
        
        # Match Kanji with SVG data
        matched_kanji = []
        
        for kanji_char, kanji_info in self.kanji_data.items():
            if kanji_char in self.svg_data:
                unicode_hex = kanji_info['unicode']
                if unicode_hex:
                    # Convert SVG to image
                    svg_content = self.svg_data[kanji_char]
                    image = self.convert_svg_to_image_fixed(svg_content)
                    
                    if image is not None:
                        # Save image
                        image_filename = f"{unicode_hex}.png"
                        image_path = self.output_dir / "images" / image_filename
                        image.save(image_path, "PNG")
                        
                        # Create metadata
                        metadata = {
                            'kanji': kanji_char,
                            'unicode': unicode_hex,
                            'meanings': kanji_info['meanings'],
                            'image_file': image_filename,
                            'prompt': f"kanji character {kanji_char}: {', '.join(kanji_info['meanings'][:3])}"
                        }
                        
                        # Save individual metadata file
                        metadata_filename = f"{unicode_hex}.json"
                        metadata_path = self.output_dir / "metadata" / metadata_filename
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                        
                        matched_kanji.append(metadata)
                        
                        if len(matched_kanji) % 100 == 0:
                            print(f"Processed {len(matched_kanji)} Kanji...")
        
        # Save complete dataset
        dataset_path = self.output_dir / "metadata" / "dataset.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(matched_kanji, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset built successfully!")
        print(f"Total Kanji: {len(matched_kanji)}")
        print(f"Images saved in: {self.output_dir / 'images'}")
        print(f"Metadata saved in: {self.output_dir / 'metadata'}")
        
        return matched_kanji

def verify_image_quality(dataset_path):
    """Verify the quality of generated images"""
    print("\n🔍 Verifying Image Quality...")
    
    images_dir = Path(dataset_path) / "images"
    if not images_dir.exists():
        print("❌ Images directory not found!")
        return
    
    # Check a few sample images
    sample_files = list(images_dir.glob("*.png"))[:5]
    
    for img_file in sample_files:
        try:
            img = Image.open(img_file)
            
            # Convert to RGB and analyze
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get color statistics
            img_array = img.convert('RGB')
            r, g, b = img_array.split()
            r_extrema = r.getextrema()
            g_extrema = g.getextrema()
            b_extrema = b.getextrema()
            
            print(f"   📸 {img_file.name}:")
            print(f"     • Size: {img.size}")
            print(f"     • Color range: R{r_extrema}, G{g_extrema}, B{b_extrema}")
            
            # Check if it has black strokes
            if r_extrema[0] == 0 and g_extrema[0] == 0 and b_extrema[0] == 0:
                print(f"     • ✅ Has black strokes")
            else:
                print(f"     • ❌ No black strokes")
            
            # Check if it has white background
            if r_extrema[1] == 255 and g_extrema[1] == 255 and b_extrema[1] == 255:
                print(f"     • ✅ Has white background")
            else:
                print(f"     • ❌ No white background")
                
        except Exception as e:
            print(f"   ❌ Error reading {img_file.name}: {e}")

def main():
    """Main function"""
    print("🎌 Fixed Kanji Dataset Builder")
    print("=" * 50)
    
    # Check if source files exist
    kanjidic_path = "kanjidic2.xml"
    kanjivg_path = "kanjivg-20220427.xml"
    
    if not Path(kanjidic_path).exists():
        print(f"❌ {kanjidic_path} not found!")
        return
    
    if not Path(kanjivg_path).exists():
        print(f"❌ {kanjivg_path} not found!")
        return
    
    # Build dataset
    builder = FixedKanjiDatasetBuilder(kanjidic_path, kanjivg_path)
    dataset = builder.build_dataset()
    
    # Verify quality
    verify_image_quality("fixed_kanji_dataset")
    
    print(f"\n🎉 Fixed dataset creation complete!")
    print(f"   • Dataset saved in: fixed_kanji_dataset/")
    print(f"   • Images: fixed_kanji_dataset/images/")
    print(f"   • Metadata: fixed_kanji_dataset/metadata/")
    print(f"   • Total Kanji: {len(dataset)}")

if __name__ == "__main__":
    main()
