#!/usr/bin/env python3
"""
Kanji Dataset Builder for Stable Diffusion Training

This script processes KANJIDIC2 and KanjiVG data to create a dataset with:
- Kanji characters
- English meanings
- Pixel images (converted from SVG)
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

class KanjiDatasetBuilder:
    def __init__(self, kanjidic_path, kanjivg_path, output_dir="kanji_dataset"):
        self.kanjidic_path = kanjidic_path
        self.kanjivg_path = kanjivg_path
        self.output_dir = Path(output_dir)
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
                        # Convert to string and clean up
                        svg_content = ET.tostring(g_element, encoding='unicode')
                        
                        # Create complete SVG with proper styling
                        complete_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 109 109" width="109" height="109">
  <style>
    path {{ stroke: #000000; stroke-width: 2; fill: none; }}
  </style>
  {svg_content}
</svg>'''
                        
                        self.svg_data[kanji_char] = complete_svg
                        
                except (ValueError, OverflowError):
                    continue
        
        print(f"Extracted {len(self.svg_data)} SVG entries")
        
    def convert_svg_to_image_rsvg(self, svg_content, size=64):
        """Convert SVG to PIL Image using rsvg-convert"""
        try:
            # Create temporary SVG file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(svg_content)
                svg_file = f.name
            
            # Create temporary PNG file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                png_file = f.name
            
            # Convert using rsvg-convert
            cmd = ['rsvg-convert', '-w', str(size), '-h', str(size), svg_file, '-o', png_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load the PNG file
                image = Image.open(png_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Ensure black strokes on white background
                # Convert to grayscale and threshold to get pure black/white
                gray = image.convert('L')
                binary = gray.point(lambda x: 0 if x < 128 else 255, '1')
                
                # Convert back to RGB with black strokes
                result_img = Image.new('RGB', binary.size, (255, 255, 255))
                result_img.paste(binary, mask=binary)
                
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
    
    def convert_svg_to_image_simple(self, svg_content, size=64):
        """Convert SVG to PIL Image using a simple approach"""
        try:
            # Create a simple canvas and draw the SVG paths manually
            # This is a simplified approach for basic stroke rendering
            image = Image.new('RGB', (size, size), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            # Parse SVG paths and draw them
            # This is a basic implementation - in practice you'd want a more robust SVG parser
            import re
            
            # Extract path data
            path_matches = re.findall(r'd="([^"]+)"', svg_content)
            
            for path_data in path_matches:
                # Simple path rendering (this is a basic implementation)
                # In practice, you'd want a proper SVG path parser
                commands = re.findall(r'([MLHVCSQTAZmlhvcsqtaz])\s*([^MLHVCSQTAZmlhvcsqtaz]*)', path_data)
                
                points = []
                for cmd, params in commands:
                    if cmd.upper() == 'M':  # Move to
                        coords = re.findall(r'([-\d.]+)', params)
                        if len(coords) >= 2:
                            x, y = float(coords[0]), float(coords[1])
                            # Scale to image size
                            x = x * size / 109
                            y = y * size / 109
                            points.append((x, y))
            
            # Draw lines between points
            if len(points) > 1:
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=(0, 0, 0), width=2)
            
            return image
            
        except Exception as e:
            print(f"Error in simple SVG conversion: {e}")
            return None
    
    def convert_svg_to_image(self, svg_content, size=64):
        """Convert SVG to PIL Image - try multiple methods"""
        # Try rsvg-convert first
        image = self.convert_svg_to_image_rsvg(svg_content, size)
        if image is not None:
            return image
        
        # Fallback to simple method
        print("Falling back to simple SVG conversion...")
        return self.convert_svg_to_image_simple(svg_content, size)
    
    def build_dataset(self):
        """Build the complete dataset"""
        print("Building Kanji dataset...")
        
        # Extract data
        self.extract_kanji_meanings()
        self.extract_svg_data()
        
        # Find common Kanji between the two datasets
        common_kanji = set(self.kanji_data.keys()) & set(self.svg_data.keys())
        print(f"Found {len(common_kanji)} Kanji with both meanings and SVG data")
        
        dataset = []
        
        for kanji in common_kanji:
            # Convert SVG to image
            image = self.convert_svg_to_image(self.svg_data[kanji])
            
            if image is not None:
                # Save image
                image_filename = f"{ord(kanji):04x}.png"
                image_path = self.output_dir / "images" / image_filename
                image.save(image_path)
                
                # Create dataset entry
                entry = {
                    'kanji': kanji,
                    'unicode': self.kanji_data[kanji]['unicode'],
                    'meanings': self.kanji_data[kanji]['meanings'],
                    'image_file': image_filename,
                    'prompt': f"kanji character {kanji}: {', '.join(self.kanji_data[kanji]['meanings'])}"
                }
                
                dataset.append(entry)
        
        # Save metadata
        metadata_path = self.output_dir / "metadata" / "dataset.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # Save individual files for easy access
        for entry in dataset:
            entry_filename = f"{ord(entry['kanji']):04x}.json"
            entry_path = self.output_dir / "metadata" / entry_filename
            with open(entry_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset built successfully!")
        print(f"Total entries: {len(dataset)}")
        print(f"Images saved to: {self.output_dir / 'images'}")
        print(f"Metadata saved to: {self.output_dir / 'metadata'}")
        
        return dataset

def main():
    # Initialize the dataset builder
    builder = KanjiDatasetBuilder(
        kanjidic_path="kanjidic2.xml",
        kanjivg_path="kanjivg-20220427.xml",
        output_dir="kanji_dataset"
    )
    
    # Build the dataset
    dataset = builder.build_dataset()
    
    # Print some examples
    print("\nExample entries:")
    for i, entry in enumerate(dataset[:5]):
        print(f"{i+1}. {entry['kanji']}: {entry['meanings'][:3]}...")

if __name__ == "__main__":
    main() 