#!/usr/bin/env python3
"""
Fixed SVG to Image Conversion
"""

import xml.etree.ElementTree as ET
import os
import re
from PIL import Image, ImageDraw
import tempfile
import subprocess
from pathlib import Path

def convert_svg_to_image_fixed(svg_content, size=64):
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

def create_simple_kanji_svg():
    """Create a simple test kanji SVG"""
    
    # Create a simple "一" (one) kanji
    simple_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 109 109" width="109" height="109">
  <rect width="109" height="109" fill="white"/>
  <g style="fill:none;stroke:#000000;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;">
    <path d="M20,55 L90,55"/>
  </g>
</svg>'''
    
    return simple_svg

def test_fixed_conversion():
    """Test the fixed conversion"""
    
    print("🔧 Testing Fixed SVG Conversion")
    print("=" * 50)
    
    # Test with simple kanji
    simple_svg = create_simple_kanji_svg()
    print("📝 Simple kanji SVG created")
    
    # Convert
    image = convert_svg_to_image_fixed(simple_svg, 64)
    
    if image is not None:
        print(f"✅ Image created: {image.size}, {image.mode}")
        
        # Analyze
        img_array = image.convert('RGB')
        r, g, b = img_array.split()
        r_extrema = r.getextrema()
        g_extrema = g.getextrema()
        b_extrema = b.getextrema()
        
        print(f"📊 Color ranges: R{r_extrema}, G{g_extrema}, B{b_extrema}")
        
        # Save
        output_path = "fixed_test_kanji.png"
        image.save(output_path)
        print(f"💾 Fixed test image saved to: {output_path}")
        
        return True
    else:
        print("❌ Conversion failed!")
        return False

def test_real_kanji_fixed():
    """Test with a real kanji using fixed conversion"""
    
    print(f"\n🔧 Testing Real Kanji with Fixed Conversion")
    print("=" * 50)
    
    # Parse KanjiVG to get a real SVG
    kanjivg_path = "data/kanjivg-20220427.xml"
    
    if not Path(kanjivg_path).exists():
        print(f"❌ {kanjivg_path} not found!")
        return False
    
    # Parse the XML file
    tree = ET.parse(kanjivg_path)
    root = tree.getroot()
    
    # Find a simple kanji (like "一")
    target_unicode = "4e00"  # 一 (one)
    
    for kanji in root.findall('.//kanji'):
        kanji_id = kanji.get('id')
        match = re.search(r'kvg:kanji_([0-9a-f]+)', kanji_id)
        if match and match.group(1) == target_unicode:
            print(f"📝 Found target kanji: {kanji_id}")
            
            # Get the SVG content
            g_element = kanji.find('.//g')
            if g_element is not None:
                svg_content = ET.tostring(g_element, encoding='unicode')
                
                # Create full SVG document with better styling
                full_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 109 109" width="109" height="109">
  <rect width="109" height="109" fill="white"/>
  <g style="fill:none;stroke:#000000;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;">
    {svg_content}
  </g>
</svg>'''
                
                print(f"📝 SVG content length: {len(full_svg)}")
                
                # Convert
                image = convert_svg_to_image_fixed(full_svg, 64)
                
                if image is not None:
                    print(f"✅ Real kanji image created: {image.size}, {image.mode}")
                    
                    # Analyze
                    img_array = image.convert('RGB')
                    r, g, b = img_array.split()
                    r_extrema = r.getextrema()
                    g_extrema = g.getextrema()
                    b_extrema = b.getextrema()
                    
                    print(f"📊 Color ranges: R{r_extrema}, G{g_extrema}, B{b_extrema}")
                    
                    # Save
                    output_path = f"fixed_real_kanji_{target_unicode}.png"
                    image.save(output_path)
                    print(f"💾 Fixed real kanji image saved to: {output_path}")
                    
                    return True
                else:
                    print("❌ Real kanji conversion failed!")
                    return False
    
    print(f"❌ Target kanji {target_unicode} not found!")
    return False

def create_fixed_dataset_builder():
    """Create a fixed version of the dataset builder"""
    
    print(f"\n🔧 Creating Fixed Dataset Builder")
    print("=" * 50)
    
    # Create the fixed conversion function
    fixed_code = '''
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
'''
    
    # Save the fixed code
    with open("fixed_conversion_function.py", "w") as f:
        f.write(fixed_code)
    
    print("💾 Fixed conversion function saved to: fixed_conversion_function.py")
    print("📝 Replace the convert_svg_to_image_fixed method in fix_kanji_dataset.py with this version")

def main():
    """Main function"""
    print("🎌 Fixed SVG Conversion")
    print("=" * 50)
    
    # Test simple conversion
    success1 = test_fixed_conversion()
    
    # Test real kanji conversion
    success2 = test_real_kanji_fixed()
    
    # Create fixed dataset builder
    create_fixed_dataset_builder()
    
    print(f"\n🎉 Fixed conversion complete!")
    if success1 and success2:
        print(f"   ✅ Both tests passed!")
        print(f"   📁 Check fixed_*.png files for results")
        print(f"   🔧 Use the fixed conversion function to rebuild the dataset")
    else:
        print(f"   ❌ Some tests failed!")
        print(f"   🔧 Check the conversion process")

if __name__ == "__main__":
    main()
