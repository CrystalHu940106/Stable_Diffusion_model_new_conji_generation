#!/usr/bin/env python3
"""
Debug SVG to Image Conversion
"""

import xml.etree.ElementTree as ET
import os
import re
from PIL import Image
import tempfile
import subprocess
from pathlib import Path

def test_svg_conversion():
    """Test SVG to image conversion with a simple example"""
    
    print("🔍 Testing SVG to Image Conversion")
    print("=" * 50)
    
    # Create a simple test SVG
    test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 109 109">
  <g style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
    <path d="M30,30 L80,30 M30,50 L80,50 M30,70 L80,70"/>
  </g>
</svg>'''
    
    print("📝 Test SVG created")
    print(test_svg)
    
    # Test conversion
    try:
        # Create temporary SVG file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            f.write(test_svg)
            svg_file = f.name
        
        print(f"📁 SVG file created: {svg_file}")
        
        # Create temporary PNG file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            png_file = f.name
        
        print(f"📁 PNG file will be: {png_file}")
        
        # Convert using rsvg-convert
        cmd = ['rsvg-convert', '-w', '64', '-h', '64', svg_file, '-o', png_file]
        print(f"🔄 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"📊 Return code: {result.returncode}")
        if result.stdout:
            print(f"📤 Stdout: {result.stdout}")
        if result.stderr:
            print(f"📤 Stderr: {result.stderr}")
        
        if result.returncode == 0:
            # Load the PNG file
            image = Image.open(png_file)
            print(f"✅ Image loaded: {image.size}, {image.mode}")
            
            # Analyze the image
            img_array = image.convert('RGB')
            r, g, b = img_array.split()
            r_extrema = r.getextrema()
            g_extrema = g.getextrema()
            b_extrema = b.getextrema()
            
            print(f"📊 Color ranges: R{r_extrema}, G{g_extrema}, B{b_extrema}")
            
            # Save for inspection
            output_path = "debug_test_image.png"
            image.save(output_path)
            print(f"💾 Test image saved to: {output_path}")
            
        else:
            print("❌ Conversion failed!")
            
        # Clean up
        os.unlink(svg_file)
        if os.path.exists(png_file):
            os.unlink(png_file)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_real_kanji_svg():
    """Test with a real Kanji SVG from the dataset"""
    
    print(f"\n🔍 Testing Real Kanji SVG")
    print("=" * 50)
    
    # Parse KanjiVG to get a real SVG
    kanjivg_path = "data/kanjivg-20220427.xml"
    
    if not Path(kanjivg_path).exists():
        print(f"❌ {kanjivg_path} not found!")
        return
    
    print(f"📁 Loading KanjiVG from: {kanjivg_path}")
    
    # Parse the XML file
    tree = ET.parse(kanjivg_path)
    root = tree.getroot()
    
    # Find the first kanji
    first_kanji = root.find('.//kanji')
    if first_kanji is None:
        print("❌ No kanji found in file!")
        return
    
    kanji_id = first_kanji.get('id')
    print(f"📝 Found kanji: {kanji_id}")
    
    # Extract the character from the ID
    match = re.search(r'kvg:kanji_([0-9a-f]+)', kanji_id)
    if match:
        unicode_hex = match.group(1)
        try:
            unicode_int = int(unicode_hex, 16)
            kanji_char = chr(unicode_int)
            print(f"🔤 Kanji character: {kanji_char} (U+{unicode_hex})")
            
            # Get the SVG content
            g_element = first_kanji.find('.//g')
            if g_element is not None:
                svg_content = ET.tostring(g_element, encoding='unicode')
                
                # Create full SVG document
                full_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 109 109">
  <g style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
    {svg_content}
  </g>
</svg>'''
                
                print(f"📝 SVG content length: {len(full_svg)}")
                print(f"📝 SVG preview: {full_svg[:200]}...")
                
                # Test conversion
                try:
                    # Create temporary SVG file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                        f.write(full_svg)
                        svg_file = f.name
                    
                    # Create temporary PNG file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        png_file = f.name
                    
                    # Convert using rsvg-convert
                    cmd = ['rsvg-convert', '-w', '64', '-h', '64', svg_file, '-o', png_file]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    print(f"📊 Return code: {result.returncode}")
                    if result.stderr:
                        print(f"📤 Stderr: {result.stderr}")
                    
                    if result.returncode == 0:
                        # Load the PNG file
                        image = Image.open(png_file)
                        print(f"✅ Image loaded: {image.size}, {image.mode}")
                        
                        # Analyze the image
                        img_array = image.convert('RGB')
                        r, g, b = img_array.split()
                        r_extrema = r.getextrema()
                        g_extrema = g.getextrema()
                        b_extrema = b.getextrema()
                        
                        print(f"📊 Color ranges: R{r_extrema}, G{g_extrema}, B{b_extrema}")
                        
                        # Save for inspection
                        output_path = f"debug_real_kanji_{unicode_hex}.png"
                        image.save(output_path)
                        print(f"💾 Real kanji image saved to: {output_path}")
                        
                    else:
                        print("❌ Conversion failed!")
                        
                    # Clean up
                    os.unlink(svg_file)
                    if os.path.exists(png_file):
                        os.unlink(png_file)
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except (ValueError, UnicodeEncodeError) as e:
            print(f"❌ Error processing unicode: {e}")

def main():
    """Main function"""
    print("🎌 SVG Conversion Debug")
    print("=" * 50)
    
    # Test simple SVG
    test_svg_conversion()
    
    # Test real kanji SVG
    test_real_kanji_svg()
    
    print(f"\n🎉 Debug complete!")
    print(f"   • Check debug_*.png files for results")

if __name__ == "__main__":
    main()
