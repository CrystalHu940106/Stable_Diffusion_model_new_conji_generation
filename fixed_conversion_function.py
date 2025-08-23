
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
