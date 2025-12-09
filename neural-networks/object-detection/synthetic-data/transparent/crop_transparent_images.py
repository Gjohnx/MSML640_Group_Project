#!/usr/bin/env python3
# Script to crop images to their content, removing transparent margins

import os
from pathlib import Path
from PIL import Image
import numpy as np


def get_bounding_box(image):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Get alpha channel (last channel)
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
    else:
        # No alpha channel, return full image bounds
        h, w = img_array.shape[:2]
        return (0, 0, w, h)
    
    # Find rows and columns with non-transparent pixels
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Image is fully transparent
        return None
    
    # Get bounding box coordinates
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])
    
    return (left, top, right, bottom)


def crop_image(input_path, output_path=None):
    try:
        # Load image
        image = Image.open(input_path)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get bounding box
        bbox = get_bounding_box(image)
        
        if bbox is None:
            print(f"  Warning: {input_path.name} appears to be fully transparent, skipping")
            return False
        
        left, top, right, bottom = bbox
        
        # Check if cropping is needed
        width, height = image.size
        if left == 0 and top == 0 and right == width and bottom == height:
            print(f" No cropping needed: {input_path.name}")
            return True
        
        # Crop image
        cropped = image.crop((left, top, right, bottom))
        
        # Save cropped image
        if output_path is None:
            output_path = input_path
        
        cropped.save(output_path, 'PNG')
        
        # Print info
        original_size = width * height
        new_size = cropped.width * cropped.height
        reduction = (1 - new_size / original_size) * 100
        
        print(f"Cropped: {input_path.name}")
        print(f"Original: {width}x{height} ({original_size:,} pixels)")
        print(f"Cropped:  {cropped.width}x{cropped.height} ({new_size:,} pixels)")
        print(f"Reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    png_files = sorted(script_dir.glob('*.png'))
    
    if not png_files:
        print(f"No PNG files found in {script_dir}")
        return
    
    print(f"Found {len(png_files)} PNG file(s) to process")
    print(f"Processing images in: {script_dir}\n")
    
    processed = 0
    skipped = 0
    failed = 0
    
    # Process each image
    for png_file in png_files:
        if crop_image(png_file):
            processed += 1
        else:
            failed += 1
    
    print(f"Processed successfully: {processed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(png_files)}")


if __name__ == '__main__':
    main()
