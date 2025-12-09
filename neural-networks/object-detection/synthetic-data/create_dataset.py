import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

def get_image_files(directory, extensions=None):
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return [f for f in directory.iterdir() 
            if f.suffix in extensions and f.is_file()]

def calculate_cube_size(background_size, coverage_percentage):
    bg_width, bg_height = background_size
    bg_area = bg_width * bg_height
    target_area = bg_area * coverage_percentage
    
    # Assume cube is roughly square, so we'll use a square approximation
    cube_side = int(np.sqrt(target_area))
    
    return (cube_side, cube_side)

def apply_brightness_contrast(image, brightness_factor, contrast_factor):
    # Apply brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    
    # Apply contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    
    return image

def bbox_to_yolo_format(bbox, image_width, image_height, class_id=0):
    # Calculate center coordinates
    x_center = (bbox['x'] + bbox['width'] / 2.0) / image_width
    y_center = (bbox['y'] + bbox['height'] / 2.0) / image_height
    
    # Normalize width and height
    width_norm = bbox['width'] / image_width
    height_norm = bbox['height'] / image_height
    
    # Format: class_id x_center y_center width height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

def composite_cube_on_background(background, cube, coverage_percentage, brightness_factor, contrast_factor):
    # Create a copy of the background
    result = background.copy()
    
    # Calculate cube size based on coverage percentage
    bg_width, bg_height = background.size
    cube_width, cube_height = calculate_cube_size((bg_width, bg_height), coverage_percentage)
    
    # Resize cube maintaining aspect ratio
    original_cube_width, original_cube_height = cube.size
    aspect_ratio = original_cube_width / original_cube_height
    
    # Resize to fit within the calculated size while maintaining aspect ratio
    if aspect_ratio > 1:
        new_cube_width = cube_width
        new_cube_height = int(cube_width / aspect_ratio)
    else:
        new_cube_width = int(cube_height * aspect_ratio)
        new_cube_height = cube_height
    
    new_cube_width = min(new_cube_width, bg_width)
    new_cube_height = min(new_cube_height, bg_height)
    
    resized_cube = cube.resize((new_cube_width, new_cube_height), Image.Resampling.LANCZOS)
    
    resized_cube = apply_brightness_contrast(resized_cube, brightness_factor, contrast_factor)
    
    max_x = bg_width - new_cube_width
    max_y = bg_height - new_cube_height
    
    if max_x < 0:
        max_x = 0
    if max_y < 0:
        max_y = 0
    
    pos_x = random.randint(0, max_x) if max_x > 0 else 0
    pos_y = random.randint(0, max_y) if max_y > 0 else 0
    
    if resized_cube.mode == 'RGBA':
        result.paste(resized_cube, (pos_x, pos_y), resized_cube)
    else:
        resized_cube = resized_cube.convert('RGBA')
        result.paste(resized_cube, (pos_x, pos_y), resized_cube)
    
    bounding_box = {
        'x': pos_x,
        'y': pos_y,
        'width': new_cube_width,
        'height': new_cube_height
    }
    
    return result, bounding_box

def create_dataset(background_dir, transparent_dir, output_dir):
    # Get all image files
    background_files = get_image_files(background_dir, {'.jpg', '.jpeg', '.JPG', '.JPEG'})
    cube_files = get_image_files(transparent_dir, {'.png', '.PNG'})
    
    print(f"Found {len(background_files)} background images")
    print(f"Found {len(cube_files)} transparent cube images")
    print(f"Will generate {len(background_files) * len(cube_files)} composite images")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each combination
    total_combinations = len(background_files) * len(cube_files)
    processed = 0
    
    for bg_file in background_files:
        try:
            background = Image.open(bg_file)
            if background.mode != 'RGBA':
                background = background.convert('RGBA')
            
            for cube_file in cube_files:
                try:
                    cube = Image.open(cube_file)
                    if cube.mode != 'RGBA':
                        cube = cube.convert('RGBA')
                    
                    coverage_percentage = random.uniform(0.02, 0.30)  # 2% to 30%
                    brightness_factor = random.uniform(0.8, 1.5)  # 80% to 120%
                    contrast_factor = random.uniform(0.9, 1.1)  # Mild contrast adjustment
                    
                    composite, bbox = composite_cube_on_background(
                        background, cube, coverage_percentage, 
                        brightness_factor, contrast_factor
                    )
                    
                    if composite.mode == 'RGBA':
                        composite = composite.convert('RGB')
                    
                    img_width, img_height = composite.size
                    
                    bg_name = bg_file.stem
                    cube_name = cube_file.stem
                    base_filename = f"{bg_name}_{cube_name}"
                    
                    image_filename = f"{base_filename}.jpg"
                    image_file = images_dir / image_filename
                    composite.save(image_file, quality=95)
                    
                    yolo_label = bbox_to_yolo_format(bbox, img_width, img_height, class_id=0)
                    label_filename = f"{base_filename}.txt"
                    label_file = labels_dir / label_filename
                    with open(label_file, 'w') as f:
                        f.write(yolo_label)
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{total_combinations} images...")
                
                except Exception as e:
                    print(f"Error processing cube {cube_file.name}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error processing background {bg_file.name}: {e}")
            continue
    
    print(f"Completed - Generated {processed} images in {images_dir}")
    print(f"Generated {processed} label files in {labels_dir}")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    background_dir = script_dir / "background"
    transparent_dir = script_dir / "transparent"
    output_dir = script_dir / "generated"
    
    if not background_dir.exists():
        print(f"Error: Background directory {background_dir} does not exist")
        exit(1)
    
    if not transparent_dir.exists():
        print(f"Error: Transparent directory {transparent_dir} does not exist")
        exit(1)
    
    create_dataset(background_dir, transparent_dir, output_dir)
