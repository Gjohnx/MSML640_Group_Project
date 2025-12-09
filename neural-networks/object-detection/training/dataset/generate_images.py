# This script takes the images from the dataset and draws the bounding boxes on them.

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

def yolo_to_bbox(yolo_coords, img_width, img_height):
    center_x, center_y, width, height = yolo_coords
    
    center_x_px = center_x * img_width
    center_y_px = center_y * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = center_x_px - (width_px / 2)
    y1 = center_y_px - (height_px / 2)
    x2 = center_x_px + (width_px / 2)
    y2 = center_y_px + (height_px / 2)
    
    return (int(x1), int(y1), int(x2), int(y2))

def draw_bounding_boxes(image_path, label_path, output_path):
    try:
        # Load the image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        # Read label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Process each bounding box
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            parts = line.split()
            if len(parts) < 5:
                print(f"  Warning: Invalid line format in {label_path}: {line}")
                continue
            
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            bbox = yolo_to_bbox((center_x, center_y, width, height), img_width, img_height)
            x1, y1, x2, y2 = bbox
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            label_text = f"Class {class_id}"
            # Get text bounding box for background
            try:
                bbox_text = draw.textbbox((x1, y1 - 20), label_text, font=font)
            except AttributeError:
                bbox_text = draw.textsize(label_text, font=font)
                bbox_text = (x1, y1 - 20, x1 + bbox_text[0], y1 - 20 + bbox_text[1])
            
            draw.rectangle(bbox_text, fill="red")
            draw.text((x1, y1 - 20), label_text, fill="white", font=font)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img.save(output_path)
        return True
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return False

def find_corresponding_image(label_file, images_dir):
    # Get the base name without extension
    label_stem = label_file.stem
    
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_path = images_dir / f"{label_stem}{ext}"
        if image_path.exists():
            return image_path
    
    return None

def main():
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Define the directories to process
    datasets = [
        ('test', script_dir / 'test'),
        ('train', script_dir / 'train'),
        ('valid', script_dir / 'valid')
    ]
    
    # Output directory
    output_base = script_dir / 'generated'
    
    total_labels = 0
    processed = 0
    skipped = 0
    failed = 0
    
    # Process each dataset
    for dataset_name, dataset_dir in datasets:
        labels_dir = dataset_dir / 'labels'
        images_dir = dataset_dir / 'images'
        output_dir = output_base / dataset_name / 'images'
        
        if not labels_dir.exists():
            print(f"Warning: Labels directory {labels_dir} does not exist. Skipping...")
            continue
        
        if not images_dir.exists():
            print(f"Warning: Images directory {images_dir} does not exist. Skipping...")
            continue
        
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Get all label files
        label_files = list(labels_dir.glob('*.txt'))
        total_labels += len(label_files)
        
        # Process each label file
        for label_file in label_files:
            # Find corresponding image
            image_path = find_corresponding_image(label_file, images_dir)
            
            if image_path is None:
                print(f"  Skipped: No corresponding image found for {label_file.name}")
                skipped += 1
                continue
            
            output_path = output_dir / image_path.name
            
            # Draw bounding boxes and save
            if draw_bounding_boxes(image_path, label_file, output_path):
                processed += 1
                print(f"  Generated: {output_path.name}")
            else:
                failed += 1
                print(f"  Failed: {label_file.name}")
    
    # Print summary
    print(f"Total label files: {total_labels}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (no image): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_base}")

if __name__ == '__main__':
    main()
