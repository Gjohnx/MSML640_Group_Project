import os
from pathlib import Path
from PIL import Image

def resize_and_crop_image(input_path, output_path, target_size=(700, 700)):
    img = Image.open(input_path)
    
    original_width, original_height = img.size
    
    aspect_ratio = original_width / original_height
    
    target_width, target_height = target_size
    
    new_width = int(target_height * aspect_ratio)
    resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    if new_width > target_width:
        left = (new_width - target_width) // 2
        right = left + target_width
        cropped_img = resized_img.crop((left, 0, right, target_height))
    elif new_width < target_width:
        new_height = int(target_width / aspect_ratio)
        resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        top = (new_height - target_height) // 2
        bottom = top + target_height
        cropped_img = resized_img.crop((0, top, target_width, bottom))
    else:
        cropped_img = resized_img
    
    cropped_img.save(output_path, quality=95)
    print(f"Processed: {input_path.name} -> {output_path.name} ({cropped_img.size[0]}x{cropped_img.size[1]})")


def process_all_images(input_dir, output_dir=None, target_size=(700, 700)):
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix in image_extensions and f.is_file()]
    
    print(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        output_file = output_path / img_file.name
        try:
            resize_and_crop_image(img_file, output_file, target_size)
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    print(f"Completed processing {len(image_files)} images")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    backgrounds_dir = script_dir / "backgrounds_unsplash"
    
    if not backgrounds_dir.exists():
        print(f"Error: Directory {backgrounds_dir} does not exist")
    else:
        process_all_images(backgrounds_dir, target_size=(700, 700))
