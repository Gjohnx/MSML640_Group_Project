import random
import shutil
from pathlib import Path

def split_dataset(generated_dir, train_ratio=0.70, test_ratio=0.15, valid_ratio=0.15):
    generated_path = Path(generated_dir)
    images_dir = generated_path / "images"
    labels_dir = generated_path / "labels"
    
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return
    
    if not labels_dir.exists():
        print(f"Error: Labels directory {labels_dir} does not exist")
        return
    
    if abs(train_ratio + test_ratio + valid_ratio - 1.0) > 0.001:
        print(f"Error: Ratios must sum to 1.0 (got {train_ratio + test_ratio + valid_ratio})")
        return
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
    
    print(f"Found {len(image_files)} images")
    
    # Filter to only include images that have corresponding labels
    image_label_pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            image_label_pairs.append((img_file, label_file))
        else:
            print(f"Warning: No label file found for {img_file.name}, skipping")
    
    print(f"Found {len(image_label_pairs)} image-label pairs")
    
    random.shuffle(image_label_pairs)
    
    total = len(image_label_pairs)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)
    
    train_pairs = image_label_pairs[:train_end]
    test_pairs = image_label_pairs[train_end:test_end]
    valid_pairs = image_label_pairs[test_end:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/total*100:.1f}%)")
    print(f"  Test: {len(test_pairs)} ({len(test_pairs)/total*100:.1f}%)")
    print(f"  Valid: {len(valid_pairs)} ({len(valid_pairs)/total*100:.1f}%)")
    
    output_base = generated_path.parent
    
    for split_name, pairs in [("train", train_pairs), ("test", test_pairs), ("valid", valid_pairs)]:
        split_images_dir = output_base / split_name / "images"
        split_labels_dir = output_base / split_name / "labels"
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} set...")
        for img_file, label_file in pairs:
            dest_img = split_images_dir / img_file.name
            shutil.copy2(img_file, dest_img)
            
            dest_label = split_labels_dir / label_file.name
            shutil.copy2(label_file, dest_label)
        
        print(f"  Copied {len(pairs)} images and labels to {split_name}/")
    
    print(f"\nCompleted! Dataset split into train, test, and valid directories.")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    generated_dir = script_dir / "generated"
    
    # Verify directory exists
    if not generated_dir.exists():
        print(f"Error: Generated directory {generated_dir} does not exist")
        exit(1)
    
    split_dataset(generated_dir, train_ratio=0.70, test_ratio=0.15, valid_ratio=0.15)
