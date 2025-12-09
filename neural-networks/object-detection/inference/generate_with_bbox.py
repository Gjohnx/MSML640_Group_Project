import argparse
from pathlib import Path
from ultralytics import YOLO
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path',type=str)
    parser.add_argument('--model',type=str,default='best.pt')
    parser.add_argument('--output',type=str,default=None)
    parser.add_argument('--conf',type=float,default=0.25)
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        script_dir = Path(__file__).parent
        model_path = script_dir / args.model
        if not model_path.exists():
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
    
    print(f"Loading model: {model_path}")
    print(f"Processing image: {image_path}")
    print(f"Confidence threshold: {args.conf}")
    
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print("\nRunning inference...")
    try:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            save=False,
            verbose=False
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        print(f"\nNo detections found with confidence >= {args.conf}")
    else:
        num_detections = len(result.boxes)
        print(f"\nFound {num_detections} detection(s)")
        
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls] if cls < len(model.names) else f"Class {cls}"
            print(f"  Detection {i+1}: {class_name} (confidence: {conf:.2f})")
    
    print(f"Saving annotated image to: {output_path}")
    try:
        annotated_image = result.plot()
        
        from PIL import Image
        im = Image.fromarray(annotated_image)
        im.save(str(output_path))
        
        print(f"Output: {output_path}")
    except Exception as e:
        print(f"Error saving annotated image: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()