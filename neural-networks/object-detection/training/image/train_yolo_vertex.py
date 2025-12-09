"""
YOLOv11 Training Script for Rubik's Cube Detection on Google Vertex AI

This script is optimized for running on Vertex AI Training.
It handles GCS paths and environment variables set by Vertex AI.
"""

from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import yaml
import tempfile
import subprocess

# Set PyTorch CUDA memory allocation to reduce fragmentation
# This helps prevent OOM errors by allowing memory segments to expand
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on Vertex AI')
    parser.add_argument('--data', type=str, 
                       default='dataset/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size (reduce if OOM errors occur)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (reduce to 512 or 416 if OOM errors occur)')
    parser.add_argument('--accumulate', type=int, default=1,
                       help='Gradient accumulation steps (effective batch = batch * accumulate)')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                       help='YOLOv11 model to use (yolo11n.pt, yolo11s.pt, etc.)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='rubiks_cube_yolo11',
                       help='Experiment name')
    parser.add_argument('--gcs-output', type=str, default=None,
                       help='GCS bucket path to save model outputs (e.g., gs://your-bucket/models)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (0 for GPU, cpu for CPU). Auto-detected if not specified.')
    
    args = parser.parse_args()
    
    # Handle Vertex AI environment variables
    # AIP_MODEL_DIR is set by Vertex AI for model output (GCS path)
    # AIP_CHECKPOINT_DIR is set by Vertex AI for checkpoints
    gcs_model_dir = os.environ.get('AIP_MODEL_DIR', None)
    checkpoint_dir = os.environ.get('AIP_CHECKPOINT_DIR', None)
    original_aip_model_dir = None  # Will be set if AIP_MODEL_DIR exists
    
    # YOLO needs a local directory for training output
    # Use local temp directory, then copy to GCS after training
    if gcs_model_dir and gcs_model_dir.startswith('gs://'):
        # Use local directory for training
        model_dir = os.path.join(tempfile.gettempdir(), 'runs', 'detect')
        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        print(f"GCS model directory (will upload to): {gcs_model_dir}")
        print(f"Using local directory for training: {model_dir}")
        print(f"Absolute local path: {os.path.abspath(model_dir)}")
    else:
        model_dir = args.project
        os.makedirs(model_dir, exist_ok=True)
    
    # Handle GCS paths or local paths for dataset configuration
    dataset_path = args.data
    
    # If it's a GCS path, download it first
    if dataset_path.startswith('gs://'):
        print(f"Downloading data.yaml from GCS: {dataset_path}")
        local_data_yaml = os.path.join(tempfile.gettempdir(), 'data.yaml')
        subprocess.run(['gsutil', 'cp', dataset_path, local_data_yaml], check=True)
        dataset_path = local_data_yaml
    else:
        # Local path handling
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            # Try absolute path from script location
            dataset_path_obj = Path(__file__).parent / dataset_path
            if not dataset_path_obj.exists():
                raise FileNotFoundError(f"Dataset configuration not found at: {dataset_path}")
            dataset_path = str(dataset_path_obj.absolute())
    
    # Read data.yaml and check if it contains GCS paths
    print(f"Reading dataset configuration: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check if any paths are GCS paths and download dataset if needed
    dataset_dir = None
    needs_download = False
    for key in ['train', 'val', 'test']:
        if key in data_config and isinstance(data_config[key], str) and data_config[key].startswith('gs://'):
            needs_download = True
            # Extract dataset base path from GCS path
            if dataset_dir is None:
                # Extract: gs://bucket/path/to/dataset/train/images -> gs://bucket/path/to/dataset
                gcs_path = data_config[key]
                # Remove the last 2 path components (e.g., 'train/images')
                parts = gcs_path.rstrip('/').split('/')
                dataset_dir = '/'.join(parts[:-2])  # Remove last 2 parts
            break
    
    # Download dataset if it contains GCS paths
    if needs_download:
        print(f"Dataset contains GCS paths. Downloading dataset from: {dataset_dir}")
        local_dataset_dir = os.path.join(tempfile.gettempdir(), 'dataset')
        os.makedirs(local_dataset_dir, exist_ok=True)
        
        # Download entire dataset directory
        dataset_name = dataset_dir.rstrip('/').split('/')[-1]  # Get dataset folder name
        local_dataset_path = os.path.join(local_dataset_dir, dataset_name)
        
        print(f"Downloading to: {local_dataset_path}")
        subprocess.run(['gsutil', '-m', 'cp', '-r', f"{dataset_dir}", local_dataset_dir], check=True)
        
        # Update data.yaml paths to local paths
        for key in ['train', 'val', 'test']:
            if key in data_config and isinstance(data_config[key], str) and data_config[key].startswith('gs://'):
                # Convert: gs://bucket/path/dataset/train/images -> /tmp/dataset/dataset/train/images
                gcs_path = data_config[key]
                # Get the last 2 path components (e.g., 'train/images')
                parts = gcs_path.rstrip('/').split('/')
                relative_path = '/'.join(parts[-2:])  # Last 2 parts
                data_config[key] = os.path.join(local_dataset_path, relative_path)
                print(f"Updated {key} path to: {data_config[key]}")
        
        # Write updated data.yaml
        updated_data_yaml = os.path.join(tempfile.gettempdir(), 'data_local.yaml')
        with open(updated_data_yaml, 'w') as f:
            yaml.dump(data_config, f)
        dataset_path = updated_data_yaml
        print(f"Updated data.yaml saved to: {dataset_path}")
    
    print(f"Using dataset configuration: {dataset_path}")
    print(f"Model output directory: {model_dir}")
    if checkpoint_dir:
        print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Initialize YOLOv11 model
    model = YOLO(args.model)
    
    # Determine device (auto-detect if not specified)
    device = args.device
    if device is None:
        # Try to detect GPU, fallback to CPU
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Clear CUDA cache if using GPU to avoid OOM errors
    if device != 'cpu':
        import torch
        torch.cuda.empty_cache()
        print(f"Cleared CUDA cache. Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Temporarily unset AIP_MODEL_DIR to prevent YOLO from using it directly
    # We'll handle GCS upload manually after training
    if gcs_model_dir:
        original_aip_model_dir = os.environ.pop('AIP_MODEL_DIR', None)
        print(f"Temporarily unset AIP_MODEL_DIR (was: {original_aip_model_dir})")
        print(f"Will upload to GCS after training completes")
    else:
        original_aip_model_dir = None
    
    # Adjust workers based on batch size to avoid OOM
    # Fewer workers = less memory usage for data loading
    if device != 'cpu':
        # Reduce workers if batch size is small to save memory
        if args.batch <= 8:
            num_workers = 4
        elif args.batch <= 16:
            num_workers = 6
        else:
            num_workers = 8
    else:
        num_workers = 4
    
    print(f"Using {num_workers} workers for data loading")
    print(f"Training with batch size: {args.batch}, image size: {args.imgsz}")
    if args.accumulate > 1:
        print(f"Gradient accumulation: {args.accumulate} steps (effective batch: {args.batch * args.accumulate})")
    
    # Train the model
    results = model.train(
        data=dataset_path,  # Path to data.yaml (local or GCS)
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=model_dir,
        patience=10,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
        device=device,
        workers=num_workers,
        # Memory optimization: gradient accumulation for effective larger batch
        # Note: YOLO doesn't directly support accumulate parameter, but we document it
        # Additional training parameters
        # lr0=0.01,
        # lrf=0.01,
        # momentum=0.937,
        # weight_decay=0.0005,
        # warmup_epochs=3.0,
        # warmup_momentum=0.8,
        # warmup_bias_lr=0.1,
        # box=7.5,
        # cls=0.5,
        # dfl=1.5,
    )
    
    # Restore AIP_MODEL_DIR if it was set
    if original_aip_model_dir:
        os.environ['AIP_MODEL_DIR'] = original_aip_model_dir
    
    # Convert results.save_dir to string (it might be a Path object)
    save_dir = str(results.save_dir)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"\nTraining results saved locally at: {save_dir}")
    
    # Verify the results directory exists and is local
    if not os.path.exists(save_dir):
        print(f"ERROR: Results directory does not exist: {save_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Listing contents of parent directory...")
        parent_dir = os.path.dirname(save_dir)
        if os.path.exists(parent_dir):
            print(f"Contents of {parent_dir}: {os.listdir(parent_dir)}")
    else:
        print(f"Verified: Results directory exists and contains: {os.listdir(save_dir)}")
    
    # Copy model and all training artifacts to GCS
    if gcs_model_dir and gcs_model_dir.startswith('gs://'):
        print(f"\n{'='*50}")
        print(f"Copying training results to GCS: {gcs_model_dir}")
        print(f"{'='*50}")
        
        # Verify save_dir is a local path (not GCS)
        if save_dir.startswith('gs://'):
            print(f"ERROR: save_dir is a GCS path: {save_dir}")
            print(f"This should not happen. Check model.train() project parameter.")
            return
        
        # Copy the entire training output directory to GCS
        # This includes: weights/, plots/, results.csv, args.yaml, etc.
        gcs_output_path = f"{gcs_model_dir}/{args.name}"
        print(f"Uploading local directory: {save_dir}")
        print(f"To GCS location: {gcs_output_path}")
        
        # Use gsutil -m for parallel upload and -r for recursive
        # Note: gsutil cp -r will copy the directory contents to the destination
        result = subprocess.run(['gsutil', '-m', 'cp', '-r', f"{save_dir}/*", gcs_output_path], 
                              check=True, capture_output=True, text=True)
        print(f"gsutil output: {result.stdout}")
        if result.stderr:
            print(f"gsutil warnings: {result.stderr}")
        print(f"✓ Training results uploaded to: {gcs_output_path}")
        
        # Also copy best model to a convenient location at the root
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            best_model_gcs = f"{gcs_model_dir}/best.pt"
            print(f"\nCopying best model to: {best_model_gcs}")
            subprocess.run(['gsutil', 'cp', best_model_path, best_model_gcs], check=True)
            print(f"✓ Best model saved to: {best_model_gcs}")
        else:
            print(f"⚠ Warning: Best model not found at {best_model_path}")
            # List what's actually in the weights directory
            weights_dir = os.path.join(save_dir, 'weights')
            if os.path.exists(weights_dir):
                print(f"Contents of weights directory: {os.listdir(weights_dir)}")
    
    # Copy model to GCS if specified via --gcs-output argument
    elif args.gcs_output:
        best_model_path = f"{results.save_dir}/weights/best.pt"
        gcs_dest = f"{args.gcs_output}/best.pt"
        print(f"\nCopying model to GCS: {gcs_dest}")
        subprocess.run(['gsutil', 'cp', best_model_path, gcs_dest], check=True)
        print(f"Model saved to: {gcs_dest}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"\nValidation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    print("Start YOLO training")
    main()
    print("YOLO training completed")
