"""
Train YOLO11 Segmentation Model for Car Damage Detection
Usage: python train.py [--epochs 50] [--batch 16] [--imgsz 640] [--model yolo11m-seg.pt]
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml: str = "carDamagedDataSet/data.yaml",
    model_name: str = "yolo11m-seg.pt",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "car_damage_seg",
    resume: bool = False
):
    """
    Train YOLO11 segmentation model for car damage detection
    
    Args:
        data_yaml: Path to dataset YAML file
        model_name: Pretrained model to start from (yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg)
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Input image size
        device: CUDA device (e.g., '0' or 'cpu')
        project: Project directory for saving runs
        name: Name of this training run
        resume: Resume from last checkpoint
    """
    
    print("=" * 70)
    print("🚀 Car Damage Detection - Model Training")
    print("=" * 70)
    print(f"📦 Model: {model_name}")
    print(f"📊 Dataset: {data_yaml}")
    print(f"🔢 Epochs: {epochs}")
    print(f"📏 Batch Size: {batch}")
    print(f"🖼️  Image Size: {imgsz}")
    print(f"💻 Device: {device}")
    print("=" * 70)
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    # Load model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        
        # Optimization parameters
        patience=100,          # Early stopping patience
        save=True,             # Save checkpoints
        save_period=10,        # Save checkpoint every N epochs
        
        # Segmentation specific
        retina_masks=True,     # Higher resolution masks
        
        # Augmentation
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation
        hsv_v=0.4,            # HSV-Value
        degrees=0.0,          # Rotation degrees
        translate=0.1,        # Translation
        scale=0.5,            # Scale
        shear=0.0,            # Shear
        perspective=0.0,      # Perspective
        flipud=0.0,           # Flip up-down
        fliplr=0.5,           # Flip left-right
        mosaic=0.5,           # Mosaic augmentation
        mixup=0.0,            # Mixup augmentation
        copy_paste=0.3,       # Copy-paste augmentation
        
        # Validation
        val=True,             # Validate during training
        plots=True,           # Save training plots
        
        # Loss weights
        box=7.5,              # Box loss weight
        cls=0.5,              # Class loss weight
        dfl=1.5,              # DFL loss weight
        
        # Optimizer
        optimizer='auto',      # Optimizer (auto, SGD, Adam, AdamW)
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate
        momentum=0.937,       # SGD momentum
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,   # Warmup bias learning rate
        
        # Other
        verbose=True,         # Verbose output
        seed=0,               # Random seed for reproducibility
        deterministic=True,   # Deterministic mode
    )
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Get best model path
    best_weights = Path(project) / name / "weights" / "best.pt"
    last_weights = Path(project) / name / "weights" / "last.pt"
    
    print(f"📁 Best weights: {best_weights}")
    print(f"📁 Last weights: {last_weights}")
    print(f"📊 Results saved to: {Path(project) / name}")
    
    # Validate the best model
    print("\n" + "=" * 70)
    print("🔍 Validating best model...")
    print("=" * 70)
    
    best_model = YOLO(str(best_weights))
    metrics = best_model.val()
    
    print("\n📊 Final Metrics:")
    print(f"   mAP50: {metrics.seg.map50:.4f}")
    print(f"   mAP50-95: {metrics.seg.map:.4f}")
    
    return str(best_weights), results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 for Car Damage Detection")
    
    parser.add_argument("--data", type=str, default="carDamagedDataSet/data.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default="yolo11m-seg.pt",
                        help="Model size: yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0, 1, 2, ...) or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory for saving runs")
    parser.add_argument("--name", type=str, default="car_damage_seg",
                        help="Name of this training run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    # Update data.yaml paths for local training
    print("\n🔧 Updating data.yaml paths for local training...")
    import yaml
    
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths to be relative to current directory
    base_dir = Path("carDamagedDataSet")
    data_config['train'] = str(base_dir / "train" / "images")
    data_config['val'] = str(base_dir / "valid" / "images")
    data_config['test'] = str(base_dir / "test" / "images")
    
    # Save updated config to temp file
    temp_yaml = "data_local.yaml"
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"✅ Using updated config: {temp_yaml}")
    
    # Train the model
    best_weights, results = train_model(
        data_yaml=temp_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    print("\n" + "=" * 70)
    print("🎉 All done! Your model is ready for inference.")
    print(f"💡 Use this weights file: {best_weights}")
    print("=" * 70)

