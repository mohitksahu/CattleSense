import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from ultralytics import YOLO

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_directories(dirs):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def visualize_training_results(results_csv_path, save_path=None):
    """Visualize training metrics from results.csv file."""
    import pandas as pd
    
    # Read training results
    df = pd.read_csv(results_csv_path)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Loss plots
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation losses
    axes[0, 1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
    axes[0, 1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
    axes[0, 1].set_title('Validation Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Metrics
    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[1, 0].set_title('Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training visualization saved to {save_path}")
    
    return fig

def plot_class_distribution(labels_dir, class_names, save_path=None):
    """Plot distribution of classes in the dataset."""
    class_counts = {}
    
    # Initialize counts
    for class_name in class_names:
        class_counts[class_name] = 0
    
    # Count classes in all label files
    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id < len(class_names):
                        class_counts[class_names[class_id]] += 1
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    return plt.gcf()

def visualize_predictions(image_path, model_path, class_names, conf_threshold=0.25, save_path=None):
    """Visualize model predictions on an image."""
    # Load model
    model = YOLO(model_path)
    
    # Load and predict
    results = model(image_path, conf=conf_threshold)
    
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw predictions
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence and class
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw label
                label = f"{class_names[cls]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (255, 0, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f'Predictions on {os.path.basename(image_path)}')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    
    return plt.gcf()

def calculate_dataset_stats(images_dir):
    """Calculate basic statistics about the dataset."""
    image_files = list(Path(images_dir).glob('**/*.jpg')) + \
                  list(Path(images_dir).glob('**/*.jpeg')) + \
                  list(Path(images_dir).glob('**/*.png'))
    
    if not image_files:
        print("No images found in the directory.")
        return None
    
    widths, heights = [], []
    file_sizes = []
    
    for img_path in image_files:
        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
        
        # Get file size
        file_sizes.append(img_path.stat().st_size / (1024 * 1024))  # MB
    
    stats = {
        'total_images': len(image_files),
        'avg_width': np.mean(widths),
        'avg_height': np.mean(heights),
        'min_width': np.min(widths),
        'max_width': np.max(widths),
        'min_height': np.min(heights),
        'max_height': np.max(heights),
        'avg_file_size_mb': np.mean(file_sizes),
        'total_size_mb': np.sum(file_sizes)
    }
    
    return stats

def print_dataset_info(config_path):
    """Print comprehensive dataset information."""
    config = load_config(config_path)
    
    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    print(f"Dataset path: {config['path']}")
    print(f"Number of classes: {config['nc']}")
    print(f"Class names: {config['names']}")
    
    # Training set info
    train_path = os.path.join(config['path'], config['train'])
    if os.path.exists(train_path):
        train_stats = calculate_dataset_stats(train_path)
        if train_stats:
            print(f"\nTraining Set:")
            print(f"  Total images: {train_stats['total_images']}")
            print(f"  Average dimensions: {train_stats['avg_width']:.0f} x {train_stats['avg_height']:.0f}")
            print(f"  Size range: {train_stats['min_width']}x{train_stats['min_height']} to {train_stats['max_width']}x{train_stats['max_height']}")
            print(f"  Total size: {train_stats['total_size_mb']:.2f} MB")
    
    # Validation set info
    val_path = os.path.join(config['path'], config['val'])
    if os.path.exists(val_path):
        val_stats = calculate_dataset_stats(val_path)
        if val_stats:
            print(f"\nValidation Set:")
            print(f"  Total images: {val_stats['total_images']}")
            print(f"  Average dimensions: {val_stats['avg_width']:.0f} x {val_stats['avg_height']:.0f}")
            print(f"  Size range: {val_stats['min_width']}x{val_stats['min_height']} to {val_stats['max_width']}x{val_stats['max_height']}")
            print(f"  Total size: {val_stats['total_size_mb']:.2f} MB")
    
    print("="*60)

def validate_dataset_structure(config_path):
    """Validate that dataset structure matches configuration."""
    config = load_config(config_path)
    
    issues = []
    
    # Check if paths exist
    dataset_path = config['path']
    if not os.path.exists(dataset_path):
        issues.append(f"Dataset path does not exist: {dataset_path}")
        return issues
    
    train_images = os.path.join(dataset_path, config['train'])
    val_images = os.path.join(dataset_path, config['val'])
    
    if not os.path.exists(train_images):
        issues.append(f"Training images path does not exist: {train_images}")
    
    if not os.path.exists(val_images):
        issues.append(f"Validation images path does not exist: {val_images}")
    
    # Check for corresponding labels
    train_labels = train_images.replace('images', 'labels')
    val_labels = val_images.replace('images', 'labels')
    
    if not os.path.exists(train_labels):
        issues.append(f"Training labels path does not exist: {train_labels}")
    
    if not os.path.exists(val_labels):
        issues.append(f"Validation labels path does not exist: {val_labels}")
    
    if not issues:
        print("✅ Dataset structure validation passed!")
    else:
        print("❌ Dataset structure validation failed:")
        for issue in issues:
            print(f"  - {issue}")
    
    return issues