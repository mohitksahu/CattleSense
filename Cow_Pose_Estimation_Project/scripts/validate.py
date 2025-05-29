from ultralytics import YOLO
import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Validate YOLO model for cow pose estimation')
    parser.add_argument('--weights', type=str, default='runs/cow_pose_detection/weights/best.pt', 
                        help='model weights path')
    parser.add_argument('--data', type=str, default='data.yml', help='data.yml path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    args = parser.parse_args()
    
    # Load the trained model
    model = YOLO(args.weights)
    
    # Validate the model
    metrics = model.val(data=args.data, imgsz=args.img_size)
    
    print(f"Validation results: {metrics}")
    
    # Create visualization directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot metrics
    metrics_dict = metrics.box
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metrics_dict)), list(metrics_dict.values()), align='center')
    plt.xticks(range(len(metrics_dict)), list(metrics_dict.keys()), rotation=45)
    plt.title('Validation Metrics')
    plt.tight_layout()
    plt.savefig('visualizations/validation_metrics.png')
    print("Validation metrics visualization saved to visualizations/validation_metrics.png")
    
    # Optional: Plot some example detections
    val_images_dir = os.path.join('Cow Pose Estimation', 'images', 'val')
    if os.path.exists(val_images_dir):
        sample_images = os.listdir(val_images_dir)[:5]  # Get first 5 validation images
        
        for i, img_file in enumerate(sample_images):
            img_path = os.path.join(val_images_dir, img_file)
            results = model(img_path)
            
            # Plot results
            fig = results[0].plot()
            plt.figure(figsize=(10, 8))
            plt.imshow(fig)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'visualizations/detection_example_{i+1}.png')
            
        print(f"Detection examples saved to visualizations/ directory")

if __name__ == '__main__':
    main()