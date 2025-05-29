#!/usr/bin/env python3
"""
Demo script for Cow Pose Estimation Project
Shows how to use the trained model for inference and visualization.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.append('scripts')
from utils import visualize_predictions, save_prediction_results

class CowPoseEstimationDemo:
    def __init__(self, model_path="models/weights/best.pt"):
        """Initialize the demo with a trained model."""
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Standing', 'Lying', 'Eating', 'Walking', 
            'Drinking', 'Grooming', 'Resting'
        ]
        
    def load_model(self):
        """Load the trained YOLO model."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("💡 Please train the model first using: cd scripts && python train.py")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_single_image(self, image_path, confidence=0.5, save_path=None):
        """Run inference on a single image."""
        if not self.model:
            print("❌ Model not loaded")
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        try:
            # Run inference
            results = self.model(image_path, conf=confidence)
            result = results[0]
            
            # Print predictions
            print(f"\n🔍 Predictions for {image_path}:")
            if len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class_{cls_id}"
                    print(f"  📊 Detection {i+1}: {class_name} ({conf:.2f} confidence)")
                    print(f"     📍 BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            else:
                print("  ℹ️  No detections found")
            
            # Save visualization if requested
            if save_path:
                self.save_visualization(result, image_path, save_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def save_visualization(self, result, image_path, save_path):
        """Save prediction visualization."""
        try:
            # Create visualization
            annotated_img = result.plot()
            
            # Convert BGR to RGB for matplotlib
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Save the image
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_img_rgb)
            plt.axis('off')
            plt.title(f'Cow Pose Estimation - {os.path.basename(image_path)}')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Visualization saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")
    
    def batch_predict(self, input_dir, output_dir="predictions", confidence=0.5):
        """Run inference on all images in a directory."""
        if not self.model:
            print("❌ Model not loaded")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No images found in {input_dir}")
            return
        
        print(f"🔍 Processing {len(image_files)} images...")
        
        results = []
        for img_path in image_files:
            print(f"\n📸 Processing: {img_path.name}")
            
            # Generate output path
            output_path = os.path.join(output_dir, f"pred_{img_path.name}")
            
            # Run prediction
            result = self.predict_single_image(str(img_path), confidence, output_path)
            if result:
                results.append((str(img_path), result))
        
        print(f"\n✅ Batch processing complete! Results saved in {output_dir}")
        return results

def main():
    parser = argparse.ArgumentParser(description='Cow Pose Estimation Demo')
    parser.add_argument('--source', type=str, required=True,
                      help='Path to image file or directory')
    parser.add_argument('--model', type=str, default='models/weights/best.pt',
                      help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions',
                      help='Output directory for predictions')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = CowPoseEstimationDemo(args.model)
    
    if not demo.load_model():
        return
    
    # Check if source is file or directory
    if os.path.isfile(args.source):
        # Single image prediction
        output_path = os.path.join(args.output, f"pred_{os.path.basename(args.source)}")
        os.makedirs(args.output, exist_ok=True)
        demo.predict_single_image(args.source, args.conf, output_path)
        
    elif os.path.isdir(args.source):
        # Batch prediction
        demo.batch_predict(args.source, args.output, args.conf)
        
    else:
        print(f"❌ Source not found: {args.source}")
        print("💡 Please provide a valid image file or directory")

if __name__ == "__main__":
    print("🐄 Cow Pose Estimation Demo")
    print("=" * 40)
    
    # Check if running without arguments (show help)
    if len(sys.argv) == 1:
        print("📋 Usage Examples:")
        print("   Single image: python demo.py --source path/to/image.jpg")
        print("   Batch mode:   python demo.py --source path/to/images/")
        print("   Custom model: python demo.py --source image.jpg --model custom_model.pt")
        print("\n💡 For full options: python demo.py --help")
        sys.exit(0)
    
    main()
