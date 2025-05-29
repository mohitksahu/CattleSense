from ultralytics import YOLO # type: ignore
import os
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for cow pose estimation')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    
    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Create directories if they don't exist
    os.makedirs('models/weights', exist_ok=True)
    os.makedirs('runs/detect', exist_ok=True)
    
    # Load a pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')
      # Train the model
    results = model.train(
        data='data/data.yaml',  # Path to dataset configuration
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        project='runs',
        name='cow_pose_detection',
        save=True,
        patience=20,  # Early stopping patience
        pretrained=True,
        optimizer='Adam'
    )
    
    # Save the final model
    model.export(format='onnx')  # Export to ONNX format
    print(f"Training completed. Model saved to {os.path.join('runs/cow_pose_detection', 'weights/best.pt')}")

if __name__ == '__main__':
    main()