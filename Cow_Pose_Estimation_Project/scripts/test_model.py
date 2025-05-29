from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
import argparse
import time
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test YOLO model on images or video')
    parser.add_argument('--model', type=str, 
                        default='runs/cow_pose_detection/weights/best.pt', 
                        help='Path to PyTorch model')
    parser.add_argument('--source', type=str, default='0', required=True, 
                        help='Path to image, video, or 0 for webcam')
    parser.add_argument('--conf-threshold', type=float, default=0.27, 
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='predictions', 
                        help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model file exists
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        print("Searching for model in common locations...")
        
        # Try different possible paths
        possible_paths = [

        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {model_path}")
                break
        else:
            print("Could not find model file. Please provide the correct path.")
            return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process source
    source = args.source
    
    # Check if source is webcam
    if source.isnumeric():
        source = int(source)
        print(f"Opening webcam...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open webcam")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                
                # Run YOLOv8 inference on the frame
                start_time = time.time()
                results = model(frame, conf=args.conf_threshold)
                inference_time = time.time() - start_time
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                
                # Add FPS info - with safety check to avoid division by zero
                if inference_time > 0:
                    fps = 1 / inference_time
                else:
                    fps = 0
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                
                # Break the loop if 'ESC' is pressed
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("ESC pressed, exiting...")
                    break
                elif key == ord('q'):  # q key
                    print("Q pressed, exiting...")
                    break
        
        except Exception as e:
            print(f"Error during webcam processing: {e}")
        
        finally:
            # Release resources
            print("Releasing webcam and closing windows...")
            cap.release()
            cv2.destroyAllWindows()
    
    # Check if source is an image or video file
    elif isinstance(source, str) and os.path.isfile(source):
        _, ext = os.path.splitext(source)
        
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            # Process image
            print(f"Processing image: {source}")
            
            try:
                # Run YOLOv8 inference
                start_time = time.time()
                results = model(source, conf=args.conf_threshold)
                inference_time = time.time() - start_time
                print(f"Inference time: {inference_time*1000:.2f} ms")
                
                # Get the annotated image
                annotated_img = results[0].plot()
                
                # Save result
                output_path = os.path.join(args.output, os.path.basename(source))
                cv2.imwrite(output_path, annotated_img)
                print(f"Result saved to {output_path}")
                
                # Display result
                cv2.imshow("Result", annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            except Exception as e:
                print(f"Error processing image: {e}")
            
        elif ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Process video
            print(f"Processing video: {source}")
            
            try:
                # Open video
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    print(f"Error: Could not open video {source}")
                    return
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create output video writer
                output_path = os.path.join(args.output, os.path.basename(source))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Run YOLOv8 inference on video
                results = model(source, conf=args.conf_threshold, stream=True)
                
                frame_count = 0
                total_time = 0
                
                # Process each frame
                for result in results:
                    # Get the original frame
                    orig_frame = result.orig_img
                    
                    # Measure inference time
                    start_time = time.time()
                    annotated_frame = result.plot()
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    
                    # Add FPS info - with safety check
                    if inference_time > 0:
                        fps = 1 / inference_time
                    else:
                        fps = 0
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Write frame to output video
                    out.write(annotated_frame)
                    
                    # Display result
                    cv2.imshow("Result", annotated_frame)
                    if cv2.waitKey(1) == 27:  # ESC key
                        break
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Processed {frame_count} frames. Average inference time: {(total_time/frame_count)*1000:.2f} ms")
                
                # Release resources
                out.release()
                cv2.destroyAllWindows()
                print(f"Result saved to {output_path}")
            
            except Exception as e:
                print(f"Error processing video: {e}")
                if 'cap' in locals():
                    cap.release()
                if 'out' in locals():
                    out.release()
                cv2.destroyAllWindows()
        
        else:
            print(f"Unsupported file format: {ext}")
    
    else:
        print(f"Error: File {source} does not exist")

if __name__ == "__main__":
    main()