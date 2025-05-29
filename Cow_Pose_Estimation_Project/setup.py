#!/usr/bin/env python3
"""
Setup script for Cow Pose Estimation Project
This script helps users set up the project environment and validate the installation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = [
        "models/weights",
        "runs",
        "predictions",
        "visualizations"
    ]
    
    print("📁 Creating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")

def validate_dataset():
    """Validate dataset structure."""
    print("🔍 Validating dataset structure...")
    
    required_paths = [
        "data/data.yaml",
        "data/images/train",
        "data/images/val",
        "data/labels/train",
        "data/labels/val"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("⚠️  Some dataset paths are missing:")
        for path in missing_paths:
            print(f"    - {path}")
        print("📋 Please ensure your dataset is properly structured.")
        return False
    else:
        print("✅ Dataset structure is valid!")
        return True

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower).")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot check GPU status.")
        return False

def run_quick_test():
    """Run a quick test to ensure everything is working."""
    print("🧪 Running quick test...")
    try:
        # Test YOLO import
        from ultralytics import YOLO
        print("  ✅ Ultralytics YOLO import successful")
        
        # Test data loading
        if os.path.exists("data/data.yaml"):
            import yaml
            with open("data/data.yaml", 'r') as f:
                config = yaml.safe_load(f)
            print(f"  ✅ Dataset config loaded: {config['nc']} classes")
        
        print("✅ Quick test passed!")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. 📚 Review the README.md for detailed instructions")
    print("2. 🏋️  Start training:")
    print("   cd scripts")
    print("   python train.py --epochs 200 --batch-size 16")
    print("3. 📊 Monitor training progress in the runs/ directory")
    print("4. 🧪 Test your trained model:")
    print("   python test_model.py --source path/to/image.jpg")
    print("\n💡 Tip: Use --help flag with any script for more options")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Setup Cow Pose Estimation Project')
    parser.add_argument('--skip-install', action='store_true', 
                        help='Skip requirements installation')
    parser.add_argument('--skip-validation', action='store_true', 
                        help='Skip dataset validation')
    parser.add_argument('--skip-test', action='store_true', 
                        help='Skip quick test')
    
    args = parser.parse_args()
    
    print("🐄 Cow Pose Estimation Project Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            print("⚠️  Installation failed. You may need to install requirements manually.")
    
    # Validate dataset
    if not args.skip_validation:
        validate_dataset()
    
    # Check GPU
    check_gpu()
    
    # Run quick test
    if not args.skip_test:
        run_quick_test()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
