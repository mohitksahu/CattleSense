#!/usr/bin/env python3
"""
Final project validation script for Cow Pose Estimation Project
Validates that all components are properly configured and ready for use.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n📦 Checking dependencies...")
    try:
        import torch
        import torchvision
        import ultralytics
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from PIL import Image
        import yaml as yaml_lib
        import tqdm
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_project_structure():
    """Validate project directory structure."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        "data",
        "data/images",
        "data/images/train",
        "data/images/val", 
        "data/labels",
        "data/labels/train",
        "data/labels/val",
        "scripts",
        "models/weights",
        "docs",
        "runs",
        "predictions",
        "visualizations"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Project structure is valid")
    return True

def check_data_config():
    """Validate data.yaml configuration."""
    print("\n📋 Checking data configuration...")
    
    if not os.path.exists("data/data.yaml"):
        print("❌ data.yaml not found")
        return False
    
    try:
        with open("data/data.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing keys in data.yaml: {missing_keys}")
            return False
        
        print(f"✅ Data config valid - {config['nc']} classes")
        return True
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability for training."""
    print("\n🖥️  Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU available - will use CPU (slower training)")
            return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def check_scripts():
    """Validate training and utility scripts."""
    print("\n📜 Checking scripts...")
    
    required_scripts = [
        "scripts/train.py",
        "scripts/test_model.py", 
        "scripts/utils.py",
        "scripts/validate.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing scripts: {missing_scripts}")
        return False
    
    print("✅ All scripts present")
    return True

def count_dataset_samples():
    """Count training and validation samples."""
    print("\n🔢 Counting dataset samples...")
    
    train_images = len([f for f in os.listdir("data/images/train") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_images = len([f for f in os.listdir("data/images/val") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    train_labels = len([f for f in os.listdir("data/labels/train") if f.endswith('.txt')])
    val_labels = len([f for f in os.listdir("data/labels/val") if f.endswith('.txt')])
    
    print(f"📊 Training: {train_images} images, {train_labels} labels")
    print(f"📊 Validation: {val_images} images, {val_labels} labels")
    
    if train_images != train_labels:
        print("⚠️  Mismatch between training images and labels")
    if val_images != val_labels:
        print("⚠️  Mismatch between validation images and labels")
    
    return train_images > 0 and val_images > 0

def main():
    """Run all validation checks."""
    print("🐄 Cow Pose Estimation Project - Final Validation")
    print("=" * 60)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_project_structure,
        check_data_config,
        check_gpu_availability,
        check_scripts,
        count_dataset_samples
    ]
    
    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"❌ Check failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("🎉 ALL CHECKS PASSED!")
        print("\n✨ Your project is ready for:")
        print("   • Training: cd scripts && python train.py")
        print("   • Testing: python test_model.py --source image.jpg")
        print("   • Validation: python validate.py")
        print("\n📚 Documentation available in:")
        print("   • README.md - Main project documentation")
        print("   • docs/ - Detailed guides")
        
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("💡 Run setup.py again if needed.")
    
    print("\n🚀 Ready to revolutionize cow behavior analysis!")

if __name__ == "__main__":
    main()
