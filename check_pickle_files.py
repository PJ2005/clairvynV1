#!/usr/bin/env python3
"""
Pickle File Checker

This script checks for pickle files and creates sample data if needed.
"""

import os
import numpy as np
import pickle
from pathlib import Path

def print_status(message, status="INFO"):
    """Print formatted status messages."""
    status_icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄"
    }
    icon = status_icons.get(status, "ℹ️")
    print(f"{icon} {message}")

def check_pickle_directories():
    """Check for pickle directories and files."""
    print_status("Checking pickle directories...", "PROGRESS")
    
    pickle_train_path = "pickle/train"
    pickle_val_path = "pickle/val"
    
    # Check if directories exist
    train_exists = os.path.exists(pickle_train_path)
    val_exists = os.path.exists(pickle_val_path)
    
    print_status(f"pickle/train exists: {train_exists}", "SUCCESS" if train_exists else "WARNING")
    print_status(f"pickle/val exists: {val_exists}", "SUCCESS" if val_exists else "WARNING")
    
    # Count files in each directory
    train_files = []
    val_files = []
    
    if train_exists:
        train_files = [f for f in os.listdir(pickle_train_path) if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
        print_status(f"Found {len(train_files)} files in pickle/train", "SUCCESS" if train_files else "WARNING")
    
    if val_exists:
        val_files = [f for f in os.listdir(pickle_val_path) if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
        print_status(f"Found {len(val_files)} files in pickle/val", "SUCCESS" if val_files else "WARNING")
    
    return train_files, val_files

def create_sample_pickle_data():
    """Create sample pickle data for testing."""
    print_status("Creating sample pickle data...", "PROGRESS")
    
    # Create directories if they don't exist
    os.makedirs("pickle/train", exist_ok=True)
    os.makedirs("pickle/val", exist_ok=True)
    
    # Create sample RPLAN data structure
    sample_data = {
        'corner_list_np_normalized_padding_withsemantics': np.random.rand(53, 16).astype(np.float32),
        'padding_mask': np.ones(53, dtype=np.uint8),
        'edges': np.random.rand(2809).astype(np.float32)
    }
    
    # Save as both .pkl and .npy formats
    train_pkl_path = "pickle/train/sample_train.pkl"
    train_npy_path = "pickle/train/sample_train.npy"
    val_pkl_path = "pickle/val/sample_val.pkl"
    val_npy_path = "pickle/val/sample_val.npy"
    
    try:
        # Save as pickle
        with open(train_pkl_path, 'wb') as f:
            pickle.dump(sample_data, f)
        print_status(f"Created {train_pkl_path}", "SUCCESS")
        
        with open(val_pkl_path, 'wb') as f:
            pickle.dump(sample_data, f)
        print_status(f"Created {val_pkl_path}", "SUCCESS")
        
        # Save as numpy
        np.save(train_npy_path, sample_data)
        print_status(f"Created {train_npy_path}", "SUCCESS")
        
        np.save(val_npy_path, sample_data)
        print_status(f"Created {val_npy_path}", "SUCCESS")
        
        return True
        
    except Exception as e:
        print_status(f"Error creating sample data: {e}", "ERROR")
        return False

def test_file_loading():
    """Test loading the created files."""
    print_status("Testing file loading...", "PROGRESS")
    
    test_files = [
        "pickle/train/sample_train.pkl",
        "pickle/train/sample_train.npy",
        "pickle/val/sample_val.pkl",
        "pickle/val/sample_val.npy"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.npy'):
                    data = np.load(file_path, allow_pickle=True).item()
                else:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                
                print_status(f"✓ Successfully loaded {file_path}", "SUCCESS")
                print_status(f"  Data keys: {list(data.keys())}", "INFO")
                
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        print_status(f"  {key}: shape {value.shape}, dtype {value.dtype}", "INFO")
                
            except Exception as e:
                print_status(f"✗ Failed to load {file_path}: {e}", "ERROR")
        else:
            print_status(f"✗ File not found: {file_path}", "WARNING")

def main():
    """Main function."""
    print("="*60)
    print("🔍 PICKLE FILE CHECKER")
    print("="*60)
    print("This script checks for pickle files and creates sample data if needed.")
    print("="*60)
    
    # Step 1: Check existing files
    train_files, val_files = check_pickle_directories()
    
    total_files = len(train_files) + len(val_files)
    
    if total_files > 0:
        print_status(f"Found {total_files} pickle files total", "SUCCESS")
        print("\n📋 Available files:")
        
        if train_files:
            print("  Training files:")
            for file in train_files[:5]:  # Show first 5
                file_path = os.path.join("pickle/train", file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"    - {file} ({file_size:.1f} MB)")
            if len(train_files) > 5:
                print(f"    ... and {len(train_files) - 5} more")
        
        if val_files:
            print("  Validation files:")
            for file in val_files[:5]:  # Show first 5
                file_path = os.path.join("pickle/val", file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"    - {file} ({file_size:.1f} MB)")
            if len(val_files) > 5:
                print(f"    ... and {len(val_files) - 5} more")
        
        # Test loading a sample file
        if train_files:
            test_file = os.path.join("pickle/train", train_files[0])
            print_status(f"Testing file: {test_file}", "PROGRESS")
            try:
                if test_file.endswith('.npy'):
                    data = np.load(test_file, allow_pickle=True).item()
                else:
                    with open(test_file, 'rb') as f:
                        data = pickle.load(f)
                print_status("✓ File loads successfully", "SUCCESS")
                print_status(f"  Data keys: {list(data.keys())}", "INFO")
            except Exception as e:
                print_status(f"✗ Error loading file: {e}", "ERROR")
    
    else:
        print_status("No pickle files found", "WARNING")
        
        # Ask if user wants to create sample data
        create_sample = input("\nWould you like to create sample pickle data for testing? (y/N): ").strip().lower()
        
        if create_sample in ['y', 'yes']:
            if create_sample_pickle_data():
                print_status("Sample data created successfully", "SUCCESS")
                test_file_loading()
            else:
                print_status("Failed to create sample data", "ERROR")
        else:
            print_status("No sample data created", "INFO")
            print("\n📋 To use the pipeline:")
            print("1. Place your .pkl or .npy files in pickle/train and pickle/val directories")
            print("2. Run: python pipeline_rplan.py")
    
    print("\n" + "="*60)
    print_status("Pickle file check completed!", "SUCCESS")
    print("="*60)
    
    if total_files > 0:
        print("\n🚀 Ready to run pipeline:")
        print("python pipeline_rplan.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCheck cancelled by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
