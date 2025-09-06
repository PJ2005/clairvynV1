#!/usr/bin/env python3
"""
GSDiff Setup Script

This script helps set up the GSDiff environment and process RPLAN data.
"""

import os
import sys
import subprocess
import shutil
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

def check_environment():
    """Check if the environment is properly set up."""
    print_status("Checking environment setup...", "PROGRESS")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment detected", "SUCCESS")
    else:
        print_status("Not in a virtual environment - consider using .venv/Scripts/activate", "WARNING")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print_status(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}", "SUCCESS")
    else:
        print_status(f"Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+", "ERROR")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print_status("Installing dependencies...", "PROGRESS")
    
    try:
        # Install from requirements.txt
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "GSDiff/requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print_status("Dependencies installed successfully", "SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies: {e}", "ERROR")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    print_status("Creating directories...", "PROGRESS")
    
    directories = [
        "GSDiff/datasets/rplandata/Data",
        "GSDiff/scripts/outputs",
        "GSDiff/scripts/test_outputs",
        "test_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_status(f"Created directory: {directory}", "SUCCESS")
    
    return True

def check_data_availability():
    """Check if RPLAN pickle data is available."""
    print_status("Checking data availability...", "PROGRESS")
    
    # Check for pickle files in the specified directories
    pickle_train_path = "pickle/train"
    pickle_val_path = "pickle/val"
    
    train_files = []
    val_files = []
    
    if os.path.exists(pickle_train_path):
        train_files = [f for f in os.listdir(pickle_train_path) if f.endswith('.pkl')]
        if train_files:
            print_status(f"Found {len(train_files)} training pickle files in {pickle_train_path}", "SUCCESS")
        else:
            print_status(f"No .pkl files found in {pickle_train_path}", "WARNING")
    else:
        print_status(f"Training pickle directory not found at {pickle_train_path}", "WARNING")
    
    if os.path.exists(pickle_val_path):
        val_files = [f for f in os.listdir(pickle_val_path) if f.endswith('.pkl')]
        if val_files:
            print_status(f"Found {len(val_files)} validation pickle files in {pickle_val_path}", "SUCCESS")
        else:
            print_status(f"No .pkl files found in {pickle_val_path}", "WARNING")
    else:
        print_status(f"Validation pickle directory not found at {pickle_val_path}", "WARNING")
    
    if train_files or val_files:
        return True
    else:
        print_status("No pickle files found. Please ensure pickle files are in pickle/train and pickle/val directories", "WARNING")
        return False

def process_pickle_data():
    """Process pickle data if available."""
    print_status("Processing pickle data...", "PROGRESS")
    
    pickle_train_path = "pickle/train"
    pickle_val_path = "pickle/val"
    
    # Check if pickle directories exist
    if not os.path.exists(pickle_train_path) and not os.path.exists(pickle_val_path):
        print_status("No pickle directories found", "WARNING")
        return False
    
    # Count pickle files
    train_count = 0
    val_count = 0
    
    if os.path.exists(pickle_train_path):
        train_count = len([f for f in os.listdir(pickle_train_path) if f.endswith('.pkl')])
        print_status(f"Found {train_count} training pickle files", "SUCCESS")
    
    if os.path.exists(pickle_val_path):
        val_count = len([f for f in os.listdir(pickle_val_path) if f.endswith('.pkl')])
        print_status(f"Found {val_count} validation pickle files", "SUCCESS")
    
    if train_count > 0 or val_count > 0:
        print_status("Pickle data is ready for use", "SUCCESS")
        return True
    else:
        print_status("No pickle files found to process", "WARNING")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print_status("Creating sample data...", "PROGRESS")
    
    import numpy as np
    
    # Create sample RPLAN data
    sample_data = {
        'corner_list_np_normalized_padding_withsemantics': np.random.rand(53, 16),
        'padding_mask': np.ones(53, dtype=np.uint8),
        'edges': np.random.rand(2809)
    }
    
    # Save sample data
    os.makedirs('test_data', exist_ok=True)
    np.save('test_data/sample_rplan.npy', sample_data)
    print_status("Sample RPLAN data created: test_data/sample_rplan.npy", "SUCCESS")
    
    return True

def test_imports():
    """Test if all modules can be imported."""
    print_status("Testing imports...", "PROGRESS")
    
    modules_to_test = [
        'torch',
        'numpy',
        'networkx',
        'cv2',
        'PIL',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print_status(f"✓ {module}", "SUCCESS")
        except ImportError as e:
            print_status(f"✗ {module}: {e}", "ERROR")
            failed_imports.append(module)
    
    if failed_imports:
        print_status(f"Failed to import: {', '.join(failed_imports)}", "ERROR")
        return False
    
    return True

def main():
    """Main setup function."""
    print("="*60)
    print("🏠 GSDiff Setup Script")
    print("="*60)
    
    # Step 1: Check environment
    if not check_environment():
        print_status("Environment check failed", "ERROR")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print_status("Dependency installation failed", "ERROR")
        return False
    
    # Step 3: Create directories
    if not create_directories():
        print_status("Directory creation failed", "ERROR")
        return False
    
    # Step 4: Test imports
    if not test_imports():
        print_status("Import test failed", "ERROR")
        return False
    
    # Step 5: Check data availability
    has_data = check_data_availability()
    
    # Step 6: Process data if available
    if has_data:
        if not process_pickle_data():
            print_status("Data processing failed", "ERROR")
            return False
    else:
        print_status("Skipping data processing - no pickle data found", "WARNING")
    
    # Step 7: Create sample data for testing
    create_sample_data()
    
    print("\n" + "="*60)
    print_status("GSDiff setup completed!", "SUCCESS")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Ensure your pickle files are in pickle/train and pickle/val directories")
    print("2. Download pre-trained weights from Google Drive links in GSDiff_COMMANDS.md")
    print("3. Place weights in GSDiff/scripts/outputs/ directory")
    print("4. Run test scripts:")
    print("   - python GSDiff/scripts/test_main.py")
    print("   - python pipeline_rplan.py")
    print("5. Or train your own models using training scripts")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Setup failed. Please check the error messages above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
