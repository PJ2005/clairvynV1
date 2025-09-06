#!/usr/bin/env python3
"""
GSDiff Training Setup Script

This script prepares your environment and data for training GSDiff models.
"""

import os
import shutil
import glob
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

def create_training_directories():
    """Create required training directories."""
    print_status("Creating training directories...", "PROGRESS")
    
    directories = [
        "GSDiff/datasets/rplang-v3-withsemantics/train",
        "GSDiff/datasets/rplang-v3-withsemantics/val", 
        "GSDiff/datasets/rplang-v3-withsemantics/test",
        "GSDiff/scripts/outputs/structure-1",
        "GSDiff/scripts/outputs/structure-2"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_status(f"Created directory: {directory}", "SUCCESS")
    
    return True

def organize_training_data():
    """Organize pickle files for training."""
    print_status("Organizing training data...", "PROGRESS")
    
    # Check for source pickle files
    pickle_train_path = "pickle/train"
    pickle_val_path = "pickle/val"
    
    if not os.path.exists(pickle_train_path) and not os.path.exists(pickle_val_path):
        print_status("No pickle directories found", "ERROR")
        print_status("Please ensure your pickle files are in pickle/train and pickle/val", "WARNING")
        return False
    
    # Collect all pickle files
    all_files = []
    if os.path.exists(pickle_train_path):
        train_files = [os.path.join(pickle_train_path, f) for f in os.listdir(pickle_train_path) 
                      if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
        all_files.extend(train_files)
    
    if os.path.exists(pickle_val_path):
        val_files = [os.path.join(pickle_val_path, f) for f in os.listdir(pickle_val_path) 
                    if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
        all_files.extend(val_files)
    
    if not all_files:
        print_status("No pickle files found", "ERROR")
        return False
    
    print_status(f"Found {len(all_files)} pickle files", "SUCCESS")
    
    # Split files: 80% train, 10% val, 10% test
    train_count = int(len(all_files) * 0.8)
    val_count = int(len(all_files) * 0.1)
    test_count = len(all_files) - train_count - val_count
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]
    
    print_status(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test", "INFO")
    
    # Copy files to training directories
    copied_count = 0
    
    # Copy training files
    for i, file_path in enumerate(train_files):
        filename = f"{i}.npy"  # Rename to sequential numbers
        dest_path = os.path.join("GSDiff/datasets/rplang-v3-withsemantics/train", filename)
        try:
            shutil.copy2(file_path, dest_path)
            copied_count += 1
        except Exception as e:
            print_status(f"Error copying {file_path}: {e}", "ERROR")
    
    # Copy validation files
    for i, file_path in enumerate(val_files):
        filename = f"{i}.npy"
        dest_path = os.path.join("GSDiff/datasets/rplang-v3-withsemantics/val", filename)
        try:
            shutil.copy2(file_path, dest_path)
            copied_count += 1
        except Exception as e:
            print_status(f"Error copying {file_path}: {e}", "ERROR")
    
    # Copy test files
    for i, file_path in enumerate(test_files):
        filename = f"{i}.npy"
        dest_path = os.path.join("GSDiff/datasets/rplang-v3-withsemantics/test", filename)
        try:
            shutil.copy2(file_path, dest_path)
            copied_count += 1
        except Exception as e:
            print_status(f"Error copying {file_path}: {e}", "ERROR")
    
    print_status(f"Successfully organized {copied_count} files", "SUCCESS")
    return True

def update_training_scripts():
    """Update training script paths."""
    print_status("Updating training script paths...", "PROGRESS")
    
    # Get current working directory
    current_dir = os.getcwd()
    gsdiff_path = os.path.join(current_dir, "GSDiff")
    
    # Scripts to update
    scripts = [
        "GSDiff/scripts/trainval_main_unconstrained.py",
        "GSDiff/scripts/trainval_main_edge_unconstrained.py",
        "GSDiff/scripts/trainval_main_topo.py",
        "GSDiff/scripts/trainval_main_edge_topo.py",
        "GSDiff/scripts/trainval_main_boun.py",
        "GSDiff/scripts/trainval_main_edge_boun.py"
    ]
    
    # Path replacements
    old_paths = [
        "/home/user00/HSZ/gsdiff-main",
        "/home/user00/HSZ/gsdiff-main/datasets",
        "/home/user00/HSZ/gsdiff-main/gsdiff",
        "/home/user00/HSZ/gsdiff-main/scripts/metrics"
    ]
    
    new_paths = [
        gsdiff_path,
        os.path.join(gsdiff_path, "datasets"),
        os.path.join(gsdiff_path, "gsdiff"),
        os.path.join(gsdiff_path, "scripts", "metrics")
    ]
    
    updated_count = 0
    
    for script_path in scripts:
        if os.path.exists(script_path):
            try:
                # Read the script
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace paths
                for old_path, new_path in zip(old_paths, new_paths):
                    content = content.replace(old_path, new_path)
                
                # Write back
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print_status(f"Updated {script_path}", "SUCCESS")
                updated_count += 1
                
            except Exception as e:
                print_status(f"Error updating {script_path}: {e}", "ERROR")
        else:
            print_status(f"Script not found: {script_path}", "WARNING")
    
    print_status(f"Updated {updated_count} training scripts", "SUCCESS")
    return True

def verify_training_setup():
    """Verify training setup is complete."""
    print_status("Verifying training setup...", "PROGRESS")
    
    # Check directories
    required_dirs = [
        "GSDiff/datasets/rplang-v3-withsemantics/train",
        "GSDiff/datasets/rplang-v3-withsemantics/val",
        "GSDiff/datasets/rplang-v3-withsemantics/test"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) if f.endswith('.npy')])
            print_status(f"✓ {directory}: {file_count} files", "SUCCESS")
        else:
            print_status(f"✗ Missing directory: {directory}", "ERROR")
            return False
    
    # Check training scripts
    training_scripts = [
        "GSDiff/scripts/trainval_main_unconstrained.py",
        "GSDiff/scripts/trainval_main_edge_unconstrained.py"
    ]
    
    for script in training_scripts:
        if os.path.exists(script):
            print_status(f"✓ Training script: {script}", "SUCCESS")
        else:
            print_status(f"✗ Missing script: {script}", "ERROR")
            return False
    
    return True

def create_training_launcher():
    """Create a training launcher script."""
    print_status("Creating training launcher...", "PROGRESS")
    
    launcher_content = '''#!/usr/bin/env python3
"""
GSDiff Training Launcher

This script launches GSDiff training with proper environment setup.
"""

import os
import sys
import subprocess

def run_training():
    """Run GSDiff training."""
    print("🚀 Starting GSDiff Training")
    print("="*50)
    
    # Change to scripts directory
    os.chdir("GSDiff/scripts")
    
    print("\\n📋 Training Options:")
    print("1. Stage 1: Node Generation (Unconstrained)")
    print("2. Stage 2: Edge Generation (Unconstrained)")
    print("3. Stage 1: Node Generation (Topology Constrained)")
    print("4. Stage 2: Edge Generation (Topology Constrained)")
    print("5. Stage 1: Node Generation (Boundary Constrained)")
    print("6. Stage 2: Edge Generation (Boundary Constrained)")
    
    choice = input("\\nSelect training option (1-6): ").strip()
    
    scripts = {
        "1": "trainval_main_unconstrained.py",
        "2": "trainval_main_edge_unconstrained.py", 
        "3": "trainval_main_topo.py",
        "4": "trainval_main_edge_topo.py",
        "5": "trainval_main_boun.py",
        "6": "trainval_main_edge_boun.py"
    }
    
    if choice in scripts:
        script = scripts[choice]
        print(f"\\n🔄 Starting {script}...")
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except KeyboardInterrupt:
            print("\\n⏹️ Training interrupted by user")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    run_training()
'''
    
    with open("launch_training.py", "w", encoding='utf-8') as f:
        f.write(launcher_content)
    
    print_status("Created launch_training.py", "SUCCESS")
    return True

def main():
    """Main setup function."""
    print("="*60)
    print("🏗️  GSDIFF TRAINING SETUP")
    print("="*60)
    print("This script prepares your environment for training GSDiff models.")
    print("="*60)
    
    # Step 1: Create directories
    if not create_training_directories():
        print_status("Failed to create directories", "ERROR")
        return False
    
    # Step 2: Organize data
    if not organize_training_data():
        print_status("Failed to organize training data", "ERROR")
        return False
    
    # Step 3: Update scripts
    if not update_training_scripts():
        print_status("Failed to update training scripts", "ERROR")
        return False
    
    # Step 4: Verify setup
    if not verify_training_setup():
        print_status("Training setup verification failed", "ERROR")
        return False
    
    # Step 5: Create launcher
    create_training_launcher()
    
    print("\\n" + "="*60)
    print_status("Training setup completed!", "SUCCESS")
    print("="*60)
    
    print("\\n📋 Next Steps:")
    print("1. Verify your data is organized correctly")
    print("2. Adjust training parameters in scripts if needed")
    print("3. Start training:")
    print("   - python launch_training.py")
    print("   - OR: cd GSDiff/scripts && python trainval_main_unconstrained.py")
    print("\\n⚠️  Training will take several days - ensure you have sufficient GPU memory!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\\n🎉 Training setup completed successfully!")
        else:
            print("\\n💥 Training setup failed. Please check the error messages above.")
    except KeyboardInterrupt:
        print("\\n\\nSetup cancelled by user.")
    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")
