#!/usr/bin/env python3
"""
Pickle File Organizer

This script helps organize RPLAN pickle files into the correct directory structure.
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

def create_directories():
    """Create pickle directories if they don't exist."""
    print_status("Creating pickle directories...", "PROGRESS")
    
    directories = ["pickle/train", "pickle/val"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_status(f"Created directory: {directory}", "SUCCESS")
    
    return True

def find_pickle_files(source_dir="."):
    """Find all pickle files in the source directory."""
    print_status(f"Searching for pickle files in {source_dir}...", "PROGRESS")
    
    # Look for .pkl files
    pickle_patterns = [
        "**/*.pkl",
        "**/*.pickle",
        "**/*.p",
        "**/*.npy"  # Also look for numpy files
    ]
    
    found_files = []
    for pattern in pickle_patterns:
        files = glob.glob(os.path.join(source_dir, pattern), recursive=True)
        found_files.extend(files)
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if found_files:
        print_status(f"Found {len(found_files)} pickle files:", "SUCCESS")
        for i, file_path in enumerate(found_files, 1):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  {i}. {file_path} ({file_size:.1f} MB)")
    else:
        print_status("No pickle files found", "WARNING")
    
    return found_files

def organize_files(found_files, split_ratio=0.8):
    """Organize files into train/val directories."""
    print_status("Organizing files into train/val directories...", "PROGRESS")
    
    if not found_files:
        print_status("No files to organize", "WARNING")
        return False
    
    # Calculate split
    total_files = len(found_files)
    train_count = int(total_files * split_ratio)
    val_count = total_files - train_count
    
    print_status(f"Will split into {train_count} training files and {val_count} validation files", "INFO")
    
    # Organize files
    train_files = found_files[:train_count]
    val_files = found_files[train_count:]
    
    # Copy files to appropriate directories
    copied_count = 0
    
    for file_path in train_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join("pickle/train", filename)
        try:
            shutil.copy2(file_path, dest_path)
            copied_count += 1
        except Exception as e:
            print_status(f"Error copying {file_path}: {e}", "ERROR")
    
    for file_path in val_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join("pickle/val", filename)
        try:
            shutil.copy2(file_path, dest_path)
            copied_count += 1
        except Exception as e:
            print_status(f"Error copying {file_path}: {e}", "ERROR")
    
    print_status(f"Successfully organized {copied_count} files", "SUCCESS")
    return True

def verify_organization():
    """Verify the organization was successful."""
    print_status("Verifying organization...", "PROGRESS")
    
    train_dir = "pickle/train"
    val_dir = "pickle/val"
    
    train_files = []
    val_files = []
    
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
    
    if os.path.exists(val_dir):
        val_files = [f for f in os.listdir(val_dir) if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
    
    print_status(f"Training files: {len(train_files)}", "SUCCESS")
    print_status(f"Validation files: {len(val_files)}", "SUCCESS")
    
    if train_files or val_files:
        print_status("Organization completed successfully!", "SUCCESS")
        return True
    else:
        print_status("No files found in organized directories", "ERROR")
        return False

def main():
    """Main function."""
    print("="*60)
    print("🗂️  PICKLE FILE ORGANIZER")
    print("="*60)
    print("This script organizes RPLAN pickle files into train/val directories.")
    print("="*60)
    
    # Step 1: Create directories
    if not create_directories():
        print_status("Failed to create directories", "ERROR")
        return False
    
    # Step 2: Find pickle files
    source_dir = input("\nEnter source directory (or press Enter for current directory): ").strip()
    if not source_dir:
        source_dir = "."
    
    found_files = find_pickle_files(source_dir)
    
    if not found_files:
        print_status("No pickle files found to organize", "WARNING")
        return False
    
    # Step 3: Ask for confirmation
    print(f"\nFound {len(found_files)} files to organize.")
    confirm = input("Do you want to organize these files? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        print_status("Organization cancelled", "INFO")
        return False
    
    # Step 4: Organize files
    if not organize_files(found_files):
        print_status("Failed to organize files", "ERROR")
        return False
    
    # Step 5: Verify organization
    if not verify_organization():
        print_status("Organization verification failed", "ERROR")
        return False
    
    print("\n" + "="*60)
    print_status("Pickle file organization completed!", "SUCCESS")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Verify files are in pickle/train and pickle/val directories")
    print("2. Run: python setup_gsdiff.py")
    print("3. Run: python pipeline_rplan.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Organization completed successfully!")
        else:
            print("\n💥 Organization failed. Please check the error messages above.")
    except KeyboardInterrupt:
        print("\n\nOrganization cancelled by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
