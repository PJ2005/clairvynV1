#!/usr/bin/env python3
"""
GSDiff Training Launcher

This script launches optimized GSDiff training with CUDA support.
"""

import os
import sys
import subprocess
import torch
import psutil
import time
from datetime import datetime

def check_gpu():
    """Check GPU availability and specs."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
        return True
    else:
        print("❌ No CUDA GPU available")
        return False

def check_system_resources():
    """Check system resources."""
    print("\n🖥️ System Resources:")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")
    print(f"Disk: {psutil.disk_usage('/').free / 1e9:.1f}GB free")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

def check_data():
    """Check if training data is available."""
    data_dirs = [
        "GSDiff/datasets/rplang-v3-withsemantics/train",
        "GSDiff/datasets/rplang-v3-withsemantics/val"
    ]
    
    total_files = 0
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            total_files += len(files)
            print(f"✅ {data_dir}: {len(files)} files")
        else:
            print(f"❌ {data_dir}: Not found")
    
    print(f"✅ Total training files: {total_files}")
    return total_files > 0

def run_training():
    """Run GSDiff training."""
    print("\n🚀 GSDiff Training Launcher")
    print("="*60)
    print("Optimized for CUDA GPU training")
    print("="*60)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check data
    if not check_data():
        print("❌ Data check failed. Please ensure training data is available.")
        return
    
    print("\n📋 Training Options:")
    print("1. Stage 1: Node Generation")
    print("2. Stage 2: Edge Generation")
    print("3. Monitor Training Progress")
    print("4. Check System Resources")
    print("5. Test GPU Performance")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    scripts = {
        "1": "trainval_main_unconstrained.py",
        "2": "trainval_main_edge_unconstrained.py"
    }
    
    if choice in scripts:
        script = scripts[choice]
        print(f"\n🔄 Starting {script}...")
        print("="*60)
        
        # Change to GSDiff scripts directory
        os.chdir("GSDiff/scripts")
        
        try:
            # Run training script
            process = subprocess.Popen([
                sys.executable, script,
                "--batch_size", "128" if choice == "1" else "64",
                "--epochs", "100",
                "--lr", "1e-4"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Stream output
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print("\n✅ Training completed successfully!")
            else:
                print(f"\n❌ Training failed with return code {process.returncode}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            if 'process' in locals():
                process.terminate()
    
    elif choice == "3":
        monitor_training()
    
    elif choice == "4":
        check_system_resources()
    
    elif choice == "5":
        test_gpu_performance()
    
    else:
        print("❌ Invalid choice")

def monitor_training():
    """Monitor training progress."""
    print("\n📊 Training Progress Monitor")
    print("="*40)
    
    # Check for training logs
    log_dirs = [
        "GSDiff/scripts/outputs/structure-1",
        "GSDiff/scripts/outputs/structure-2"
    ]
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            log_file = os.path.join(log_dir, "training.log")
            if os.path.exists(log_file):
                print(f"\n📁 {log_dir}:")
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print("Last 5 lines:")
                            for line in lines[-5:]:
                                print(f"  {line.strip()}")
                        else:
                            print("  No log entries yet")
                except Exception as e:
                    print(f"  Error reading log: {e}")
            else:
                print(f"  No training.log found")
        else:
            print(f"  Directory not found: {log_dir}")
    
    # Check for checkpoints
    print("\n💾 Checkpoints:")
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            checkpoints = [f for f in os.listdir(log_dir) if f.startswith('checkpoint_')]
            if checkpoints:
                print(f"  {log_dir}: {len(checkpoints)} checkpoints")
                for cp in sorted(checkpoints)[-3:]:  # Show last 3
                    print(f"    - {cp}")
            else:
                print(f"  {log_dir}: No checkpoints yet")

def test_gpu_performance():
    """Test GPU performance."""
    print("\n⚡ GPU Performance Test")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda:0')
    
    # Test matrix multiplication
    print("🧮 Testing matrix multiplication...")
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        try:
            # Create random matrices
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Time the operation
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # Wait for GPU to finish
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"  {size}x{size}: {duration:.3f}s")
            
        except Exception as e:
            print(f"  {size}x{size}: Error - {e}")
    
    # Test memory allocation
    print("\n💾 Testing memory allocation...")
    try:
        # Allocate large tensor
        large_tensor = torch.randn(1000, 1000, 1000, device=device)
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"  Allocated: {memory_used:.2f}GB")
        
        # Clear memory
        del large_tensor
        torch.cuda.empty_cache()
        memory_after = torch.cuda.memory_allocated(0) / 1e9
        print(f"  After cleanup: {memory_after:.2f}GB")
        
    except Exception as e:
        print(f"  Memory test error: {e}")

def main():
    """Main function."""
    print("🎯 GSDiff Training Launcher")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("GSDiff"):
        print("❌ GSDiff directory not found. Please run from project root.")
        return
    
    try:
        run_training()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
