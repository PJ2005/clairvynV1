#!/usr/bin/env python3
"""
GSDiff Training Launcher

This script launches optimized GSDiff training using MPS (Metal Performance Shaders) on Apple Silicon.
"""

import os
import sys
import subprocess
import torch
import time

def check_mps():
    """Check MPS availability and specs."""
    print("🔍 Checking MPS Configuration...")
    print("="*50)
    
    if torch.backends.mps.is_available():
        print("✅ MPS Available: True")
        print("✅ Metal Performance Shaders: Enabled")
        print("✅ GPU Acceleration: Available")
        
        # Check M3 Air specs
        try:
            import platform
            system_info = platform.platform()
            print(f"✅ System: {system_info}")
            
            # Check memory
            import psutil
            memory_gb = psutil.virtual_memory().total / 1e9
            print(f"✅ Total Memory: {memory_gb:.1f} GB")
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            print(f"✅ CPU Cores: {cpu_cores}")
            
        except ImportError:
            print("⚠️ Install psutil for detailed system info: pip install psutil")
        
        return True
    else:
        print("❌ MPS not available - Cannot run GPU training")
        print("⚠️ Falling back to CPU training (very slow)")
        return False

def check_data():
    """Check if training data is available."""
    print("\n📁 Checking Training Data...")
    print("="*50)
    
    data_dirs = [
        "GSDiff/datasets/rplang-v3-withsemantics/train",
        "GSDiff/datasets/rplang-v3-withsemantics/val",
        "GSDiff/datasets/rplang-v3-withsemantics/test"
    ]
    
    total_files = 0
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
            print(f"✅ {data_dir}: {len(files)} files")
            total_files += len(files)
        else:
            print(f"❌ Missing: {data_dir}")
            return False
    
    print(f"✅ Total training files: {total_files}")
    return total_files > 0

def run_training():
    """Run GSDiff training."""
    print("\n🚀 GSDiff Training Launcher")
    print("="*60)
    print("Optimized for Apple Silicon MPS (Metal Performance Shaders)")
    print("="*60)
    
    # Check MPS
    mps_available = check_mps()
    
    # Check data
    if not check_data():
        print("❌ Data check failed. Please ensure training data is available.")
        return
    
    print("\n📋 Training Options:")
    print("1. Stage 1: Node Generation")
    print("2. Stage 2: Edge Generation")
    print("3. Monitor Training Progress")
    print("4. Check System Resources")
    print("5. Test MPS Performance")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    scripts = {
        "1": "trainval_main_unconstrained.py",
        "2": "trainval_main_edge_unconstrained.py"
    }
    
    if choice in scripts:
        script = scripts[choice]
        print(f"\n🔄 Starting {script}...")
        print("="*60)
        
        # Change to scripts directory
        os.chdir("GSDiff/scripts")
        
        # Set environment variables for MPS optimization
        env = os.environ.copy()
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback
        
        try:
            # Start training with MPS environment
            process = subprocess.Popen(
                [sys.executable, script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✅ {script} completed successfully!")
            else:
                print(f"\n❌ {script} failed with return code {process.returncode}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            if 'process' in locals():
                process.terminate()
    
    elif choice == "3":
        monitor_training()
    
    elif choice == "4":
        check_system_resources()
    
    elif choice == "5":
        test_mps_performance()
    
    else:
        print("❌ Invalid choice")

def monitor_training():
    """Monitor training progress."""
    print("\n📊 Training Progress Monitor")
    print("="*50)
    
    # Check for checkpoint files
    checkpoint_dirs = [
        "GSDiff/scripts/outputs/structure-1",
        "GSDiff/scripts/outputs/structure-2"
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
            if files:
                print(f"✅ {checkpoint_dir}: {len(files)} checkpoints")
                # Show latest checkpoint
                latest = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                print(f"   Latest: {latest}")
            else:
                print(f"⚠️  {checkpoint_dir}: No checkpoints found")
        else:
            print(f"❌ {checkpoint_dir}: Directory not found")

def check_system_resources():
    """Check system resources."""
    print("\n💻 System Resources")
    print("="*50)
    
    # Check MPS
    if torch.backends.mps.is_available():
        print("✅ MPS: Available")
        print("✅ GPU: M3 Air GPU (Metal Performance Shaders)")
    else:
        print("❌ MPS: Not available")
    
    # Check CPU and Memory
    try:
        import psutil
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        print(f"RAM Used: {psutil.virtual_memory().used / 1e9:.1f} GB")
        print(f"RAM Available: {psutil.virtual_memory().available / 1e9:.1f} GB")
    except ImportError:
        print("Install psutil for detailed system info: pip install psutil")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        print(f"Disk Space: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
    except:
        print("Could not get disk space info")

def test_mps_performance():
    """Test MPS performance."""
    print("\n🧪 MPS Performance Test")
    print("="*50)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available - cannot run performance test")
        return
    
    print("Testing MPS performance...")
    
    # Test 1: Basic tensor operations
    print("\n1. Basic Tensor Operations:")
    start_time = time.time()
    
    # Create tensors on MPS
    x = torch.randn(1000, 1000, device='mps')
    y = torch.randn(1000, 1000, device='mps')
    
    # Matrix multiplication
    z = torch.mm(x, y)
    
    mps_time = time.time() - start_time
    print(f"   MPS Time: {mps_time:.4f} seconds")
    
    # Test 2: Neural network operations
    print("\n2. Neural Network Operations:")
    start_time = time.time()
    
    # Create a simple model
    model = torch.nn.Linear(1000, 1000).to('mps')
    input_tensor = torch.randn(100, 1000, device='mps')
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    nn_time = time.time() - start_time
    print(f"   Neural Network Time: {nn_time:.4f} seconds")
    
    # Test 3: Memory usage
    print("\n3. Memory Usage:")
    if hasattr(torch.mps, 'current_allocated_memory'):
        allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"   MPS Memory Allocated: {allocated:.2f} GB")
    
    print("\n✅ MPS Performance Test Completed!")
    print("If times are reasonable (< 1 second), MPS is working well.")

def main():
    """Main function."""
    try:
        run_training()
    except KeyboardInterrupt:
        print("\n\n👋 Training launcher cancelled by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
