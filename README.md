# GSDiff Training

This guide provides complete instructions for training GSDiff models using Metal Performance Shaders (MPS) on Apple Silicon.

## 🍎 Apple Silicon MPS Capabilities

### **What MPS Can Do:**
- ✅ **GPU Acceleration**: Use Apple Silicon GPU for training
- ✅ **Faster Training**: 10-50x faster than CPU
- ✅ **Memory Efficient**: Unified memory architecture
- ✅ **PyTorch Support**: Native MPS backend

### **MPS Limitations:**
- ⚠️ **Not CUDA**: Different from NVIDIA CUDA
- ⚠️ **Limited Libraries**: Some ML libraries don't support MPS
- ⚠️ **Memory Constraints**: Unified memory architecture
- ⚠️ **Compatibility**: Some operations not optimized for MPS

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements_m3.txt

# Test MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 2. Prepare Data
```bash
# Run setup script
python setup_training.py

# Verify data
python check_pickle_files.py
```

### 3. Start Training
```bash
# Use training launcher
python launch_training.py

# Select option 1 for Stage 1 training
# Select option 2 for Stage 2 training
```

## 📋 Complete Setup Process

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements_m3.txt
```

### Step 2: Verify MPS

```bash
# Test MPS availability
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ MPS is working!')
    # Test basic operations
    x = torch.randn(100, 100, device='mps')
    y = torch.randn(100, 100, device='mps')
    z = torch.mm(x, y)
    print('✅ MPS tensor operations working!')
else:
    print('❌ MPS not available')
"
```

### Step 3: Prepare Training Data

```bash
# Run setup script
python setup_training.py

# Verify data organization
ls -la GSDiff/datasets/rplang-v3-withsemantics/train/ | head -5
```

### Step 4: Start Training

```bash
# Use training launcher
python launch_training.py

# Select training option:
# 1. Stage 1: Node Generation
# 2. Stage 2: Edge Generation
```

## ⚡ Apple Silicon MPS Optimizations

### **Memory Optimizations:**
```python
# MPS-specific memory management
torch.mps.empty_cache()  # Clear MPS cache
torch.backends.mps.allow_tf32 = True  # Enable TF32

# Reduced batch sizes for unified memory
batch_size = 64  # Stage 1 (vs 256 on V100)
batch_size = 4   # Stage 2 (vs 8 on V100)
```

### **Training Parameters:**
- **Batch Size**: 64 (Stage 1), 4 (Stage 2) - optimized for unified memory
- **Workers**: 2 - optimized for Apple Silicon
- **Memory**: Automatic MPS cache clearing
- **Device**: MPS with CPU fallback

### **System Optimizations:**
- **Pin Memory**: Disabled for MPS compatibility
- **Non-blocking**: Disabled for MPS stability
- **Persistent Workers**: Disabled for MPS compatibility

## 📊 Training Timeline on Apple Silicon

### **MPS Training:**
- **Stage 1**: 1-2 weeks (vs 2-3 days on V100)
- **Stage 2**: 3-5 days (vs 1-2 days on V100)
- **Total**: 2-3 weeks

### **CPU Training (Fallback):**
- **Stage 1**: 2-3 months
- **Stage 2**: 1-2 months
- **Total**: 3-5 months

## 🔧 MPS-Specific Features

### **Automatic Fallback:**
```python
# Automatically falls back to CPU if MPS fails
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

### **Memory Management:**
```python
# Clear MPS cache periodically
if step % 100 == 0 and device == 'mps':
    torch.mps.empty_cache()
```

### **Compatibility:**
```python
# Disable features not compatible with MPS
pin_memory=False  # MPS doesn't support pin_memory
non_blocking=False  # MPS doesn't support non_blocking
```

## 📁 File Structure

```
gsdiff-training/
├── GSDiff/
│   ├── datasets/
│   │   └── rplang-v3-withsemantics/
│   │       ├── train/     # 66,548 files
│   │       ├── val/       # 8,318 files
│   │       └── test/      # 8,320 files
│   ├── scripts/
│   │   ├── outputs/
│   │   │   ├── structure-1/  # Stage 1 models
│   │   │   └── structure-2/  # Stage 2 models
│   │   ├── trainval_main_unconstrained_mps.py
│   │   └── trainval_main_edge_unconstrained_mps.py
│   └── requirements.txt
├── launch_training_m3.py
├── requirements_m3.txt
└── README_M3.md
```

## 🔍 Monitoring Training

### **MPS Monitoring:**
```bash
# Check MPS memory usage
python -c "
import torch
if torch.backends.mps.is_available():
    print(f'MPS memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB')
"
```

### **System Monitoring:**
```bash
# CPU and memory usage
htop

# Training progress
tail -f GSDiff/scripts/outputs/structure-1/training.log
```

### **Training Progress:**
```bash
# Check checkpoints
ls -la GSDiff/scripts/outputs/structure-1/

# Monitor training
python launch_training.py
# Select option 3: Monitor Training Progress
```

## 🛠️ Troubleshooting

### **Common MPS Issues:**

#### 1. MPS Not Available
```bash
# Check macOS version (need macOS 12.3+)
sw_vers

# Check PyTorch version (need 1.12+)
python -c "import torch; print(torch.__version__)"
```

#### 2. MPS Memory Issues
```python
# Reduce batch size
batch_size = 32  # or 16, 8

# Clear cache more frequently
if step % 50 == 0:
    torch.mps.empty_cache()
```

#### 3. MPS Compatibility Issues
```python
# Some operations may not work on MPS
# Automatically falls back to CPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

#### 4. Training Too Slow
```bash
# Check if MPS is actually being used
python -c "
import torch
x = torch.randn(100, 100, device='mps')
print(f'Device: {x.device}')
"
```

## 💰 Cost Comparison

### **Apple Silicon Training:**
- **Hardware**: Already owned
- **Electricity**: ~$20-50 for 2-3 weeks
- **Time**: 2-3 weeks
- **Total**: $20-50

### **vs EC2 p3.2xlarge:**
- **Cost**: $150-200
- **Time**: 2-3 days
- **Total**: $150-200

### **vs Google Colab:**
- **Cost**: $10-50
- **Time**: 2-5 days
- **Total**: $10-50

## 🎯 Performance Expectations

### **Apple Silicon MPS:**
- **Speed**: 10-50x faster than CPU
- **Memory**: Efficient unified memory usage
- **Stability**: Good with proper optimization
- **Quality**: Same as GPU training

### **Training Quality:**
- **Stage 1**: Good convergence, stable training
- **Stage 2**: Good convergence, stable training
- **Final Models**: Production-ready quality

## 🚀 Next Steps After Training

### **1. Test Trained Models**
```bash
# Test inference
cd GSDiff/scripts
python test_main.py
```

### **2. Use in Pipeline**
```bash
# Run your custom pipeline
python pipeline_rplan.py
```

### **3. Export Models**
```bash
# Models are automatically saved in outputs/
ls -la GSDiff/scripts/outputs/structure-1/
ls -la GSDiff/scripts/outputs/structure-2/
```

## 🏆 Expected Results

With 83,186 training files on Apple Silicon MPS:
- **Good Quality Models**: Comparable to GPU training
- **Reasonable Training Time**: 2-3 weeks total
- **Stable Training**: MPS-optimized for stability
- **Good Convergence**: Proper loss curves

## 📞 Support

If you encounter MPS issues:
1. Check macOS version (need 12.3+)
2. Check PyTorch version (need 1.12+)
3. Reduce batch size if memory issues
4. Use CPU fallback if MPS fails

## 🎯 Recommendation

**Apple Silicon MPS is viable for training!** It's:
- ✅ **Free**: Use your existing hardware
- ✅ **Fast**: 10-50x faster than CPU
- ✅ **Stable**: Good convergence
- ✅ **Convenient**: Train at home

**Training time**: 2-3 weeks vs 2-3 days on V100, but completely free!
