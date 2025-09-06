# GSDiff Training

This guide provides complete instructions for training GSDiff models with CUDA GPU support.

## 🚀 GPU Specifications

### **Recommended Hardware:**
- **GPU**: NVIDIA A10G, V100, or similar (8GB+ VRAM)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD

### **Performance:**
- **Training Speed**: Fast GPU-accelerated training
- **Batch Size**: 128 (Stage 1), 64 (Stage 2)
- **Training Time**: 1-2 days total

## 🎯 Why GPU Training?

### **Advantages:**
- ✅ **Dedicated GPU Memory**: Large VRAM for large batch sizes
- ✅ **CUDA Optimization**: Professional-grade ML training
- ✅ **Fast Training**: 1-2 days vs weeks on CPU
- ✅ **Stable**: No memory constraints
- ✅ **Scalable**: Can handle large datasets

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Option A: Local GPU machine
# Ensure you have NVIDIA GPU with CUDA support

# Option B: Cloud instance (AWS, GCP, etc.)
# Launch GPU instance with CUDA support
```

### 2. Install CUDA and Dependencies
```bash
# Install CUDA (if not already installed)
# Follow NVIDIA CUDA installation guide for your OS

# Verify CUDA installation
nvidia-smi
```

### 3. Clone and Setup Project
```bash
# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Prepare Data
```bash
# Organize your RPLAN pickle files
python setup_training.py

# Verify data
python check_pickle_files.py
```

### 5. Start Training
```bash
# Use training launcher
python launch_training.py

# Select option 1 for Stage 1 training
# Select option 2 for Stage 2 training
```

## 📊 Training Parameters

### **Stage 1: Node Generation**
```python
batch_size = 128      # Optimized for GPU training
learning_rate = 1e-4
epochs = 100
workers = 4
pin_memory = True
persistent_workers = True
```

### **Stage 2: Edge Generation**
```python
batch_size = 64       # Optimized for GPU training
learning_rate = 1e-4
epochs = 100
workers = 4
pin_memory = True
persistent_workers = True
```

## ⚡ GPU Optimizations

### **Memory Management:**
```python
# Automatic GPU cache clearing
torch.cuda.empty_cache()

# Gradient clipping for stability
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Mixed precision training (optional)
from torch.cuda.amp import autocast, GradScaler
```

### **Data Loading:**
```python
# Optimized DataLoader settings
DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### **Training Loop:**
```python
# Professional training loop
for epoch in range(epochs):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Validate
    val_loss = validate_epoch(model, val_loader, criterion, device)
    
    # Save checkpoint
    if (epoch + 1) % save_interval == 0:
        save_checkpoint(model, optimizer, epoch, val_loss, save_dir)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
```

## 📈 Training Timeline

### **GPU Training:**
- **Stage 1**: 12-24 hours
- **Stage 2**: 6-12 hours
- **Total**: 1-2 days

### **Progress Monitoring:**
```bash
# Check training logs
tail -f GSDiff/scripts/outputs/structure-1/training.log

# Monitor GPU usage
nvidia-smi -l 1

# Check checkpoints
ls -la GSDiff/scripts/outputs/structure-1/
```

## 💰 Cost Analysis

### **GPU Training Options:**
| Option | Cost | Time | Total |
|--------|------|------|-------|
| **G5.2xlarge** | $1.20/hr | 1-2 days | $30-60 |
| **p3.2xlarge** | $3.06/hr | 2-3 days | $150-200 |
| **p4d.xlarge** | $3.06/hr | 1-2 days | $75-150 |
| **Local GPU** | Free | 1-2 days | $0 |

## 🛠️ Troubleshooting

### **Common Issues:**

#### **CUDA Out of Memory:**
```bash
# Reduce batch size
python launch_training_g5.py
# Use smaller batch sizes: 64, 32, 16

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### **Training Slow:**
```bash
# Check GPU utilization
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check data loading
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

#### **Connection Issues:**
```bash
# Check instance status
aws ec2 describe-instances --instance-ids i-your-instance-id

# Restart instance if needed
aws ec2 reboot-instances --instance-ids i-your-instance-id
```

## 🎯 Best Practices

### **Cost Optimization:**
1. **Use Spot Instances**: Save 50-70% on costs
2. **Stop When Not Training**: Don't leave instance running
3. **Use EBS Snapshots**: Save your work
4. **Monitor Usage**: Set up billing alerts

### **Training Optimization:**
1. **Use Mixed Precision**: Faster training, less memory
2. **Gradient Accumulation**: Simulate larger batch sizes
3. **Learning Rate Scheduling**: Better convergence
4. **Early Stopping**: Prevent overfitting

### **Data Management:**
1. **Use S3**: Store large datasets
2. **EBS Optimization**: Use GP3 for better performance
3. **Data Caching**: Cache frequently used data
4. **Backup Checkpoints**: Save to S3 regularly

## 🏆 Expected Results

With 83,186 training files on GPU:
- **High Quality Models**: Professional-grade results
- **Fast Training**: 1-2 days total
- **Stable Training**: CUDA-optimized stability
- **Good Convergence**: Proper loss curves
- **Cost Effective**: Reasonable cost for fast training

## 🎯 Recommendation

**Use GPU training!** It's:
- ✅ **Fast**: 10-15x faster than CPU
- ✅ **Professional**: Designed for ML workloads
- ✅ **Stable**: CUDA optimization
- ✅ **Reliable**: No memory constraints
- ✅ **Scalable**: Can handle large datasets

**Training time**: 1-2 days vs weeks on CPU, worth the investment!
