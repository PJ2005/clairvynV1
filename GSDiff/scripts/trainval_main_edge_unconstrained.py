#!/usr/bin/env python3
"""
GSDiff Stage 2 Training - G5.2xlarge Optimized

Optimized for AWS G5.2xlarge instance with NVIDIA A10G GPU:
- 24GB VRAM allows larger batch sizes
- CUDA optimizations for faster training
- Professional-grade training parameters
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

# Add GSDiff to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Setup logging for training."""
    log_dir = "outputs/structure-2"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and specs."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {gpu_name}")
        logger.info(f"✅ GPU Memory: {gpu_memory:.1f}GB")
        return True
    else:
        logger.error("❌ No CUDA GPU available")
        return False

def load_rplan_data(data_dir):
    """Load RPLAN pickle data."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logger.error(f"❌ Data directories not found: {train_dir}, {val_dir}")
        return None, None
    
    # Load training data
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.pkl')]
    val_files = [f for f in os.listdir(val_dir) if f.endswith('.pkl')]
    
    logger.info(f"📁 Training files: {len(train_files)}")
    logger.info(f"📁 Validation files: {len(val_files)}")
    
    return train_files, val_files

def create_data_loader(files, data_dir, batch_size=64, shuffle=True):
    """Create DataLoader for RPLAN data."""
    def collate_fn(batch):
        # Custom collate function for RPLAN data
        return batch
    
    dataset = RPLANDataset(files, data_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

class RPLANDataset:
    """RPLAN dataset wrapper."""
    def __init__(self, files, data_dir):
        self.files = files
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

def train_epoch(model, train_loader, optimizer, criterion, device, logger):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move batch to GPU
            if isinstance(batch, list):
                batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
            else:
                batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Clear GPU cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0

def validate_epoch(model, val_loader, criterion, device, logger):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to GPU
                if isinstance(batch, list):
                    batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
                else:
                    batch = batch.to(device)
                
                # Forward pass
                output = model(batch)
                loss = criterion(output, batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return total_loss / num_batches if num_batches > 0 else 0

def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='GSDiff Stage 2 Training - G5.2xlarge')
    parser.add_argument('--data_dir', type=str, default='datasets/rplang-v3-withsemantics',
                       help='Path to RPLAN dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (optimized for A10G 24GB)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting GSDiff Stage 2 Training - G5.2xlarge Optimized")
    logger.info(f"📊 Batch Size: {args.batch_size}")
    logger.info(f"📊 Learning Rate: {args.lr}")
    logger.info(f"📊 Epochs: {args.epochs}")
    
    # Check GPU
    if not check_gpu():
        return
    
    device = torch.device('cuda:0')
    logger.info(f"🎯 Using device: {device}")
    
    # Load data
    train_files, val_files = load_rplan_data(args.data_dir)
    if not train_files or not val_files:
        return
    
    # Create data loaders
    train_loader = create_data_loader(train_files, os.path.join(args.data_dir, "train"), 
                                    batch_size=args.batch_size, shuffle=True)
    val_loader = create_data_loader(val_files, os.path.join(args.data_dir, "val"), 
                                  batch_size=args.batch_size, shuffle=False)
    
    # Initialize model (placeholder - replace with actual GSDiff model)
    model = nn.Linear(100, 100).to(device)  # Replace with actual model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    save_dir = "outputs/structure-2"
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("🎯 Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        logger.info(f"📈 Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, logger)
        logger.info(f"📉 Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch, val_loss, save_dir)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            logger.info(f"🏆 New best model saved: {best_path}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    logger.info("✅ Training completed!")
    logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
