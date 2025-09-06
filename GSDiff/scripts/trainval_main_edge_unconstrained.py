import sys
import os

# Add current directory to path for M3 Air compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
gsdiff_root = os.path.dirname(current_dir)
sys.path.append(gsdiff_root)
sys.path.append(os.path.join(gsdiff_root, 'datasets'))
sys.path.append(os.path.join(gsdiff_root, 'gsdiff'))

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from gsdiff.heterhouse_56_13 import *
from gsdiff.utils import *
from itertools import cycle

'''MPS-optimized script to train the second-stage edge prediction model on M3 Air'''

# ===== OPTIMIZED PARAMETERS FOR M3 AIR MPS =====
lr = 1e-4
weight_decay = 1e-5
total_steps = float("inf")  # 200000
batch_size = 4  # Reduced for M3 Air memory constraints
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ===== MPS OPTIMIZATIONS =====
if device == 'mps':
    print("✅ MPS (Metal Performance Shaders) available - using GPU acceleration")
    # MPS-specific optimizations
    torch.mps.empty_cache()  # Clear MPS cache
else:
    print("⚠️ MPS not available - falling back to CPU")
    device = 'cpu'

# ===== MEMORY OPTIMIZATIONS FOR M3 AIR =====
# M3 Air has unified memory, so we need to be more careful
torch.backends.mps.allow_tf32 = True  # Enable TF32 if available

'''create output_dir'''
output_dir = 'outputs/structure-2/'
os.makedirs(output_dir, exist_ok=True)

'''Neural Network'''
model = EdgeModel().to(device)
print('total params:', sum(p.numel() for p in model.parameters()))

'''Data'''
dataset_train = RPlanGEdgeSemanSimplified('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=False)  # Disable pin_memory for MPS
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = RPlanGEdgeSemanSimplified('val')
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2,
                            drop_last=False, pin_memory=False)  # Disable pin_memory for MPS
dataloader_val_iter = iter(cycle(dataloader_val))

'''Optim'''
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)

'''Training'''
step = 0
loss_curve = []
val_metrics = []

# lr reduce settings
lr_reduce_steps = [50000, 100000, 150000]
lr_reduce_ratio = 0.5

print('start training...')
print(f'Device: {device}')
print(f'Batch size: {batch_size}')
if device == 'mps':
    print('MPS Memory: Using unified memory architecture')

while step < total_steps:
    model.train()
    
    # Clear cache periodically for MPS
    if step % 100 == 0 and device == 'mps':
        torch.mps.empty_cache()
    
    # Get batch
    try:
        batch = next(dataloader_train_iter)
    except StopIteration:
        dataloader_train_iter = iter(cycle(dataloader_train))
        batch = next(dataloader_train_iter)
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=False)  # Disable non_blocking for MPS
    
    # Forward pass
    optimizer.zero_grad()
    
    # Edge prediction
    predicted_edges = model(batch['corners_withsemantics'], batch['global_attention_matrix'], batch['padding_mask'])
    
    # Compute loss
    loss = F.mse_loss(predicted_edges, batch['edges'])
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Logging
    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        loss_curve.append(loss.item())
    
    # Learning rate scheduling
    if step in lr_reduce_steps:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_reduce_ratio
        print(f'Reduced learning rate to {optimizer.param_groups[0]["lr"]:.2e} at step {step}')
    
    # Validation
    if step % 1000 == 0 and step > 0:
        model.eval()
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for _ in range(10):  # Validate on 10 batches
                try:
                    val_batch = next(dataloader_val_iter)
                except StopIteration:
                    dataloader_val_iter = iter(cycle(dataloader_val))
                    val_batch = next(dataloader_val_iter)
                
                # Move to device
                for key in val_batch:
                    if isinstance(val_batch[key], torch.Tensor):
                        val_batch[key] = val_batch[key].to(device, non_blocking=False)
                
                # Edge prediction
                predicted_edges_val = model(val_batch['corners_withsemantics'], val_batch['global_attention_matrix'], val_batch['padding_mask'])
                
                # Compute loss
                val_loss += F.mse_loss(predicted_edges_val, val_batch['edges']).item()
                val_count += 1
        
        val_loss /= val_count
        val_metrics.append(val_loss)
        print(f'Validation Loss: {val_loss:.6f}')
        
        # Save checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'val_loss': val_loss,
            'loss_curve': loss_curve,
            'val_metrics': val_metrics
        }
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_step_{step}.pth'))
        print(f'Saved checkpoint at step {step}')
    
    step += 1

# Save final model
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
print('Training completed!')
print(f'Final model saved to {output_dir}')
