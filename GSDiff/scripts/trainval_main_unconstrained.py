import sys
import os

# Add current directory to path for M3 Air compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
gsdiff_root = os.path.dirname(current_dir)
sys.path.append(gsdiff_root)
sys.path.append(os.path.join(gsdiff_root, 'datasets'))
sys.path.append(os.path.join(gsdiff_root, 'gsdiff'))
sys.path.append(os.path.join(gsdiff_root, 'scripts', 'metrics'))

import math
import torch
import shutil
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from itertools import cycle
from datasets.rplang_edge_semantics_simplified_55_100 import RPlanGEdgeSemanSimplified_55_100
from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from gsdiff.house_nn1 import HeterHouseModel
from gsdiff.house_nn3 import EdgeModel
from gsdiff.utils import *
import torch.nn.functional as F
from scripts.metrics.fid import fid
from scripts.metrics.kid import kid

'''MPS-optimized script for training the first stage node generation model on M3 Air'''

# ===== OPTIMIZED PARAMETERS FOR M3 AIR MPS =====
diffusion_steps = 1000
lr = 1e-4
weight_decay = 0
total_steps = 1000000
batch_size = 64  # Reduced for M3 Air memory constraints
batch_size_val = 512  # Reduced for validation
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
merge_points = False
clamp_trick_training = True

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

def map_to_binary(tensor):
    batch_size, n_values = tensor.shape
    binary_tensor = torch.zeros((batch_size, n_values, 12), dtype=torch.float32, device=tensor.device)

    # Create a mask to mark values other than 99999
    mask = tensor != 99999

    # Processing values other than 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # Separate integer and fractional parts
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # Processing the integer part
    for i in range(8):
        binary_tensor[:, :, 7 - i] = integer_part % 2
        integer_part //= 2

    # Processing decimals
    fractional_part *= 16
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(4):
        binary_tensor[:, :, 11 - i] = fractional_part % 2
        fractional_part //= 4

    # Use mask to ensure that the original 99999 value is 0 in the binary vector
    binary_tensor = torch.where(mask.unsqueeze(-1), binary_tensor, torch.zeros_like(binary_tensor))

    return binary_tensor

def map_to_fournary(tensor):
    batch_size, n_values = tensor.shape
    fournary_tensor = torch.zeros((batch_size, n_values, 6), dtype=torch.float32, device=tensor.device)

    # Create a mask to mark values other than 99999
    mask = tensor != 99999

    # Processing values other than 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # Separate integer and fractional parts
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # Processing the integer part
    for i in range(4):
        fournary_tensor[:, :, 3 - i] = integer_part % 4
        integer_part //= 4

    # Processing decimals
    fractional_part *= 16
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        fournary_tensor[:, :, 5 - i] = fractional_part % 4
        fractional_part //= 4

    # Use mask to ensure that the original 99999 value is 0
    fournary_tensor = torch.where(mask.unsqueeze(-1), fournary_tensor, torch.zeros_like(fournary_tensor))

    return fournary_tensor

def map_to_eightnary(tensor):
    batch_size, n_values = tensor.shape
    eightnary_tensor = torch.zeros((batch_size, n_values, 5), dtype=torch.float32, device=tensor.device)

    # Create a mask to mark values other than 99999
    mask = tensor != 99999

    # Processing values other than 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # Separate integer and fractional parts
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # Processing the integer part
    for i in range(3):
        eightnary_tensor[:, :, 2 - i] = integer_part % 8
        integer_part //= 8

    # Processing decimals
    fractional_part *= 8
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        eightnary_tensor[:, :, 4 - i] = fractional_part % 8
        fractional_part //= 8

    # Use mask to ensure that the original 99999 value is 0
    eightnary_tensor = torch.where(mask.unsqueeze(-1), eightnary_tensor, torch.zeros_like(eightnary_tensor))

    return eightnary_tensor

def map_to_sixteennary(tensor):
    batch_size, n_values = tensor.shape
    sixteennary_tensor = torch.zeros((batch_size, n_values, 4), dtype=torch.float32, device=tensor.device)

    # Create a mask to mark values other than 99999
    mask = tensor != 99999

    # Processing values other than 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # Separate integer and fractional parts
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # Processing the integer part
    for i in range(2):
        sixteennary_tensor[:, :, 1 - i] = integer_part % 16
        integer_part //= 16

    # Processing decimals
    fractional_part *= 16
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        sixteennary_tensor[:, :, 3 - i] = fractional_part % 16
        fractional_part //= 16

    # Use mask to ensure that the original 99999 value is 0
    sixteennary_tensor = torch.where(mask.unsqueeze(-1), sixteennary_tensor, torch.zeros_like(sixteennary_tensor))

    return sixteennary_tensor

'''create output_dir'''
output_dir = 'outputs/structure-1/'
os.makedirs(output_dir, exist_ok=True)

'''Diffusion Settings'''
# cosine beta
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
betas = []
max_beta = 0.999
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
betas = np.array(betas, dtype=np.float64)

alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

'''Neural Network'''
model = HeterHouseModel().to(device)
print('total params:', sum(p.numel() for p in model.parameters()))

'''Data'''
dataset_train = RPlanGEdgeSemanSimplified_55_100('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=False)  # Disable pin_memory for MPS
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = RPlanGEdgeSemanSimplified_55_100('val')
dataloader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=2,
                            drop_last=False, pin_memory=False)  # Disable pin_memory for MPS
dataloader_val_iter = iter(cycle(dataloader_val))

'''Optim'''
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)

'''Training'''
step = 0
loss_curve = []
val_metrics = []

# lr reduce settings
lr_reduce_steps = [200000, 400000, 600000, 800000]
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
    
    # Sample random timesteps
    t = torch.randint(0, diffusion_steps, (batch_size,), device=device).long()
    
    # Get noisy data
    noise = torch.randn_like(batch['corners_withsemantics'])
    sqrt_alphas_cumprod_t = torch.tensor(sqrt_recip_alphas_cumprod[t], device=device).float()
    sqrt_one_minus_alphas_cumprod_t = torch.tensor(sqrt_recipm1_alphas_cumprod[t], device=device).float()
    
    noisy_corners = sqrt_alphas_cumprod_t.view(-1, 1, 1) * batch['corners_withsemantics'] + \
                   sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * noise
    
    # Predict noise
    predicted_noise = model(noisy_corners, t, batch['global_attention_matrix'], batch['padding_mask'])
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
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
                
                # Sample timestep
                t_val = torch.randint(0, diffusion_steps, (val_batch['corners_withsemantics'].shape[0],), device=device).long()
                
                # Get noisy data
                noise_val = torch.randn_like(val_batch['corners_withsemantics'])
                sqrt_alphas_cumprod_t_val = torch.tensor(sqrt_recip_alphas_cumprod[t_val], device=device).float()
                sqrt_one_minus_alphas_cumprod_t_val = torch.tensor(sqrt_recipm1_alphas_cumprod[t_val], device=device).float()
                
                noisy_corners_val = sqrt_alphas_cumprod_t_val.view(-1, 1, 1) * val_batch['corners_withsemantics'] + \
                                   sqrt_one_minus_alphas_cumprod_t_val.view(-1, 1, 1) * noise_val
                
                # Predict noise
                predicted_noise_val = model(noisy_corners_val, t_val, val_batch['global_attention_matrix'], val_batch['padding_mask'])
                
                # Compute loss
                val_loss += F.mse_loss(predicted_noise_val, noise_val).item()
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
