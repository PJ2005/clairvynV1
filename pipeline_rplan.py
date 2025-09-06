#!/usr/bin/env python3
"""
RPLAN Pipeline Script for Architectural Design Generation

This script orchestrates the workflow using RPLAN pickle files:
1. Load RPLAN pickle file
2. Convert to GSDiff format
3. Run GSDiff inference → Vector primitives
4. Export result to DXF
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# Add current directory to path for imports
sys.path.append('.')

def print_status(step: str, message: str, status: str = "INFO"):
    """Print formatted status messages."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄"
    }
    icon = status_icons.get(status, "ℹ️")
    print(f"[{timestamp}] {icon} {step}: {message}")


def get_rplan_file() -> str:
    """Get RPLAN file path from user."""
    print("\n" + "="*60)
    print("🏠 RPLAN ARCHITECTURAL DESIGN PIPELINE")
    print("="*60)
    print("This pipeline processes RPLAN pickle files to generate floorplans.")
    print("Expected file format: .pkl files in pickle/train or pickle/val directories")
    print("="*60)
    
    # Check for available pickle files
    pickle_train_path = "pickle/train"
    pickle_val_path = "pickle/val"
    
    available_files = []
    if os.path.exists(pickle_train_path):
        train_files = [f for f in os.listdir(pickle_train_path) if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
        available_files.extend([(os.path.join(pickle_train_path, f), "train") for f in train_files[:10]])  # Show first 10
    
    if os.path.exists(pickle_val_path):
        val_files = [f for f in os.listdir(pickle_val_path) if f.endswith(('.pkl', '.pickle', '.p', '.npy'))]
        available_files.extend([(os.path.join(pickle_val_path, f), "val") for f in val_files[:10]])  # Show first 10
    
    if available_files:
        print(f"\nFound {len(available_files)} pickle files:")
        for i, (file_path, split) in enumerate(available_files, 1):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  {i}. {file_path} ({split}, {file_size:.1f} MB)")
        print(f"  {len(available_files) + 1}. Enter custom path")
    else:
        print("\nNo pickle files found in pickle/train or pickle/val directories")
        print("Please ensure your .pkl files are in the correct directories")
    
    while True:
        try:
            if available_files:
                choice = input(f"\nSelect file (1-{len(available_files) + 1}): ").strip()
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(available_files):
                        return available_files[choice_num - 1][0]
                    elif choice_num == len(available_files) + 1:
                        # Custom path
                        file_path = input("Enter custom path to RPLAN file: ").strip()
                    else:
                        print("Invalid choice. Please try again.")
                        continue
                else:
                    file_path = choice
            else:
                file_path = input("\nEnter path to RPLAN file: ").strip()
            
            if file_path:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        return file_path
                    else:
                        print(f"Path is a directory, not a file: {file_path}")
                        print("Please select a specific file.")
                else:
                    print(f"File not found: {file_path}")
            else:
                print("Please enter a file path.")
        except KeyboardInterrupt:
            print("\n\nPipeline cancelled by user.")
            sys.exit(0)
        except EOFError:
            print("\n\nPipeline cancelled.")
            sys.exit(0)


def step1_load_rplan(file_path: str) -> Optional[Dict[str, Any]]:
    """Step 1: Load RPLAN pickle file."""
    print_status("STEP 1", f"Loading RPLAN data from {file_path}...", "PROGRESS")
    
    try:
        # Handle different file formats
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True).item()
        elif file_path.endswith(('.pkl', '.pickle')):
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            # Try numpy first, then pickle
            try:
                data = np.load(file_path, allow_pickle=True).item()
            except:
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
        
        if data:
            print_status("STEP 1", f"RPLAN data loaded successfully", "SUCCESS")
            print_status("STEP 1", f"Data keys: {list(data.keys())}")
            
            # Show data structure
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print_status("STEP 1", f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print_status("STEP 1", f"  {key}: {type(value)}")
            
            return data
        else:
            print_status("STEP 1", "Failed to load RPLAN data", "ERROR")
            return None
            
    except Exception as e:
        print_status("STEP 1", f"Error loading RPLAN data: {e}", "ERROR")
        print_status("STEP 1", "Make sure the file is a valid pickle or numpy file", "WARNING")
        return None


def step2_convert_to_tensors(rplan_data: Dict[str, Any]) -> Optional[tuple]:
    """Step 2: Convert RPLAN data to GSDiff tensor format."""
    print_status("STEP 2", "Converting RPLAN data to GSDiff format...", "PROGRESS")
    
    try:
        # Extract corners with semantics (53, 16)
        corners_withsemantics = rplan_data['corner_list_np_normalized_padding_withsemantics']
        
        # Simplify semantics to 9 dimensions as in the original code
        corners_simplified = np.zeros((corners_withsemantics.shape[0], 9))
        
        # Copy coordinates (columns 0, 1)
        corners_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
        
        # Combine semantic features
        corners_simplified[:, 2] = corners_withsemantics[:, [2, 6, 12]].sum(axis=1)
        corners_simplified[:, 3] = corners_withsemantics[:, [3, 7, 8, 9, 10]].sum(axis=1)
        corners_simplified[:, 4] = corners_withsemantics[:, [13, 14]].sum(axis=1)
        corners_simplified[:, 5] = corners_withsemantics[:, 4]
        corners_simplified[:, 6] = corners_withsemantics[:, 5]
        corners_simplified[:, 7] = corners_withsemantics[:, 11]
        corners_simplified[:, 8] = corners_withsemantics[:, 15]
        
        # Create tensors
        import torch
        corners_tensor = torch.tensor(corners_simplified, dtype=torch.float32).unsqueeze(0)  # (1, 53, 9)
        global_attn = torch.ones(1, 53, 53, dtype=torch.bool)
        padding_mask = torch.tensor(rplan_data['padding_mask'], dtype=torch.bool).unsqueeze(0)  # (1, 53)
        
        print_status("STEP 2", f"Converted to tensors: corners {corners_tensor.shape}, attention {global_attn.shape}, mask {padding_mask.shape}", "SUCCESS")
        
        return corners_tensor, global_attn, padding_mask
        
    except Exception as e:
        print_status("STEP 2", f"Error converting to tensors: {e}", "ERROR")
        return None


def step3_simulate_gsdiff(tensors: tuple) -> List[Dict[str, Any]]:
    """Step 3: Simulate GSDiff inference (since models aren't available)."""
    print_status("STEP 3", "Simulating GSDiff inference...", "PROGRESS")
    
    try:
        corners_tensor, global_attn, padding_mask = tensors
        
        # Simulate diffusion process by adding some noise and processing
        corners_np = corners_tensor.numpy()
        
        # Extract valid corners
        valid_mask = padding_mask.numpy()[0]
        valid_corners = corners_np[0, valid_mask]
        
        # Generate simple floorplan from corners
        primitives = []
        
        # Convert normalized coordinates to pixel coordinates
        resolution = 512
        pixel_corners = []
        for corner in valid_corners:
            x, y = corner[:2]
            pixel_x = int((x + 1) * resolution // 2)
            pixel_y = int((y + 1) * resolution // 2)
            pixel_corners.append((pixel_x, pixel_y))
        
        # Create simple rectangular rooms from corners
        if len(pixel_corners) >= 4:
            # Create a simple rectangular room
            min_x = min(p[0] for p in pixel_corners)
            max_x = max(p[0] for p in pixel_corners)
            min_y = min(p[1] for p in pixel_corners)
            max_y = max(p[1] for p in pixel_corners)
            
            # Add walls
            primitives.extend([
                {'type': 'line', 'start': (min_x, min_y), 'end': (max_x, min_y), 'thickness': 3},
                {'type': 'line', 'start': (max_x, min_y), 'end': (max_x, max_y), 'thickness': 3},
                {'type': 'line', 'start': (max_x, max_y), 'end': (min_x, max_y), 'thickness': 3},
                {'type': 'line', 'start': (min_x, max_y), 'end': (min_x, min_y), 'thickness': 3},
            ])
        
        # Add corner points
        for corner in pixel_corners:
            primitives.append({
                'type': 'point',
                'position': corner,
                'radius': 3
            })
        
        print_status("STEP 3", f"Simulated GSDiff inference: {len(primitives)} primitives generated", "SUCCESS")
        return primitives
        
    except Exception as e:
        print_status("STEP 3", f"Error in GSDiff simulation: {e}", "ERROR")
        return []


def step4_export_dxf(primitives: List[Dict[str, Any]], file_path: str) -> bool:
    """Step 4: Export vector primitives to DXF file."""
    print_status("STEP 4", "Exporting to DXF format...", "PROGRESS")
    
    try:
        from dxf_generator import export_dxf
        
        # Generate filename based on input file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rplan_{base_name}_{timestamp}.dxf"
        
        print_status("STEP 4", f"Exporting {len(primitives)} primitives to {filename}...")
        
        success = export_dxf(primitives, filename)
        
        if success:
            print_status("STEP 4", f"DXF file saved successfully: {filename}", "SUCCESS")
            return True
        else:
            print_status("STEP 4", "Failed to save DXF file", "ERROR")
            return False
            
    except ImportError as e:
        print_status("STEP 4", f"Import error: {e}", "ERROR")
        return False
    except Exception as e:
        print_status("STEP 4", f"Error exporting DXF: {e}", "ERROR")
        return False


def run_rplan_pipeline():
    """Run the RPLAN architectural design pipeline."""
    start_time = time.time()
    
    try:
        # Step 1: Get RPLAN file and load data
        file_path = get_rplan_file()
        rplan_data = step1_load_rplan(file_path)
        
        if not rplan_data:
            print_status("PIPELINE", "Pipeline failed at Step 1", "ERROR")
            return False
        
        # Step 2: Convert to tensors
        tensors = step2_convert_to_tensors(rplan_data)
        
        if not tensors:
            print_status("PIPELINE", "Pipeline failed at Step 2", "ERROR")
            return False
        
        # Step 3: Simulate GSDiff inference
        primitives = step3_simulate_gsdiff(tensors)
        
        if not primitives:
            print_status("PIPELINE", "Pipeline failed at Step 3", "ERROR")
            return False
        
        # Step 4: Export to DXF
        dxf_success = step4_export_dxf(primitives, file_path)
        
        if not dxf_success:
            print_status("PIPELINE", "Pipeline failed at Step 4", "ERROR")
            return False
        
        # Pipeline completed successfully
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print_status("PIPELINE", f"RPLAN pipeline completed successfully in {duration:.2f} seconds!", "SUCCESS")
        print("="*60)
        
        # Print summary
        print(f"📊 SUMMARY:")
        print(f"   • Input: {os.path.basename(file_path)}")
        print(f"   • Corners: {rplan_data['corner_list_np_normalized_padding_withsemantics'].shape[0]}")
        print(f"   • Valid corners: {rplan_data['padding_mask'].sum()}")
        print(f"   • Primitives: {len(primitives)} generated")
        print(f"   • Output: DXF file created")
        print(f"   • Duration: {duration:.2f} seconds")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
        return False
    except Exception as e:
        print_status("PIPELINE", f"Unexpected error: {e}", "ERROR")
        return False


def main():
    """Main entry point."""
    try:
        # Check if required modules are available
        required_modules = ['dxf_generator']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print_status("SETUP", f"Missing required modules: {', '.join(missing_modules)}", "ERROR")
            print("Please ensure all modules are properly installed and accessible.")
            return 1
        
        # Run the pipeline
        success = run_rplan_pipeline()
        
        if success:
            print("\n🎉 RPLAN pipeline completed successfully!")
            print("You can now open the generated DXF file in any CAD application.")
            return 0
        else:
            print("\n💥 Pipeline failed. Please check the error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

