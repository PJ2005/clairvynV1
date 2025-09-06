#!/usr/bin/env python3
"""
GSDiff Runner for RPLAN Pickle Files

This module adapts the GSDiff runner to work with RPLAN pickle files
instead of the original floorplan images.
"""

import sys
import os
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
import math
import cv2
from PIL import Image, ImageDraw

# Add GSDiff paths to system path
sys.path.append('./GSDiff')
sys.path.append('./GSDiff/gsdiff')
sys.path.append('./GSDiff/scripts/metrics')

# Import GSDiff modules
from gsdiff.house_nn1 import HeterHouseModel
from gsdiff.house_nn2 import EdgeModel
from gsdiff.utils import (
    get_cycle_basis_and_semantic_3_semansimplified,
    edges_to_coordinates,
    visualize_withsemantic
)


class GSDiffRPlanRunner:
    """
    Wrapper for GSDiff model that works with RPLAN pickle files.
    """
    
    def __init__(self, 
                 model_dir: str = "./GSDiff/outputs",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize GSDiff runner with pre-trained models.
        
        Args:
            model_dir: Directory containing pre-trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_dir = model_dir
        
        # Diffusion parameters
        self.diffusion_steps = 1000
        self.resolution = 512
        self.aa_scale = 1
        
        # Initialize models
        self.edge_model = None
        self.node_model = None
        
        # Load models
        self._load_models()
        
        # Setup diffusion parameters
        self._setup_diffusion()
    
    def _load_models(self):
        """Load pre-trained GSDiff models."""
        try:
            # Load Edge Model (structure-2)
            edge_model_path = os.path.join(self.model_dir, "structure-2", "model_stage2_best_061000.pt")
            if os.path.exists(edge_model_path):
                self.edge_model = EdgeModel().to(self.device)
                self.edge_model.load_state_dict(torch.load(edge_model_path, map_location=self.device))
                self.edge_model.eval()
                print(f"✓ Edge model loaded from {edge_model_path}")
            else:
                print(f"⚠ Edge model not found at {edge_model_path}")
            
            # Load Node Model (structure-1)
            node_model_path = os.path.join(self.model_dir, "structure-1", "model1000000.pt")
            if os.path.exists(node_model_path):
                self.node_model = HeterHouseModel().to(self.device)
                self.node_model.load_state_dict(torch.load(node_model_path, map_location=self.device))
                self.node_model.eval()
                print(f"✓ Node model loaded from {node_model_path}")
            else:
                print(f"⚠ Node model not found at {node_model_path}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure models are downloaded and placed in the correct directory")
    
    def _setup_diffusion(self):
        """Setup diffusion parameters for inference."""
        # Cosine beta schedule
        alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
        betas = []
        max_beta = 0.999
        
        for i in range(self.diffusion_steps):
            t1 = i / self.diffusion_steps
            t2 = (i + 1) / self.diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        
        self.betas = np.array(betas, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        
        # Gaussian diffusion settings
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def load_rplan_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load RPLAN pickle file and return the data structure.
        
        Args:
            file_path: Path to the .npy pickle file
            
        Returns:
            Dictionary containing the RPLAN data
        """
        try:
            data = np.load(file_path, allow_pickle=True).item()
            print(f"✓ Loaded RPLAN data from {file_path}")
            print(f"  Data keys: {list(data.keys())}")
            
            # Print data structure
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            
            return data
            
        except Exception as e:
            print(f"Error loading RPLAN data: {e}")
            return {}
    
    def rplan_to_tensor_input(self, rplan_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert RPLAN data to tensor format expected by GSDiff.
        
        Args:
            rplan_data: RPLAN data dictionary
            
        Returns:
            Tuple of (corners_with_semantics, global_attention_matrix, corners_padding_mask)
        """
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
            
            # Convert to tensor
            corners_tensor = torch.tensor(corners_simplified, dtype=torch.float32).unsqueeze(0)  # (1, 53, 9)
            
            # Create global attention matrix (fully connected)
            global_attn = torch.ones(1, 53, 53, dtype=torch.bool)
            
            # Get padding mask
            padding_mask = torch.tensor(rplan_data['padding_mask'], dtype=torch.bool).unsqueeze(0)  # (1, 53)
            
            print(f"✓ Converted RPLAN data to tensors:")
            print(f"  Corners: {corners_tensor.shape}")
            print(f"  Global attention: {global_attn.shape}")
            print(f"  Padding mask: {padding_mask.shape}")
            
            return corners_tensor, global_attn, padding_mask
            
        except Exception as e:
            print(f"Error converting RPLAN data to tensors: {e}")
            return None, None, None
    
    def _run_diffusion_inference(self, 
                                 corners_input: torch.Tensor,
                                 global_attn: torch.Tensor,
                                 padding_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run diffusion inference to generate wall junctions and edges.
        
        Args:
            corners_input: Input corners tensor
            global_attn: Global attention matrix
            padding_mask: Padding mask for valid corners
            
        Returns:
            Tuple of (generated_corners, generated_edges)
        """
        if self.node_model is None:
            raise RuntimeError("Node model not loaded")
        
        # Move to device
        corners_input = corners_input.to(self.device)
        global_attn = global_attn.to(self.device)
        padding_mask = padding_mask.to(self.device)
        
        # Add padding mask to input
        corners_with_padding = torch.cat((corners_input, (1 - padding_mask).type(corners_input.dtype)), dim=2)
        
        # Reverse diffusion process: 999->998->...->1->0
        corners_t = torch.randn(*corners_with_padding.shape, device=self.device)
        
        for current_step in range(self.diffusion_steps - 1, -1, -1):
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.node_model(corners_t, global_attn)
            
            # Calculate mean
            alpha_t = self.alphas_cumprod[current_step]
            alpha_t_prev = self.alphas_cumprod_prev[current_step]
            beta_t = self.betas[current_step]
            
            if current_step > 0:
                noise = torch.randn_like(corners_t)
            else:
                noise = torch.zeros_like(corners_t)
            
            # Update corners
            corners_t = (1 / np.sqrt(alpha_t)) * (
                corners_t - ((1 - alpha_t) / np.sqrt(1 - alpha_t)) * predicted_noise
            ) + np.sqrt(beta_t) * noise
        
        # Convert back to numpy
        generated_corners = corners_t.cpu().numpy()
        generated_edges = self._generate_edges_from_corners(generated_corners, global_attn.cpu().numpy())
        
        return generated_corners, generated_edges
    
    def _generate_edges_from_corners(self, corners: np.ndarray, global_attn: np.ndarray) -> np.ndarray:
        """Generate edge connections from generated corners."""
        # This is a simplified edge generation - in practice, use the edge model
        batch_size, num_corners, _ = corners.shape
        edges = np.zeros((batch_size, num_corners, num_corners, 2))
        
        # Simple heuristic: connect corners that are close to each other
        for b in range(batch_size):
            for i in range(num_corners):
                for j in range(i + 1, num_corners):
                    # Extract 2D coordinates
                    pos_i = corners[b, i, :2]
                    pos_j = corners[b, j, :2]
                    
                    # Calculate distance
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # Connect if close enough (threshold)
                    if distance < 0.3:  # Normalized distance threshold
                        edges[b, i, j, 1] = 1  # Edge exists
                        edges[b, j, i, 1] = 1  # Symmetric
        
        return edges
    
    def _extract_vector_primitives(self, 
                                  corners: np.ndarray, 
                                  edges: np.ndarray,
                                  padding_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract vector primitives (lines, polylines) from generated corners and edges.
        
        Args:
            corners: Generated corner coordinates
            edges: Generated edge connections
            padding_mask: Mask indicating valid corners
            
        Returns:
            List of vector primitives
        """
        primitives = []
        
        for batch_idx in range(corners.shape[0]):
            # Extract valid corners for this batch
            valid_mask = padding_mask[batch_idx]
            valid_corners = corners[batch_idx, valid_mask]
            
            # Convert normalized coordinates back to pixel coordinates
            pixel_corners = []
            for corner in valid_corners:
                x, y = corner[:2]
                pixel_x = int((x + 1) * self.resolution // 2)
                pixel_y = int((y + 1) * self.resolution // 2)
                pixel_corners.append((pixel_x, pixel_y))
            
            # Extract edges for this batch
            batch_edges = edges[batch_idx]
            
            # Generate wall segments
            wall_segments = []
            for i in range(len(pixel_corners)):
                for j in range(i + 1, len(pixel_corners)):
                    if batch_edges[i, j, 1] > 0.5:  # Edge threshold
                        start_point = pixel_corners[i]
                        end_point = pixel_corners[j]
                        
                        wall_segments.append({
                            'type': 'line',
                            'start': start_point,
                            'end': end_point,
                            'thickness': 3
                        })
            
            # Add wall segments to primitives
            primitives.extend(wall_segments)
            
            # Add corner points
            for corner in pixel_corners:
                primitives.append({
                    'type': 'point',
                    'position': corner,
                    'radius': 3
                })
        
        return primitives
    
    def generate_floorplan_from_rplan(self, rplan_file: str) -> List[Dict[str, Any]]:
        """
        Generate floorplan from RPLAN pickle file.
        
        Args:
            rplan_file: Path to RPLAN .npy file
            
        Returns:
            List of vector primitives (lines, polylines, points)
        """
        try:
            # Load RPLAN data
            rplan_data = self.load_rplan_data(rplan_file)
            if not rplan_data:
                return []
            
            # Convert to tensor input
            corners_input, global_attn, padding_mask = self.rplan_to_tensor_input(rplan_data)
            if corners_input is None:
                return []
            
            # Run diffusion inference
            generated_corners, generated_edges = self._run_diffusion_inference(
                corners_input, global_attn, padding_mask
            )
            
            # Extract vector primitives
            primitives = self._extract_vector_primitives(
                generated_corners, generated_edges, padding_mask
            )
            
            return primitives
            
        except Exception as e:
            print(f"Error generating floorplan from RPLAN: {e}")
            return []
    
    def save_floorplan_image(self, 
                            primitives: List[Dict[str, Any]], 
                            output_path: str,
                            size: int = 512):
        """
        Save floorplan as an image for visualization.
        
        Args:
            primitives: List of vector primitives
            output_path: Path to save the image
            size: Image size
        """
        # Create blank image
        img = Image.new('RGB', (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw primitives
        for primitive in primitives:
            if primitive['type'] == 'line':
                start = primitive['start']
                end = primitive['end']
                thickness = primitive.get('thickness', 3)
                draw.line([start, end], fill=(0, 0, 0), width=thickness)
            
            elif primitive['type'] == 'point':
                pos = primitive['position']
                radius = primitive.get('radius', 3)
                draw.ellipse([
                    pos[0] - radius, pos[1] - radius,
                    pos[0] + radius, pos[1] + radius
                ], fill=(255, 0, 0))
        
        # Save image
        img.save(output_path)
        print(f"Floorplan image saved to {output_path}")


def test_rplan_runner():
    """Test the RPLAN runner with sample data."""
    print("Testing GSDiff RPLAN Runner...")
    
    # Create sample RPLAN data
    sample_data = {
        'corner_list_np_normalized_padding_withsemantics': np.random.rand(53, 16),
        'padding_mask': np.ones(53, dtype=np.uint8),
        'edges': np.random.rand(2809)
    }
    
    # Save sample data
    os.makedirs('test_data', exist_ok=True)
    sample_file = 'test_data/sample_rplan.npy'
    np.save(sample_file, sample_data)
    
    # Test the runner
    runner = GSDiffRPlanRunner()
    
    # Load and process the data
    primitives = runner.generate_floorplan_from_rplan(sample_file)
    
    if primitives:
        print(f"✓ Generated {len(primitives)} primitives")
        runner.save_floorplan_image(primitives, "test_rplan_output.png")
        return True
    else:
        print("✗ No primitives generated")
        return False


if __name__ == "__main__":
    test_rplan_runner()

