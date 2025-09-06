import ezdxf
from ezdxf import colors
from typing import Dict, Any, List, Tuple, Optional
import os


class DXFGenerator:
    """
    Generate DXF files from vector primitives (lines, polylines, points).
    """
    
    def __init__(self, dxfversion: str = "R2010"):
        """
        Initialize DXF generator.
        
        Args:
            dxfversion: DXF version to use (default: R2010 for compatibility)
        """
        self.dxfversion = dxfversion
        self.doc = None
        self.msp = None  # Model space
        
    def create_document(self):
        """Create a new DXF document."""
        try:
            self.doc = ezdxf.new(self.dxfversion)
            self.msp = self.doc.modelspace()
            return True
        except Exception as e:
            print(f"Error creating DXF document: {e}")
            return False
    
    def add_line(self, start_point: Tuple[float, float], end_point: Tuple[float, float], 
                 layer: str = "WALLS", color: int = colors.BLACK, lineweight: int = 30):
        """
        Add a line to the DXF document.
        
        Args:
            start_point: Start coordinates (x, y)
            end_point: End coordinates (x, y)
            layer: Layer name for the line
            color: Line color (DXF color index)
            lineweight: Line weight in 1/100mm
        """
        if self.msp is None:
            print("Error: DXF document not created")
            return False
        
        try:
            # Create layer if it doesn't exist
            if layer not in self.doc.layers:
                self.doc.layers.new(name=layer)
            
            # Add line to model space
            self.msp.add_line(
                start=start_point,
                end=end_point,
                dxfattribs={
                    'layer': layer,
                    'color': color,
                    'lineweight': lineweight
                }
            )
            return True
        except Exception as e:
            print(f"Error adding line: {e}")
            return False
    
    def add_point(self, position: Tuple[float, float], layer: str = "CORNERS", 
                  color: int = colors.RED, size: float = 3.0):
        """
        Add a point to the DXF document.
        
        Args:
            position: Point coordinates (x, y)
            layer: Layer name for the point
            color: Point color (DXF color index)
            size: Point size
        """
        if self.msp is None:
            print("Error: DXF document not created")
            return False
        
        try:
            # Create layer if it doesn't exist
            if layer not in self.doc.layers:
                self.doc.layers.new(name=layer)
            
            # Add point to model space
            self.msp.add_point(
                position,
                dxfattribs={
                    'layer': layer,
                    'color': color
                }
            )
            return True
        except Exception as e:
            print(f"Error adding point: {e}")
            return False
    
    def process_primitives(self, primitives: List[Dict[str, Any]]) -> bool:
        """
        Process a list of vector primitives and add them to the DXF document.
        
        Args:
            primitives: List of primitive dictionaries from GSDiff
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.create_document():
            return False
        
        success_count = 0
        total_count = len(primitives)
        
        for primitive in primitives:
            primitive_type = primitive.get('type', '')
            
            try:
                if primitive_type == 'line':
                    start = primitive.get('start', (0, 0))
                    end = primitive.get('end', (0, 0))
                    thickness = primitive.get('thickness', 3)
                    
                    # Convert thickness to lineweight (approximate)
                    lineweight = max(10, min(100, int(thickness * 10)))
                    
                    if self.add_line(start, end, lineweight=lineweight):
                        success_count += 1
                
                elif primitive_type == 'point':
                    position = primitive.get('position', (0, 0))
                    radius = primitive.get('radius', 3)
                    
                    if self.add_point(position, size=radius):
                        success_count += 1
                
                else:
                    print(f"Warning: Unknown primitive type: {primitive_type}")
                    
            except Exception as e:
                print(f"Error processing primitive {primitive_type}: {e}")
                continue
        
        print(f"Successfully processed {success_count}/{total_count} primitives")
        return success_count > 0
    
    def save_dxf(self, filename: str) -> bool:
        """
        Save the DXF document to a file.
        
        Args:
            filename: Output filename (with or without .dxf extension)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.doc is None:
            print("Error: No DXF document to save")
            return False
        
        try:
            # Ensure filename has .dxf extension
            if not filename.lower().endswith('.dxf'):
                filename += '.dxf'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Save the document
            self.doc.saveas(filename)
            print(f"DXF file saved successfully: {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving DXF file: {e}")
            return False


def export_dxf(primitives: List[Dict[str, Any]], filename: str) -> bool:
    """
    Export vector primitives to a DXF file.
    
    Args:
        primitives: List of vector primitives from GSDiff
        filename: Output filename for the DXF file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create DXF generator
        generator = DXFGenerator()
        
        # Process primitives
        if not generator.process_primitives(primitives):
            print("Warning: No primitives were processed successfully")
        
        # Save DXF file
        return generator.save_dxf(filename)
        
    except Exception as e:
        print(f"Error exporting DXF: {e}")
        return False


def create_sample_dxf():
    """Create a sample DXF file for testing."""
    # Sample primitives
    sample_primitives = [
        {
            'type': 'line',
            'start': (0, 0),
            'end': (100, 0),
            'thickness': 3
        },
        {
            'type': 'line',
            'start': (100, 0),
            'end': (100, 80),
            'thickness': 3
        },
        {
            'type': 'line',
            'start': (100, 80),
            'end': (0, 80),
            'thickness': 3
        },
        {
            'type': 'line',
            'start': (0, 80),
            'end': (0, 0),
            'thickness': 3
        },
        {
            'type': 'point',
            'position': (0, 0),
            'radius': 3
        },
        {
            'type': 'point',
            'position': (100, 0),
            'radius': 3
        },
        {
            'type': 'point',
            'position': (100, 80),
            'radius': 3
        },
        {
            'type': 'point',
            'position': (0, 80),
            'radius': 3
        }
    ]
    
    # Export to DXF
    success = export_dxf(sample_primitives, "sample_floorplan.dxf")
    if success:
        print("Sample DXF file created successfully")
    else:
        print("Failed to create sample DXF file")


if __name__ == "__main__":
    print("DXF Generator Module")
    print("=" * 25)
    
    # Create sample DXF file
    create_sample_dxf()
