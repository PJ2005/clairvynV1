#!/usr/bin/env python3
"""
Main Pipeline Script for Architectural Design Generation

This script orchestrates the complete workflow:
1. Natural language input → Design intent JSON
2. JSON → NetworkX graph
3. Graph → GSDiff inference → Vector primitives
4. Vector primitives → DXF file
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

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


def get_user_input() -> str:
    """Get natural language design description from user."""
    print("\n" + "="*60)
    print("🏠 ARCHITECTURAL DESIGN PIPELINE")
    print("="*60)
    print("Describe your architectural design in natural language.")
    print("Examples:")
    print("  - 'Create a 3-bedroom house with open kitchen and living room'")
    print("  - 'Design a 2-bedroom apartment with balcony'")
    print("  - 'Build a studio apartment with kitchenette'")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter your design description: ").strip()
            if user_input:
                return user_input
            else:
                print("Please enter a description.")
        except KeyboardInterrupt:
            print("\n\nPipeline cancelled by user.")
            sys.exit(0)
        except EOFError:
            print("\n\nPipeline cancelled.")
            sys.exit(0)


def step1_get_design_intent(prompt: str) -> Optional[Dict[str, Any]]:
    """Step 1: Convert natural language to structured design intent."""
    print_status("STEP 1", "Converting natural language to design intent...", "PROGRESS")
    
    try:
        from schema_handler import get_design_intent
        
        print_status("STEP 1", f"Calling OpenRouter API with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        design_intent = get_design_intent(prompt)
        
        if design_intent:
            num_rooms = len(design_intent.get('rooms', []))
            num_edges = len(design_intent.get('adjacency_edges', []))
            print_status("STEP 1", f"Design intent generated successfully: {num_rooms} rooms, {num_edges} adjacencies", "SUCCESS")
            return design_intent
        else:
            print_status("STEP 1", "Failed to generate design intent", "ERROR")
            return None
            
    except ImportError as e:
        print_status("STEP 1", f"Import error: {e}", "ERROR")
        return None
    except Exception as e:
        print_status("STEP 1", f"Error generating design intent: {e}", "ERROR")
        return None


def step2_convert_to_graph(design_intent: Dict[str, Any]) -> Optional[Any]:
    """Step 2: Convert design intent JSON to NetworkX graph."""
    print_status("STEP 2", "Converting design intent to graph structure...", "PROGRESS")
    
    try:
        from graph_converter import json_to_graph, get_graph_summary
        
        print_status("STEP 2", "Creating NetworkX graph with GSDiff-ready attributes...")
        
        G = json_to_graph(design_intent)
        
        if G:
            summary = get_graph_summary(G)
            print_status("STEP 2", f"Graph created successfully: {summary['num_nodes']} nodes, {summary['num_edges']} edges", "SUCCESS")
            print_status("STEP 2", f"Room types: {', '.join(summary['room_types'])}")
            return G
        else:
            print_status("STEP 2", "Failed to create graph", "ERROR")
            return None
            
    except ImportError as e:
        print_status("STEP 2", f"Import error: {e}", "ERROR")
        return None
    except Exception as e:
        print_status("STEP 2", f"Error converting to graph: {e}", "ERROR")
        return None


def step3_run_gsdiff(G: Any) -> Optional[List[Dict[str, Any]]]:
    """Step 3: Run GSDiff inference to generate vector primitives."""
    print_status("STEP 3", "Running GSDiff inference...", "PROGRESS")
    
    try:
        from gsdiff_runner import GSDiffRunner
        
        print_status("STEP 3", "Initializing GSDiff runner...")
        runner = GSDiffRunner()
        
        print_status("STEP 3", "Generating floorplan from graph...")
        primitives = runner.generate_floorplan(G)
        
        if primitives:
            num_lines = len([p for p in primitives if p.get('type') == 'line'])
            num_points = len([p for p in primitives if p.get('type') == 'point'])
            print_status("STEP 3", f"GSDiff inference completed: {len(primitives)} primitives ({num_lines} lines, {num_points} points)", "SUCCESS")
            return primitives
        else:
            print_status("STEP 3", "No primitives generated", "WARNING")
            return []
            
    except ImportError as e:
        print_status("STEP 3", f"Import error: {e}", "ERROR")
        return None
    except Exception as e:
        print_status("STEP 3", f"Error running GSDiff: {e}", "ERROR")
        return None


def step4_export_dxf(primitives: List[Dict[str, Any]], prompt: str) -> bool:
    """Step 4: Export vector primitives to DXF file."""
    print_status("STEP 4", "Exporting to DXF format...", "PROGRESS")
    
    try:
        from dxf_generator import export_dxf
        
        # Generate filename based on prompt and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"floorplan_{safe_prompt}_{timestamp}.dxf"
        
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


def run_pipeline():
    """Run the complete architectural design pipeline."""
    start_time = time.time()
    
    try:
        # Step 1: Get user input and convert to design intent
        prompt = get_user_input()
        design_intent = step1_get_design_intent(prompt)
        
        if not design_intent:
            print_status("PIPELINE", "Pipeline failed at Step 1", "ERROR")
            return False
        
        # Step 2: Convert to graph
        G = step2_convert_to_graph(design_intent)
        
        if not G:
            print_status("PIPELINE", "Pipeline failed at Step 2", "ERROR")
            return False
        
        # Step 3: Run GSDiff inference
        primitives = step3_run_gsdiff(G)
        
        if primitives is None:
            print_status("PIPELINE", "Pipeline failed at Step 3", "ERROR")
            return False
        
        # Step 4: Export to DXF
        dxf_success = step4_export_dxf(primitives, prompt)
        
        if not dxf_success:
            print_status("PIPELINE", "Pipeline failed at Step 4", "ERROR")
            return False
        
        # Pipeline completed successfully
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print_status("PIPELINE", f"Pipeline completed successfully in {duration:.2f} seconds!", "SUCCESS")
        print("="*60)
        
        # Print summary
        print(f"📊 SUMMARY:")
        print(f"   • Design: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        print(f"   • Rooms: {len(design_intent.get('rooms', []))}")
        print(f"   • Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
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
        required_modules = ['schema_handler', 'graph_converter', 'gsdiff_runner', 'dxf_generator']
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
        success = run_pipeline()
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
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
