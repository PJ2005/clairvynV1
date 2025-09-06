import json
import jsonschema
from typing import Dict, Any, List, Optional
import requests
from dataclasses import dataclass


@dataclass
class Room:
    """Represents a room in the design intent."""
    id: str
    name: str
    type: str
    dimensions: Dict[str, float]
    position: Dict[str, float]
    constraints: Optional[List[str]] = None


@dataclass
class AdjacencyEdge:
    """Represents an adjacency relationship between rooms."""
    room1_id: str
    room2_id: str
    relationship_type: str
    distance: Optional[float] = None


@dataclass
class DesignIntent:
    """Represents the complete design intent."""
    rooms: List[Room]
    adjacency_edges: List[AdjacencyEdge]
    global_constraints: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# JSON Schema for design intent validation
DESIGN_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "number", "minimum": 0},
                            "length": {"type": "number", "minimum": 0},
                            "height": {"type": "number", "minimum": 0}
                        },
                        "required": ["width", "length"],
                        "additionalProperties": False
                    },
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"}
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["id", "name", "type", "dimensions", "position"],
                "additionalProperties": False
            },
            "minItems": 1
        },
        "adjacency_edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "room1_id": {"type": "string"},
                    "room2_id": {"type": "string"},
                    "relationship_type": {"type": "string"},
                    "distance": {"type": "number", "minimum": 0}
                },
                "required": ["room1_id", "room2_id", "relationship_type"],
                "additionalProperties": False
            }
        },
        "global_constraints": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {
            "type": "object",
            "additionalProperties": True
        }
    },
    "required": ["rooms", "adjacency_edges"],
    "additionalProperties": False
}


def validate_design_intent(json_data: Dict[str, Any]) -> bool:
    """
    Validate JSON data against the design intent schema.
    
    Args:
        json_data: Dictionary containing the design intent data
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        jsonschema.ValidationError: If validation fails
    """
    try:
        jsonschema.validate(instance=json_data, schema=DESIGN_INTENT_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e}")
        return False


def validate_room_references(design_intent: Dict[str, Any]) -> bool:
    """
    Validate that all room IDs referenced in adjacency edges exist in the rooms list.
    
    Args:
        design_intent: Validated design intent dictionary
        
    Returns:
        bool: True if all references are valid, False otherwise
    """
    room_ids = {room["id"] for room in design_intent["rooms"]}
    
    for edge in design_intent["adjacency_edges"]:
        if edge["room1_id"] not in room_ids or edge["room2_id"] not in room_ids:
            print(f"Invalid room reference in adjacency edge: {edge}")
            return False
    
    return True


def get_design_intent(prompt: str) -> Dict[str, Any]:
    """
    Get design intent from OpenRouter LLM API based on natural language prompt.
    
    Args:
        prompt: Natural language description of the design intent
        
    Returns:
        dict: Validated design intent JSON
        
    Raises:
        ValueError: If the response cannot be validated
        requests.RequestException: If the API request fails
    """
    # OpenRouter API configuration (placeholder)
    OPENROUTER_API_KEY = "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"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "openai/gpt-oss-120b:free"  # Example model
    
    # System prompt to guide the LLM
    system_prompt = """
    You are an architectural design assistant. Convert the user's natural language description 
    into a structured JSON representation of design intent. The response must include:
    
    1. rooms: Array of room objects with id, name, type, dimensions, position, and optional constraints
    2. adjacency_edges: Array of adjacency relationships between rooms
    3. global_constraints: Optional array of overall design constraints
    4. metadata: Optional additional information
    
    Return ONLY valid JSON, no additional text or explanations.
    """
    
    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Low temperature for consistent JSON output
        "max_tokens": 2000
    }
    
    try:
        # Make the API request
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Extract the response content
        response_data = response.json()
        llm_response = response_data["choices"][0]["message"]["content"]
        
        # Try to parse the JSON response
        try:
            design_intent = json.loads(llm_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}")
        
        # Validate against schema
        if not validate_design_intent(design_intent):
            raise ValueError("Design intent does not match required schema")
        
        # Validate room references
        if not validate_room_references(design_intent):
            raise ValueError("Invalid room references in adjacency edges")
        
        return design_intent
        
    except requests.RequestException as e:
        raise requests.RequestException(f"API request failed: {e}")
    except KeyError as e:
        raise ValueError(f"Unexpected API response format: {e}")


def create_sample_design_intent() -> Dict[str, Any]:
    """
    Create a sample design intent for testing purposes.
    
    Returns:
        dict: Sample design intent that validates against the schema
    """
    return {
        "rooms": [
            {
                "id": "living_room",
                "name": "Living Room",
                "type": "living",
                "dimensions": {"width": 5.0, "length": 6.0, "height": 2.7},
                "position": {"x": 0, "y": 0, "z": 0},
                "constraints": ["natural_light", "view_to_garden"]
            },
            {
                "id": "kitchen",
                "name": "Kitchen",
                "type": "kitchen",
                "dimensions": {"width": 4.0, "length": 3.5, "height": 2.7},
                "position": {"x": 5.0, "y": 0, "z": 0},
                "constraints": ["adjacent_to_dining", "utilities_access"]
            },
            {
                "id": "bedroom",
                "name": "Master Bedroom",
                "type": "bedroom",
                "dimensions": {"width": 4.5, "length": 4.0, "height": 2.7},
                "position": {"x": 0, "y": 6.0, "z": 0},
                "constraints": ["quiet_location", "ensuite_access"]
            }
        ],
        "adjacency_edges": [
            {
                "room1_id": "living_room",
                "room2_id": "kitchen",
                "relationship_type": "adjacent",
                "distance": 0.0
            },
            {
                "room1_id": "living_room",
                "room2_id": "bedroom",
                "relationship_type": "adjacent",
                "distance": 0.0
            }
        ],
        "global_constraints": [
            "max_total_area: 100_sqm",
            "energy_efficient",
            "accessible_design"
        ],
        "metadata": {
            "project_name": "Sample House Design",
            "designer": "AI Assistant",
            "version": "1.0"
        }
    }


if __name__ == "__main__":
    # Test the module functionality
    print("Testing schema_handler.py module...")
    
    # Test with sample data
    sample_data = create_sample_design_intent()
    print(f"Sample data validation: {validate_design_intent(sample_data)}")
    
    # Test room reference validation
    print(f"Room reference validation: {validate_room_references(sample_data)}")
    
    print("Module test completed successfully!")
