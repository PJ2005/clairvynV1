import networkx as nx
from typing import Dict, Any, List, Tuple
import numpy as np


def json_to_graph(json_data: dict) -> nx.Graph:
    """
    Convert validated design intent JSON into a NetworkX graph object.
    
    Args:
        json_data: Validated design intent dictionary from schema_handler
        
    Returns:
        nx.Graph: NetworkX graph with nodes (rooms) and edges (adjacency relationships)
                 Attributes are formatted for GSDiff input
    """
    # Create an undirected graph
    G = nx.Graph()
    
    # Add room nodes with attributes
    for room in json_data["rooms"]:
        node_id = room["id"]
        
        # Extract position coordinates
        pos_x = room["position"]["x"]
        pos_y = room["position"]["y"]
        pos_z = room["position"].get("z", 0.0)  # Default to 0 if not specified
        
        # Extract dimensions
        width = room["dimensions"]["width"]
        length = room["dimensions"]["length"]
        height = room["dimensions"].get("height", 2.7)  # Default height if not specified
        
        # Calculate room center and area
        center_x = pos_x + width / 2
        center_y = pos_y + length / 2
        center_z = pos_z + height / 2
        area = width * length
        
        # Node attributes formatted for GSDiff
        node_attrs = {
            # Position attributes
            "pos": np.array([center_x, center_y, center_z]),
            "x": center_x,
            "y": center_y,
            "z": center_z,
            
            # Geometric attributes
            "width": width,
            "length": length,
            "height": height,
            "area": area,
            "volume": area * height,
            
            # Room properties
            "room_type": room["type"],
            "room_name": room["name"],
            "room_id": room["id"],
            
            # Constraints (if any)
            "constraints": room.get("constraints", []),
            
            # GSDiff-specific attributes
            "node_type": "room",
            "feature_vector": _create_feature_vector(room),
            "bounding_box": _create_bounding_box(pos_x, pos_y, pos_z, width, length, height)
        }
        
        G.add_node(node_id, **node_attrs)
    
    # Add adjacency edges with attributes
    for edge in json_data["adjacency_edges"]:
        room1_id = edge["room1_id"]
        room2_id = edge["room2_id"]
        
        # Get room positions for edge calculations
        room1_pos = G.nodes[room1_id]["pos"]
        room2_pos = G.nodes[room2_id]["pos"]
        
        # Calculate edge attributes
        distance = np.linalg.norm(room1_pos - room2_pos)
        relationship_type = edge["relationship_type"]
        edge_distance = edge.get("distance", distance)
        
        # Edge attributes formatted for GSDiff
        edge_attrs = {
            "relationship_type": relationship_type,
            "distance": edge_distance,
            "euclidean_distance": distance,
            "edge_type": "adjacency",
            
            # GSDiff-specific attributes
            "feature_vector": _create_edge_feature_vector(edge, room1_pos, room2_pos),
            "weight": 1.0 / (1.0 + distance),  # Inverse distance weight
            "connectivity_strength": _calculate_connectivity_strength(edge, G.nodes[room1_id], G.nodes[room2_id])
        }
        
        G.add_edge(room1_id, room2_id, **edge_attrs)
    
    # Add global graph attributes
    G.graph.update({
        "design_intent": json_data.get("global_constraints", []),
        "metadata": json_data.get("metadata", {}),
        "total_rooms": len(json_data["rooms"]),
        "total_adjacencies": len(json_data["adjacency_edges"]),
        "graph_type": "architectural_design"
    })
    
    return G


def _create_feature_vector(room: Dict[str, Any]) -> np.ndarray:
    """
    Create a feature vector for a room node suitable for GSDiff.
    
    Args:
        room: Room dictionary from the JSON data
        
    Returns:
        np.ndarray: Feature vector with normalized values
    """
    # Extract basic features
    width = room["dimensions"]["width"]
    length = room["dimensions"]["length"]
    height = room["dimensions"].get("height", 2.7)
    
    # Normalize dimensions (assuming typical room sizes)
    norm_width = min(width / 10.0, 1.0)  # Normalize to 0-1, cap at 10m
    norm_length = min(length / 10.0, 1.0)
    norm_height = min(height / 4.0, 1.0)  # Normalize to 0-1, cap at 4m
    
    # Room type encoding (one-hot like)
    room_type = room["type"]
    type_features = {
        "bedroom": [1, 0, 0, 0, 0],
        "kitchen": [0, 1, 0, 0, 0],
        "living": [0, 0, 1, 0, 0],
        "bathroom": [0, 0, 0, 1, 0],
        "dining": [0, 0, 0, 0, 1]
    }
    type_vector = type_features.get(room_type, [0, 0, 0, 0, 0])
    
    # Constraints count
    constraints_count = len(room.get("constraints", []))
    norm_constraints = min(constraints_count / 5.0, 1.0)  # Normalize to 0-1, cap at 5
    
    # Combine all features
    feature_vector = np.array([
        norm_width, norm_length, norm_height,
        norm_constraints,
        *type_vector
    ])
    
    return feature_vector


def _create_edge_feature_vector(edge: Dict[str, Any], pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Create a feature vector for an edge suitable for GSDiff.
    
    Args:
        edge: Edge dictionary from the JSON data
        pos1: Position of first room
        pos2: Position of second room
        
    Returns:
        np.ndarray: Feature vector for the edge
    """
    # Distance features
    distance = np.linalg.norm(pos1 - pos2)
    norm_distance = min(distance / 20.0, 1.0)  # Normalize to 0-1, cap at 20m
    
    # Relationship type encoding
    relationship_type = edge["relationship_type"]
    relationship_features = {
        "adjacent": [1, 0, 0],
        "connected": [0, 1, 0],
        "near": [0, 0, 1]
    }
    rel_vector = relationship_features.get(relationship_type, [0, 0, 0])
    
    # Edge distance from JSON (if specified)
    edge_distance = edge.get("distance", distance)
    norm_edge_distance = min(edge_distance / 20.0, 1.0)
    
    # Combine features
    feature_vector = np.array([
        norm_distance,
        norm_edge_distance,
        *rel_vector
    ])
    
    return feature_vector


def _create_bounding_box(x: float, y: float, z: float, width: float, length: float, height: float) -> Dict[str, np.ndarray]:
    """
    Create a bounding box representation for a room.
    
    Args:
        x, y, z: Room position coordinates
        width, length, height: Room dimensions
        
    Returns:
        Dict containing min and max points of the bounding box
    """
    min_point = np.array([x, y, z])
    max_point = np.array([x + width, y + length, z + height])
    
    return {
        "min": min_point,
        "max": max_point,
        "center": (min_point + max_point) / 2,
        "extents": np.array([width, length, height])
    }


def _calculate_connectivity_strength(edge: Dict[str, Any], room1_attrs: Dict[str, Any], room2_attrs: Dict[str, Any]) -> float:
    """
    Calculate the connectivity strength between two rooms.
    
    Args:
        edge: Edge dictionary
        room1_attrs: Attributes of first room
        room2_attrs: Attributes of second room
        
    Returns:
        float: Connectivity strength value between 0 and 1
    """
    # Base strength from relationship type
    base_strength = {
        "adjacent": 1.0,
        "connected": 0.8,
        "near": 0.6
    }.get(edge["relationship_type"], 0.5)
    
    # Adjust based on room types (some combinations are more natural)
    room1_type = room1_attrs["room_type"]
    room2_type = room2_attrs["room_type"]
    
    # Natural adjacencies get bonus
    natural_adjacencies = [
        ("kitchen", "dining"),
        ("living", "dining"),
        ("bedroom", "bathroom"),
        ("entrance", "living")
    ]
    
    if (room1_type, room2_type) in natural_adjacencies or (room2_type, room1_type) in natural_adjacencies:
        base_strength *= 1.2
    
    # Distance penalty
    distance = edge.get("distance", 0.0)
    if distance > 0:
        distance_penalty = 1.0 / (1.0 + distance)
        base_strength *= distance_penalty
    
    return min(base_strength, 1.0)


def get_graph_summary(G: nx.Graph) -> Dict[str, Any]:
    """
    Get a summary of the graph structure and properties.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        Dict containing graph summary information
    """
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "room_types": list(set(nx.get_node_attributes(G, "room_type").values())),
        "total_area": sum(nx.get_node_attributes(G, "area").values()),
        "total_volume": sum(nx.get_node_attributes(G, "volume").values()),
        "relationship_types": list(set(nx.get_edge_attributes(G, "relationship_type").values())),
        "graph_density": nx.density(G),
        "is_connected": nx.is_connected(G),
        "num_components": nx.number_connected_components(G)
    }
