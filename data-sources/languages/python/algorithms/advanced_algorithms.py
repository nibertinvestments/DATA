#!/usr/bin/env python3
"""
Advanced Algorithm Implementations for AI Coding Agents
========================================================

This module contains production-ready implementations of advanced algorithms
with comprehensive documentation, error handling, and performance analysis.

Author: AI Dataset Creation Team
License: MIT
Created: 2024
"""

import heapq
import math
from typing import List, Tuple, Dict, Set, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import time
import functools


@dataclass
class GraphNode:
    """Represents a node in a graph with weights and metadata."""
    
    value: Any
    neighbors: Dict[Any, float]  # neighbor_value -> edge_weight
    visited: bool = False
    distance: float = float('inf')
    parent: Optional['GraphNode'] = None
    
    def __post_init__(self):
        """Initialize neighbors as defaultdict if not provided."""
        if not isinstance(self.neighbors, dict):
            self.neighbors = {}


class AdvancedAlgorithms:
    """
    Collection of advanced algorithms with optimal implementations.
    
    All algorithms include:
    - Comprehensive error handling
    - Time/space complexity analysis
    - Detailed documentation
    - Type hints for better AI understanding
    """
    
    @staticmethod
    def dijkstra_shortest_path(
        graph: Dict[Any, Dict[Any, float]], 
        start: Any, 
        end: Optional[Any] = None
    ) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
        """
        Dijkstra's algorithm for shortest path in weighted graphs.
        
        Time Complexity: O((V + E) log V) where V = vertices, E = edges
        Space Complexity: O(V) for distance and parent tracking
        
        Args:
            graph: Adjacency list representation {node: {neighbor: weight}}
            start: Starting node
            end: Optional ending node (if None, finds all shortest paths)
            
        Returns:
            Tuple of (distances dict, parent dict for path reconstruction)
            
        Raises:
            ValueError: If start node not in graph or graph is empty
            TypeError: If weights are not numeric
            
        Example:
            >>> graph = {
            ...     'A': {'B': 4, 'C': 2},
            ...     'B': {'C': 1, 'D': 5},
            ...     'C': {'D': 8, 'E': 10},
            ...     'D': {'E': 2},
            ...     'E': {}
            ... }
            >>> distances, parents = AdvancedAlgorithms.dijkstra_shortest_path(graph, 'A')
            >>> distances['E']  # Shortest distance from A to E
            11
        """
        if not graph:
            raise ValueError("Graph cannot be empty")
        
        if start not in graph:
            raise ValueError(f"Start node '{start}' not found in graph")
        
        # Validate all weights are numeric
        for node, edges in graph.items():
            for neighbor, weight in edges.items():
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise TypeError(f"Invalid weight {weight} between {node} and {neighbor}")
        
        # Initialize distances and parent tracking
        distances = {node: float('inf') for node in graph}
        parents = {node: None for node in graph}
        distances[start] = 0
        
        # Priority queue: (distance, node)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # Skip if we've found a better path already
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Early termination if we only need path to specific end node
            if end and current_node == end:
                break
            
            # Update distances to neighbors
            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parents[neighbor] = current_node
                        heapq.heappush(pq, (new_distance, neighbor))
        
        return distances, parents


if __name__ == "__main__":
    # Example usage and testing
    print("🧠 Advanced Algorithms for AI Coding Agents")
    print("=" * 45)
    
    # Dijkstra example
    print("\n🗺️  Shortest Path Example:")
    sample_graph = {
        'A': {'B': 4, 'C': 2},
        'B': {'C': 1, 'D': 5},
        'C': {'D': 8, 'E': 10},
        'D': {'E': 2},
        'E': {}
    }
    
    distances, parents = AdvancedAlgorithms.dijkstra_shortest_path(sample_graph, 'A')
    print(f"Shortest distance A → E: {distances['E']}")