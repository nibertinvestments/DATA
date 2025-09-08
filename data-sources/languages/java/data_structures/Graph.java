/**
 * Advanced Graph Implementation with Multiple Algorithms
 * 
 * This class provides a comprehensive graph implementation supporting both
 * directed and undirected graphs, with various algorithms including:
 * - Dijkstra's shortest path
 * - Floyd-Warshall all-pairs shortest path
 * - Breadth-First Search (BFS)
 * - Depth-First Search (DFS)
 * - Topological Sort
 * - Cycle detection
 * 
 * @author AI Training Dataset
 * @version 1.0
 */

import java.util.*;

public class Graph<T> {
    
    /**
     * Internal representation of graph vertices and edges
     */
    private Map<T, Map<T, Double>> adjacencyList;
    private boolean isDirected;
    private int vertexCount;
    
    /**
     * Edge class for representing weighted edges
     */
    public static class Edge<T> {
        T source, destination;
        double weight;
        
        public Edge(T source, T destination, double weight) {
            this.source = source;
            this.destination = destination;
            this.weight = weight;
        }
        
        @Override
        public String toString() {
            return String.format("(%s -> %s, weight: %.2f)", source, destination, weight);
        }
    }
    
    /**
     * Result class for shortest path algorithms
     */
    public static class PathResult<T> {
        Map<T, Double> distances;
        Map<T, T> predecessors;
        
        public PathResult(Map<T, Double> distances, Map<T, T> predecessors) {
            this.distances = distances;
            this.predecessors = predecessors;
        }
        
        public List<T> getPath(T destination) {
            List<T> path = new ArrayList<>();
            T current = destination;
            
            while (current != null) {
                path.add(0, current);
                current = predecessors.get(current);
            }
            
            return path.isEmpty() || !path.get(0).equals(predecessors.keySet().iterator().next()) 
                   ? new ArrayList<>() : path;
        }
    }
    
    /**
     * Constructor
     * @param isDirected Whether the graph is directed
     */
    public Graph(boolean isDirected) {
        this.adjacencyList = new HashMap<>();
        this.isDirected = isDirected;
        this.vertexCount = 0;
    }
    
    /**
     * Add a vertex to the graph
     * @param vertex The vertex to add
     */
    public void addVertex(T vertex) {
        if (!adjacencyList.containsKey(vertex)) {
            adjacencyList.put(vertex, new HashMap<>());
            vertexCount++;
        }
    }
    
    /**
     * Add an edge to the graph
     * @param source Source vertex
     * @param destination Destination vertex
     * @param weight Edge weight
     */
    public void addEdge(T source, T destination, double weight) {
        addVertex(source);
        addVertex(destination);
        
        adjacencyList.get(source).put(destination, weight);
        
        if (!isDirected) {
            adjacencyList.get(destination).put(source, weight);
        }
    }
    
    /**
     * Add an unweighted edge (weight = 1.0)
     * @param source Source vertex
     * @param destination Destination vertex
     */
    public void addEdge(T source, T destination) {
        addEdge(source, destination, 1.0);
    }
    
    /**
     * Remove an edge from the graph
     * @param source Source vertex
     * @param destination Destination vertex
     */
    public void removeEdge(T source, T destination) {
        if (adjacencyList.containsKey(source)) {
            adjacencyList.get(source).remove(destination);
        }
        
        if (!isDirected && adjacencyList.containsKey(destination)) {
            adjacencyList.get(destination).remove(source);
        }
    }
    
    /**
     * Check if there's an edge between two vertices
     * @param source Source vertex
     * @param destination Destination vertex
     * @return true if edge exists, false otherwise
     */
    public boolean hasEdge(T source, T destination) {
        return adjacencyList.containsKey(source) && 
               adjacencyList.get(source).containsKey(destination);
    }
    
    /**
     * Get all vertices in the graph
     * @return Set of all vertices
     */
    public Set<T> getVertices() {
        return new HashSet<>(adjacencyList.keySet());
    }
    
    /**
     * Get neighbors of a vertex
     * @param vertex The vertex
     * @return Set of neighboring vertices
     */
    public Set<T> getNeighbors(T vertex) {
        return adjacencyList.getOrDefault(vertex, new HashMap<>()).keySet();
    }
    
    /**
     * Get edge weight between two vertices
     * @param source Source vertex
     * @param destination Destination vertex
     * @return Edge weight, or Double.POSITIVE_INFINITY if no edge exists
     */
    public double getEdgeWeight(T source, T destination) {
        if (hasEdge(source, destination)) {
            return adjacencyList.get(source).get(destination);
        }
        return Double.POSITIVE_INFINITY;
    }
    
    /**
     * Breadth-First Search traversal
     * @param start Starting vertex
     * @return List of vertices in BFS order
     */
    public List<T> breadthFirstSearch(T start) {
        List<T> result = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        Queue<T> queue = new LinkedList<>();
        
        queue.offer(start);
        visited.add(start);
        
        while (!queue.isEmpty()) {
            T current = queue.poll();
            result.add(current);
            
            for (T neighbor : getNeighbors(current)) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Depth-First Search traversal
     * @param start Starting vertex
     * @return List of vertices in DFS order
     */
    public List<T> depthFirstSearch(T start) {
        List<T> result = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        dfsHelper(start, visited, result);
        return result;
    }
    
    /**
     * Recursive helper for DFS
     */
    private void dfsHelper(T vertex, Set<T> visited, List<T> result) {
        visited.add(vertex);
        result.add(vertex);
        
        for (T neighbor : getNeighbors(vertex)) {
            if (!visited.contains(neighbor)) {
                dfsHelper(neighbor, visited, result);
            }
        }
    }
    
    /**
     * Dijkstra's shortest path algorithm
     * @param source Source vertex
     * @return PathResult containing distances and predecessors
     */
    public PathResult<T> dijkstra(T source) {
        Map<T, Double> distances = new HashMap<>();
        Map<T, T> predecessors = new HashMap<>();
        PriorityQueue<T> pq = new PriorityQueue<>(
            Comparator.comparing(distances::get)
        );
        
        // Initialize distances
        for (T vertex : getVertices()) {
            distances.put(vertex, Double.POSITIVE_INFINITY);
        }
        distances.put(source, 0.0);
        pq.offer(source);
        
        while (!pq.isEmpty()) {
            T current = pq.poll();
            double currentDistance = distances.get(current);
            
            for (T neighbor : getNeighbors(current)) {
                double edgeWeight = getEdgeWeight(current, neighbor);
                double newDistance = currentDistance + edgeWeight;
                
                if (newDistance < distances.get(neighbor)) {
                    distances.put(neighbor, newDistance);
                    predecessors.put(neighbor, current);
                    pq.offer(neighbor);
                }
            }
        }
        
        return new PathResult<>(distances, predecessors);
    }
    
    /**
     * Floyd-Warshall all-pairs shortest path algorithm
     * @return 2D map of distances between all pairs of vertices
     */
    public Map<T, Map<T, Double>> floydWarshall() {
        Map<T, Map<T, Double>> distances = new HashMap<>();
        List<T> vertices = new ArrayList<>(getVertices());
        
        // Initialize distances
        for (T i : vertices) {
            distances.put(i, new HashMap<>());
            for (T j : vertices) {
                if (i.equals(j)) {
                    distances.get(i).put(j, 0.0);
                } else if (hasEdge(i, j)) {
                    distances.get(i).put(j, getEdgeWeight(i, j));
                } else {
                    distances.get(i).put(j, Double.POSITIVE_INFINITY);
                }
            }
        }
        
        // Floyd-Warshall algorithm
        for (T k : vertices) {
            for (T i : vertices) {
                for (T j : vertices) {
                    double currentDistance = distances.get(i).get(j);
                    double newDistance = distances.get(i).get(k) + distances.get(k).get(j);
                    
                    if (newDistance < currentDistance) {
                        distances.get(i).put(j, newDistance);
                    }
                }
            }
        }
        
        return distances;
    }
    
    /**
     * Topological sort (only for directed acyclic graphs)
     * @return List of vertices in topological order, or empty list if cycle exists
     */
    public List<T> topologicalSort() {
        if (!isDirected) {
            throw new IllegalStateException("Topological sort only applies to directed graphs");
        }
        
        Map<T, Integer> inDegree = new HashMap<>();
        
        // Calculate in-degrees
        for (T vertex : getVertices()) {
            inDegree.put(vertex, 0);
        }
        
        for (T vertex : getVertices()) {
            for (T neighbor : getNeighbors(vertex)) {
                inDegree.put(neighbor, inDegree.get(neighbor) + 1);
            }
        }
        
        // Use queue for vertices with in-degree 0
        Queue<T> queue = new LinkedList<>();
        for (T vertex : getVertices()) {
            if (inDegree.get(vertex) == 0) {
                queue.offer(vertex);
            }
        }
        
        List<T> result = new ArrayList<>();
        
        while (!queue.isEmpty()) {
            T current = queue.poll();
            result.add(current);
            
            for (T neighbor : getNeighbors(current)) {
                inDegree.put(neighbor, inDegree.get(neighbor) - 1);
                if (inDegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        
        // Check if all vertices are included (no cycle)
        if (result.size() != vertexCount) {
            return new ArrayList<>(); // Cycle detected
        }
        
        return result;
    }
    
    /**
     * Detect if the graph has a cycle
     * @return true if cycle exists, false otherwise
     */
    public boolean hasCycle() {
        if (isDirected) {
            return hasDirectedCycle();
        } else {
            return hasUndirectedCycle();
        }
    }
    
    /**
     * Detect cycle in directed graph using DFS
     */
    private boolean hasDirectedCycle() {
        Set<T> visited = new HashSet<>();
        Set<T> recursionStack = new HashSet<>();
        
        for (T vertex : getVertices()) {
            if (!visited.contains(vertex)) {
                if (dfsDirectedCycle(vertex, visited, recursionStack)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    /**
     * Helper method for directed cycle detection
     */
    private boolean dfsDirectedCycle(T vertex, Set<T> visited, Set<T> recursionStack) {
        visited.add(vertex);
        recursionStack.add(vertex);
        
        for (T neighbor : getNeighbors(vertex)) {
            if (!visited.contains(neighbor)) {
                if (dfsDirectedCycle(neighbor, visited, recursionStack)) {
                    return true;
                }
            } else if (recursionStack.contains(neighbor)) {
                return true;
            }
        }
        
        recursionStack.remove(vertex);
        return false;
    }
    
    /**
     * Detect cycle in undirected graph using DFS
     */
    private boolean hasUndirectedCycle() {
        Set<T> visited = new HashSet<>();
        
        for (T vertex : getVertices()) {
            if (!visited.contains(vertex)) {
                if (dfsUndirectedCycle(vertex, null, visited)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    /**
     * Helper method for undirected cycle detection
     */
    private boolean dfsUndirectedCycle(T vertex, T parent, Set<T> visited) {
        visited.add(vertex);
        
        for (T neighbor : getNeighbors(vertex)) {
            if (!visited.contains(neighbor)) {
                if (dfsUndirectedCycle(neighbor, vertex, visited)) {
                    return true;
                }
            } else if (!neighbor.equals(parent)) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Get the number of vertices
     * @return Vertex count
     */
    public int getVertexCount() {
        return vertexCount;
    }
    
    /**
     * Get the number of edges
     * @return Edge count
     */
    public int getEdgeCount() {
        int count = 0;
        for (Map<T, Double> neighbors : adjacencyList.values()) {
            count += neighbors.size();
        }
        return isDirected ? count : count / 2;
    }
    
    /**
     * Get string representation of the graph
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Graph (").append(isDirected ? "Directed" : "Undirected").append("):\n");
        
        for (T vertex : getVertices()) {
            sb.append(vertex).append(" -> ");
            boolean first = true;
            for (Map.Entry<T, Double> entry : adjacencyList.get(vertex).entrySet()) {
                if (!first) sb.append(", ");
                sb.append(entry.getKey()).append("(").append(entry.getValue()).append(")");
                first = false;
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Demo method showing graph algorithms
     */
    public static void main(String[] args) {
        // Create a directed graph
        Graph<String> graph = new Graph<>(true);
        
        // Add vertices and edges
        graph.addEdge("A", "B", 4);
        graph.addEdge("A", "C", 2);
        graph.addEdge("B", "C", 1);
        graph.addEdge("B", "D", 5);
        graph.addEdge("C", "D", 8);
        graph.addEdge("C", "E", 10);
        graph.addEdge("D", "E", 2);
        
        System.out.println(graph);
        
        // BFS and DFS
        System.out.println("BFS from A: " + graph.breadthFirstSearch("A"));
        System.out.println("DFS from A: " + graph.depthFirstSearch("A"));
        
        // Shortest paths
        PathResult<String> result = graph.dijkstra("A");
        System.out.println("Shortest distances from A:");
        for (Map.Entry<String, Double> entry : result.distances.entrySet()) {
            System.out.println("  To " + entry.getKey() + ": " + entry.getValue());
            List<String> path = result.getPath(entry.getKey());
            if (!path.isEmpty()) {
                System.out.println("    Path: " + path);
            }
        }
        
        // Cycle detection
        System.out.println("Has cycle: " + graph.hasCycle());
        
        // Topological sort
        List<String> topoSort = graph.topologicalSort();
        if (!topoSort.isEmpty()) {
            System.out.println("Topological sort: " + topoSort);
        } else {
            System.out.println("Graph has a cycle, cannot perform topological sort");
        }
    }
}