// Advanced Graph Algorithms in Kotlin
// Comprehensive graph theory implementations

data class Graph(val vertices: Int) {
    private val adjacencyList = Array(vertices) { mutableListOf<Pair<Int, Int>>() }
    
    fun addEdge(from: Int, to: Int, weight: Int = 1) {
        adjacencyList[from].add(Pair(to, weight))
    }
    
    fun getNeighbors(vertex: Int): List<Pair<Int, Int>> = adjacencyList[vertex]
}

// Dijkstra's Shortest Path Algorithm
class DijkstraShortestPath {
    fun findShortestPath(graph: Graph, start: Int, vertices: Int): IntArray {
        val distances = IntArray(vertices) { Int.MAX_VALUE }
        val visited = BooleanArray(vertices)
        distances[start] = 0
        
        repeat(vertices) {
            val u = findMinDistance(distances, visited)
            if (u == -1) return@repeat
            
            visited[u] = true
            
            for ((v, weight) in graph.getNeighbors(u)) {
                if (!visited[v] && distances[u] != Int.MAX_VALUE) {
                    val newDist = distances[u] + weight
                    if (newDist < distances[v]) {
                        distances[v] = newDist
                    }
                }
            }
        }
        
        return distances
    }
    
    private fun findMinDistance(distances: IntArray, visited: BooleanArray): Int {
        var minDist = Int.MAX_VALUE
        var minIndex = -1
        
        for (i in distances.indices) {
            if (!visited[i] && distances[i] < minDist) {
                minDist = distances[i]
                minIndex = i
            }
        }
        
        return minIndex
    }
}

// Bellman-Ford Algorithm
class BellmanFordAlgorithm {
    fun findShortestPaths(graph: Graph, start: Int, vertices: Int, edges: List<Triple<Int, Int, Int>>): IntArray? {
        val distances = IntArray(vertices) { Int.MAX_VALUE }
        distances[start] = 0
        
        // Relax edges
        repeat(vertices - 1) {
            for ((u, v, weight) in edges) {
                if (distances[u] != Int.MAX_VALUE && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight
                }
            }
        }
        
        // Check for negative cycles
        for ((u, v, weight) in edges) {
            if (distances[u] != Int.MAX_VALUE && distances[u] + weight < distances[v]) {
                return null // Negative cycle detected
            }
        }
        
        return distances
    }
}

// Floyd-Warshall All-Pairs Shortest Path
class FloydWarshallAlgorithm {
    fun allPairsShortestPath(vertices: Int, edges: List<Triple<Int, Int, Int>>): Array<IntArray> {
        val dist = Array(vertices) { IntArray(vertices) { Int.MAX_VALUE / 2 } }
        
        // Initialize diagonal
        for (i in 0 until vertices) {
            dist[i][i] = 0
        }
        
        // Add edges
        for ((u, v, weight) in edges) {
            dist[u][v] = weight
        }
        
        // Floyd-Warshall algorithm
        for (k in 0 until vertices) {
            for (i in 0 until vertices) {
                for (j in 0 until vertices) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j]
                    }
                }
            }
        }
        
        return dist
    }
}

// Topological Sort
class TopologicalSort {
    fun sort(graph: Graph, vertices: Int): List<Int>? {
        val inDegree = IntArray(vertices)
        val result = mutableListOf<Int>()
        
        // Calculate in-degrees
        for (u in 0 until vertices) {
            for ((v, _) in graph.getNeighbors(u)) {
                inDegree[v]++
            }
        }
        
        // Queue for vertices with in-degree 0
        val queue = ArrayDeque<Int>()
        for (i in inDegree.indices) {
            if (inDegree[i] == 0) {
                queue.add(i)
            }
        }
        
        // Process vertices
        while (queue.isNotEmpty()) {
            val u = queue.removeFirst()
            result.add(u)
            
            for ((v, _) in graph.getNeighbors(u)) {
                inDegree[v]--
                if (inDegree[v] == 0) {
                    queue.add(v)
                }
            }
        }
        
        return if (result.size == vertices) result else null // Cycle detected
    }
}

fun main() {
    println("Advanced Graph Algorithms in Kotlin")
    println("====================================")
    
    // Create sample graph
    val graph = Graph(5)
    graph.addEdge(0, 1, 4)
    graph.addEdge(0, 2, 1)
    graph.addEdge(2, 1, 2)
    graph.addEdge(1, 3, 1)
    graph.addEdge(2, 3, 5)
    graph.addEdge(3, 4, 3)
    
    // Test Dijkstra
    val dijkstra = DijkstraShortestPath()
    val distances = dijkstra.findShortestPath(graph, 0, 5)
    println("Shortest paths from vertex 0: ${distances.contentToString()}")
}
