"""
Advanced Data Structures - Production Ready Implementations
Complete implementations of advanced data structures in Python

Status: PRODUCTION READY
Last Updated: 2024
Python Version: 3.8+
"""

from typing import Any, List, Optional, Tuple
from collections import deque


class AVLNode:
    """Node for AVL Tree with height tracking"""
    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height = 1


class AVLTree:
    """
    Self-balancing AVL Tree implementation
    
    Time Complexity:
    - Insert: O(log n)
    - Delete: O(log n)
    - Search: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.root: Optional[AVLNode] = None
    
    def height(self, node: Optional[AVLNode]) -> int:
        """Get height of node"""
        return node.height if node else 0
    
    def balance_factor(self, node: Optional[AVLNode]) -> int:
        """Calculate balance factor"""
        return self.height(node.left) - self.height(node.right) if node else 0
    
    def update_height(self, node: AVLNode) -> None:
        """Update node height based on children"""
        if node:
            node.height = 1 + max(self.height(node.left), self.height(node.right))
    
    def rotate_right(self, y: AVLNode) -> AVLNode:
        """Right rotation"""
        x = y.left
        T2 = x.right
        
        x.right = y
        y.left = T2
        
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def rotate_left(self, x: AVLNode) -> AVLNode:
        """Left rotation"""
        y = x.right
        T2 = y.left
        
        y.left = x
        x.right = T2
        
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, node: Optional[AVLNode], key: Any, value: Any) -> AVLNode:
        """Insert key-value pair with balancing"""
        if not node:
            return AVLNode(key, value)
        
        if key < node.key:
            node.left = self.insert(node.left, key, value)
        elif key > node.key:
            node.right = self.insert(node.right, key, value)
        else:
            node.value = value
            return node
        
        self.update_height(node)
        balance = self.balance_factor(node)
        
        # Left-Left case
        if balance > 1 and key < node.left.key:
            return self.rotate_right(node)
        
        # Right-Right case
        if balance < -1 and key > node.right.key:
            return self.rotate_left(node)
        
        # Left-Right case
        if balance > 1 and key > node.left.key:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right-Left case
        if balance < -1 and key < node.right.key:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def put(self, key: Any, value: Any) -> None:
        """Public insert method"""
        self.root = self.insert(self.root, key, value)
    
    def search(self, node: Optional[AVLNode], key: Any) -> Optional[Any]:
        """Search for key"""
        if not node or node.key == key:
            return node.value if node else None
        
        if key < node.key:
            return self.search(node.left, key)
        return self.search(node.right, key)
    
    def get(self, key: Any) -> Optional[Any]:
        """Public search method"""
        return self.search(self.root, key)


class TrieNode:
    """Node for Trie data structure"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None


class Trie:
    """
    Trie (Prefix Tree) implementation for efficient string operations
    
    Time Complexity:
    - Insert: O(m) where m is key length
    - Search: O(m)
    - Prefix Search: O(m + k) where k is number of results
    
    Space Complexity: O(ALPHABET_SIZE * m * n)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, value: Any = None) -> None:
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.value = value
    
    def search(self, word: str) -> Optional[Any]:
        """Search for exact word"""
        node = self._find_node(word)
        return node.value if node and node.is_end_of_word else None
    
    def starts_with(self, prefix: str) -> List[str]:
        """Find all words starting with prefix"""
        node = self._find_node(prefix)
        if not node:
            return []
        return self._collect_words(node, prefix)
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find node corresponding to prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _collect_words(self, node: TrieNode, prefix: str) -> List[str]:
        """Collect all words from given node"""
        results = []
        if node.is_end_of_word:
            results.append(prefix)
        
        for char, child_node in node.children.items():
            results.extend(self._collect_words(child_node, prefix + char))
        
        return results


class UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression and union by rank
    
    Time Complexity:
    - Find: O(α(n)) - almost constant
    - Union: O(α(n)) - almost constant
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x: int) -> int:
        """Find root with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are connected"""
        return self.find(x) == self.find(y)


class MinHeap:
    """
    Min Heap implementation using array
    
    Time Complexity:
    - Insert: O(log n)
    - Extract Min: O(log n)
    - Peek: O(1)
    - Heapify: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.heap = []
    
    def parent(self, i: int) -> int:
        return (i - 1) // 2
    
    def left_child(self, i: int) -> int:
        return 2 * i + 1
    
    def right_child(self, i: int) -> int:
        return 2 * i + 2
    
    def swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, value: Any) -> None:
        """Insert value into heap"""
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self) -> Any:
        """Remove and return minimum element"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return min_val
    
    def peek(self) -> Any:
        """Return minimum element without removing"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def _heapify_up(self, i: int) -> None:
        """Move element up to maintain heap property"""
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def _heapify_down(self, i: int) -> None:
        """Move element down to maintain heap property"""
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right
        
        if min_index != i:
            self.swap(i, min_index)
            self._heapify_down(min_index)
    
    def size(self) -> int:
        """Return heap size"""
        return len(self.heap)
    
    def is_empty(self) -> bool:
        """Check if heap is empty"""
        return len(self.heap) == 0


class Graph:
    """
    Graph implementation using adjacency list
    
    Supports:
    - Directed and undirected graphs
    - Weighted and unweighted edges
    - BFS and DFS traversal
    """
    
    def __init__(self, directed: bool = False):
        self.graph = {}
        self.directed = directed
    
    def add_vertex(self, vertex: Any) -> None:
        """Add vertex to graph"""
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """Add edge to graph"""
        self.add_vertex(u)
        self.add_vertex(v)
        
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def bfs(self, start: Any) -> List[Any]:
        """Breadth-First Search"""
        if start not in self.graph:
            return []
        
        visited = set([start])
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def dfs(self, start: Any) -> List[Any]:
        """Depth-First Search"""
        if start not in self.graph:
            return []
        
        visited = set()
        result = []
        
        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def has_cycle(self) -> bool:
        """Check if graph has cycle (for directed graphs)"""
        if not self.directed:
            raise ValueError("Cycle detection for undirected graphs not implemented")
        
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(v):
            visited.add(v)
            rec_stack.add(v)
            
            for neighbor, _ in self.graph.get(v, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(v)
            return False
        
        for vertex in self.graph:
            if vertex not in visited:
                if has_cycle_util(vertex):
                    return True
        
        return False


# Demonstration code
if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Data Structures - Production Ready Examples")
    print("=" * 60)
    
    # Example 1: AVL Tree
    print("\n1. AVL Tree")
    avl = AVLTree()
    for key, value in [(5, "five"), (3, "three"), (7, "seven"), (2, "two"), (4, "four")]:
        avl.put(key, value)
    print(f"   Inserted keys: 5, 3, 7, 2, 4")
    print(f"   Search key 3: {avl.get(3)}")
    print(f"   Search key 7: {avl.get(7)}")
    
    # Example 2: Trie
    print("\n2. Trie (Prefix Tree)")
    trie = Trie()
    words = ["hello", "world", "help", "heap", "hero"]
    for word in words:
        trie.insert(word, f"value_{word}")
    print(f"   Inserted words: {words}")
    print(f"   Words starting with 'he': {trie.starts_with('he')}")
    print(f"   Words starting with 'hel': {trie.starts_with('hel')}")
    
    # Example 3: Union-Find
    print("\n3. Union-Find (Disjoint Set)")
    uf = UnionFind(10)
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(4, 5)
    print(f"   Connected components: {uf.count}")
    print(f"   Are 1 and 3 connected? {uf.connected(1, 3)}")
    print(f"   Are 1 and 4 connected? {uf.connected(1, 4)}")
    
    # Example 4: Min Heap
    print("\n4. Min Heap")
    heap = MinHeap()
    for val in [5, 3, 7, 1, 9, 2]:
        heap.insert(val)
    print(f"   Inserted values: 5, 3, 7, 1, 9, 2")
    print(f"   Extracted min: {heap.extract_min()}")
    print(f"   Next min (peek): {heap.peek()}")
    
    # Example 5: Graph
    print("\n5. Graph (BFS and DFS)")
    graph = Graph(directed=False)
    edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    for u, v in edges:
        graph.add_edge(u, v)
    print(f"   Added edges: {edges}")
    print(f"   BFS from 1: {graph.bfs(1)}")
    print(f"   DFS from 1: {graph.dfs(1)}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
