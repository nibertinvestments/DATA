// Comprehensive data structures implementation in Java

import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.lang.reflect.Array;

/**
 * Advanced data structures implementations in Java
 * Covers trees, heaps, graphs, hash tables, and more
 */

// Binary Search Tree implementation
class BinarySearchTree<T extends Comparable<T>> {
    private Node<T> root;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> left, right;
        int height; // For AVL tree balancing
        
        Node(T data) {
            this.data = data;
            this.height = 1;
        }
    }
    
    public void insert(T data) {
        root = insertRec(root, data);
        size++;
    }
    
    private Node<T> insertRec(Node<T> root, T data) {
        if (root == null) {
            return new Node<>(data);
        }
        
        int cmp = data.compareTo(root.data);
        if (cmp < 0) {
            root.left = insertRec(root.left, data);
        } else if (cmp > 0) {
            root.right = insertRec(root.right, data);
        }
        
        // Update height and balance (AVL tree logic)
        updateHeight(root);
        return balance(root);
    }
    
    public boolean search(T data) {
        return searchRec(root, data);
    }
    
    private boolean searchRec(Node<T> root, T data) {
        if (root == null) return false;
        
        int cmp = data.compareTo(root.data);
        if (cmp == 0) return true;
        if (cmp < 0) return searchRec(root.left, data);
        return searchRec(root.right, data);
    }
    
    public void delete(T data) {
        root = deleteRec(root, data);
        size--;
    }
    
    private Node<T> deleteRec(Node<T> root, T data) {
        if (root == null) return null;
        
        int cmp = data.compareTo(root.data);
        if (cmp < 0) {
            root.left = deleteRec(root.left, data);
        } else if (cmp > 0) {
            root.right = deleteRec(root.right, data);
        } else {
            // Node to delete found
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            
            // Node with two children
            Node<T> minRight = findMin(root.right);
            root.data = minRight.data;
            root.right = deleteRec(root.right, minRight.data);
        }
        
        updateHeight(root);
        return balance(root);
    }
    
    private Node<T> findMin(Node<T> root) {
        while (root.left != null) {
            root = root.left;
        }
        return root;
    }
    
    // AVL tree balancing methods
    private void updateHeight(Node<T> node) {
        if (node != null) {
            node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));
        }
    }
    
    private int getHeight(Node<T> node) {
        return node == null ? 0 : node.height;
    }
    
    private int getBalance(Node<T> node) {
        return node == null ? 0 : getHeight(node.left) - getHeight(node.right);
    }
    
    private Node<T> rotateRight(Node<T> y) {
        Node<T> x = y.left;
        Node<T> T2 = x.right;
        
        x.right = y;
        y.left = T2;
        
        updateHeight(y);
        updateHeight(x);
        
        return x;
    }
    
    private Node<T> rotateLeft(Node<T> x) {
        Node<T> y = x.right;
        Node<T> T2 = y.left;
        
        y.left = x;
        x.right = T2;
        
        updateHeight(x);
        updateHeight(y);
        
        return y;
    }
    
    private Node<T> balance(Node<T> node) {
        if (node == null) return null;
        
        int balance = getBalance(node);
        
        // Left-heavy
        if (balance > 1) {
            if (getBalance(node.left) < 0) {
                node.left = rotateLeft(node.left);
            }
            return rotateRight(node);
        }
        
        // Right-heavy
        if (balance < -1) {
            if (getBalance(node.right) > 0) {
                node.right = rotateRight(node.right);
            }
            return rotateLeft(node);
        }
        
        return node;
    }
    
    public List<T> inorderTraversal() {
        List<T> result = new ArrayList<>();
        inorderRec(root, result);
        return result;
    }
    
    private void inorderRec(Node<T> root, List<T> result) {
        if (root != null) {
            inorderRec(root.left, result);
            result.add(root.data);
            inorderRec(root.right, result);
        }
    }
    
    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }
}

// Generic Heap implementation
class Heap<T extends Comparable<T>> {
    private List<T> heap;
    private final boolean isMaxHeap;
    
    public Heap(boolean isMaxHeap) {
        this.heap = new ArrayList<>();
        this.isMaxHeap = isMaxHeap;
    }
    
    public void insert(T item) {
        heap.add(item);
        heapifyUp(heap.size() - 1);
    }
    
    public T extractTop() {
        if (heap.isEmpty()) {
            throw new NoSuchElementException("Heap is empty");
        }
        
        T top = heap.get(0);
        T lastItem = heap.remove(heap.size() - 1);
        
        if (!heap.isEmpty()) {
            heap.set(0, lastItem);
            heapifyDown(0);
        }
        
        return top;
    }
    
    public T peek() {
        if (heap.isEmpty()) {
            throw new NoSuchElementException("Heap is empty");
        }
        return heap.get(0);
    }
    
    private void heapifyUp(int index) {
        while (index > 0) {
            int parentIndex = (index - 1) / 2;
            if (!shouldSwap(index, parentIndex)) break;
            
            Collections.swap(heap, index, parentIndex);
            index = parentIndex;
        }
    }
    
    private void heapifyDown(int index) {
        while (true) {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            int targetIndex = index;
            
            if (leftChild < heap.size() && shouldSwap(leftChild, targetIndex)) {
                targetIndex = leftChild;
            }
            
            if (rightChild < heap.size() && shouldSwap(rightChild, targetIndex)) {
                targetIndex = rightChild;
            }
            
            if (targetIndex == index) break;
            
            Collections.swap(heap, index, targetIndex);
            index = targetIndex;
        }
    }
    
    private boolean shouldSwap(int childIndex, int parentIndex) {
        int comparison = heap.get(childIndex).compareTo(heap.get(parentIndex));
        return isMaxHeap ? comparison > 0 : comparison < 0;
    }
    
    public int size() { return heap.size(); }
    public boolean isEmpty() { return heap.isEmpty(); }
    
    public List<T> getSortedList() {
        List<T> original = new ArrayList<>(heap);
        List<T> sorted = new ArrayList<>();
        
        while (!isEmpty()) {
            sorted.add(extractTop());
        }
        
        // Restore original heap
        heap = original;
        for (int i = heap.size() / 2 - 1; i >= 0; i--) {
            heapifyDown(i);
        }
        
        return sorted;
    }
}

// Trie (Prefix Tree) implementation
class Trie {
    private TrieNode root;
    
    private static class TrieNode {
        Map<Character, TrieNode> children;
        boolean isEndOfWord;
        int wordCount;
        
        TrieNode() {
            children = new HashMap<>();
            isEndOfWord = false;
            wordCount = 0;
        }
    }
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode current = root;
        
        for (char ch : word.toCharArray()) {
            current.children.putIfAbsent(ch, new TrieNode());
            current = current.children.get(ch);
        }
        
        current.isEndOfWord = true;
        current.wordCount++;
    }
    
    public boolean search(String word) {
        TrieNode node = searchNode(word);
        return node != null && node.isEndOfWord;
    }
    
    public boolean startsWith(String prefix) {
        return searchNode(prefix) != null;
    }
    
    private TrieNode searchNode(String word) {
        TrieNode current = root;
        
        for (char ch : word.toCharArray()) {
            current = current.children.get(ch);
            if (current == null) return null;
        }
        
        return current;
    }
    
    public List<String> getAllWordsWithPrefix(String prefix) {
        List<String> words = new ArrayList<>();
        TrieNode prefixNode = searchNode(prefix);
        
        if (prefixNode != null) {
            collectWords(prefixNode, prefix, words);
        }
        
        return words;
    }
    
    private void collectWords(TrieNode node, String currentWord, List<String> words) {
        if (node.isEndOfWord) {
            words.add(currentWord);
        }
        
        for (Map.Entry<Character, TrieNode> entry : node.children.entrySet()) {
            collectWords(entry.getValue(), currentWord + entry.getKey(), words);
        }
    }
    
    public void delete(String word) {
        delete(root, word, 0);
    }
    
    private boolean delete(TrieNode current, String word, int index) {
        if (index == word.length()) {
            if (!current.isEndOfWord) return false;
            
            current.isEndOfWord = false;
            current.wordCount = 0;
            return current.children.isEmpty();
        }
        
        char ch = word.charAt(index);
        TrieNode node = current.children.get(ch);
        
        if (node == null) return false;
        
        boolean shouldDeleteChild = delete(node, word, index + 1);
        
        if (shouldDeleteChild) {
            current.children.remove(ch);
            return !current.isEndOfWord && current.children.isEmpty();
        }
        
        return false;
    }
}

// Graph implementation with adjacency list
class Graph<T> {
    private Map<T, List<Edge<T>>> adjacencyList;
    private boolean isDirected;
    
    private static class Edge<T> {
        T destination;
        double weight;
        
        Edge(T destination, double weight) {
            this.destination = destination;
            this.weight = weight;
        }
        
        @Override
        public String toString() {
            return String.format("->%s(%.1f)", destination, weight);
        }
    }
    
    public Graph(boolean isDirected) {
        this.adjacencyList = new HashMap<>();
        this.isDirected = isDirected;
    }
    
    public void addVertex(T vertex) {
        adjacencyList.putIfAbsent(vertex, new ArrayList<>());
    }
    
    public void addEdge(T source, T destination, double weight) {
        addVertex(source);
        addVertex(destination);
        
        adjacencyList.get(source).add(new Edge<>(destination, weight));
        
        if (!isDirected) {
            adjacencyList.get(destination).add(new Edge<>(source, weight));
        }
    }
    
    public void addEdge(T source, T destination) {
        addEdge(source, destination, 1.0);
    }
    
    public List<T> depthFirstSearch(T startVertex) {
        List<T> result = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        dfsHelper(startVertex, visited, result);
        return result;
    }
    
    private void dfsHelper(T vertex, Set<T> visited, List<T> result) {
        visited.add(vertex);
        result.add(vertex);
        
        List<Edge<T>> neighbors = adjacencyList.getOrDefault(vertex, new ArrayList<>());
        for (Edge<T> edge : neighbors) {
            if (!visited.contains(edge.destination)) {
                dfsHelper(edge.destination, visited, result);
            }
        }
    }
    
    public List<T> breadthFirstSearch(T startVertex) {
        List<T> result = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        Queue<T> queue = new LinkedList<>();
        
        queue.add(startVertex);
        visited.add(startVertex);
        
        while (!queue.isEmpty()) {
            T vertex = queue.poll();
            result.add(vertex);
            
            List<Edge<T>> neighbors = adjacencyList.getOrDefault(vertex, new ArrayList<>());
            for (Edge<T> edge : neighbors) {
                if (!visited.contains(edge.destination)) {
                    visited.add(edge.destination);
                    queue.add(edge.destination);
                }
            }
        }
        
        return result;
    }
    
    public Map<T, Double> dijkstraShortestPath(T startVertex) {
        Map<T, Double> distances = new HashMap<>();
        PriorityQueue<VertexDistance<T>> pq = new PriorityQueue<>();
        Set<T> visited = new HashSet<>();
        
        // Initialize distances
        for (T vertex : adjacencyList.keySet()) {
            distances.put(vertex, Double.POSITIVE_INFINITY);
        }
        distances.put(startVertex, 0.0);
        pq.add(new VertexDistance<>(startVertex, 0.0));
        
        while (!pq.isEmpty()) {
            VertexDistance<T> current = pq.poll();
            T currentVertex = current.vertex;
            
            if (visited.contains(currentVertex)) continue;
            visited.add(currentVertex);
            
            List<Edge<T>> neighbors = adjacencyList.getOrDefault(currentVertex, new ArrayList<>());
            for (Edge<T> edge : neighbors) {
                if (!visited.contains(edge.destination)) {
                    double newDistance = distances.get(currentVertex) + edge.weight;
                    
                    if (newDistance < distances.get(edge.destination)) {
                        distances.put(edge.destination, newDistance);
                        pq.add(new VertexDistance<>(edge.destination, newDistance));
                    }
                }
            }
        }
        
        return distances;
    }
    
    private static class VertexDistance<T> implements Comparable<VertexDistance<T>> {
        T vertex;
        double distance;
        
        VertexDistance(T vertex, double distance) {
            this.vertex = vertex;
            this.distance = distance;
        }
        
        @Override
        public int compareTo(VertexDistance<T> other) {
            return Double.compare(this.distance, other.distance);
        }
    }
    
    public boolean hasCycle() {
        if (!isDirected) {
            return hasCycleUndirected();
        } else {
            return hasCycleDirected();
        }
    }
    
    private boolean hasCycleUndirected() {
        Set<T> visited = new HashSet<>();
        
        for (T vertex : adjacencyList.keySet()) {
            if (!visited.contains(vertex)) {
                if (hasCycleUndirectedUtil(vertex, visited, null)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean hasCycleUndirectedUtil(T vertex, Set<T> visited, T parent) {
        visited.add(vertex);
        
        List<Edge<T>> neighbors = adjacencyList.getOrDefault(vertex, new ArrayList<>());
        for (Edge<T> edge : neighbors) {
            if (!visited.contains(edge.destination)) {
                if (hasCycleUndirectedUtil(edge.destination, visited, vertex)) {
                    return true;
                }
            } else if (!edge.destination.equals(parent)) {
                return true;
            }
        }
        return false;
    }
    
    private boolean hasCycleDirected() {
        Set<T> visited = new HashSet<>();
        Set<T> recursionStack = new HashSet<>();
        
        for (T vertex : adjacencyList.keySet()) {
            if (!visited.contains(vertex)) {
                if (hasCycleDirectedUtil(vertex, visited, recursionStack)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean hasCycleDirectedUtil(T vertex, Set<T> visited, Set<T> recursionStack) {
        visited.add(vertex);
        recursionStack.add(vertex);
        
        List<Edge<T>> neighbors = adjacencyList.getOrDefault(vertex, new ArrayList<>());
        for (Edge<T> edge : neighbors) {
            if (!visited.contains(edge.destination)) {
                if (hasCycleDirectedUtil(edge.destination, visited, recursionStack)) {
                    return true;
                }
            } else if (recursionStack.contains(edge.destination)) {
                return true;
            }
        }
        
        recursionStack.remove(vertex);
        return false;
    }
    
    public Set<T> getVertices() {
        return new HashSet<>(adjacencyList.keySet());
    }
    
    public int getVertexCount() {
        return adjacencyList.size();
    }
    
    public int getEdgeCount() {
        int count = 0;
        for (List<Edge<T>> edges : adjacencyList.values()) {
            count += edges.size();
        }
        return isDirected ? count : count / 2;
    }
}

// Hash Table implementation with chaining
class HashTable<K, V> {
    private static final int DEFAULT_CAPACITY = 16;
    private static final double LOAD_FACTOR_THRESHOLD = 0.75;
    
    private LinkedList<Entry<K, V>>[] buckets;
    private int size;
    private int capacity;
    
    private static class Entry<K, V> {
        K key;
        V value;
        
        Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Entry<?, ?> entry = (Entry<?, ?>) obj;
            return Objects.equals(key, entry.key);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(key);
        }
    }
    
    @SuppressWarnings("unchecked")
    public HashTable() {
        this.capacity = DEFAULT_CAPACITY;
        this.buckets = (LinkedList<Entry<K, V>>[]) Array.newInstance(LinkedList.class, capacity);
        this.size = 0;
        
        for (int i = 0; i < capacity; i++) {
            buckets[i] = new LinkedList<>();
        }
    }
    
    private int hash(K key) {
        return Math.abs(key.hashCode()) % capacity;
    }
    
    public void put(K key, V value) {
        int index = hash(key);
        LinkedList<Entry<K, V>> bucket = buckets[index];
        
        for (Entry<K, V> entry : bucket) {
            if (entry.key.equals(key)) {
                entry.value = value; // Update existing
                return;
            }
        }
        
        bucket.add(new Entry<>(key, value));
        size++;
        
        if ((double) size / capacity > LOAD_FACTOR_THRESHOLD) {
            resize();
        }
    }
    
    public V get(K key) {
        int index = hash(key);
        LinkedList<Entry<K, V>> bucket = buckets[index];
        
        for (Entry<K, V> entry : bucket) {
            if (entry.key.equals(key)) {
                return entry.value;
            }
        }
        
        return null;
    }
    
    public V remove(K key) {
        int index = hash(key);
        LinkedList<Entry<K, V>> bucket = buckets[index];
        
        Iterator<Entry<K, V>> iterator = bucket.iterator();
        while (iterator.hasNext()) {
            Entry<K, V> entry = iterator.next();
            if (entry.key.equals(key)) {
                iterator.remove();
                size--;
                return entry.value;
            }
        }
        
        return null;
    }
    
    public boolean containsKey(K key) {
        return get(key) != null;
    }
    
    @SuppressWarnings("unchecked")
    private void resize() {
        LinkedList<Entry<K, V>>[] oldBuckets = buckets;
        capacity *= 2;
        buckets = (LinkedList<Entry<K, V>>[]) Array.newInstance(LinkedList.class, capacity);
        size = 0;
        
        for (int i = 0; i < capacity; i++) {
            buckets[i] = new LinkedList<>();
        }
        
        for (LinkedList<Entry<K, V>> bucket : oldBuckets) {
            for (Entry<K, V> entry : bucket) {
                put(entry.key, entry.value);
            }
        }
    }
    
    public Set<K> keySet() {
        Set<K> keys = new HashSet<>();
        for (LinkedList<Entry<K, V>> bucket : buckets) {
            for (Entry<K, V> entry : bucket) {
                keys.add(entry.key);
            }
        }
        return keys;
    }
    
    public Collection<V> values() {
        List<V> values = new ArrayList<>();
        for (LinkedList<Entry<K, V>> bucket : buckets) {
            for (Entry<K, V> entry : bucket) {
                values.add(entry.value);
            }
        }
        return values;
    }
    
    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }
    public double getLoadFactor() { return (double) size / capacity; }
}

// Union-Find (Disjoint Set) data structure
class UnionFind {
    private int[] parent;
    private int[] rank;
    private int components;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        components = n;
        
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }
    
    public boolean union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return false;
        
        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        
        components--;
        return true;
    }
    
    public boolean isConnected(int x, int y) {
        return find(x) == find(y);
    }
    
    public int getComponents() {
        return components;
    }
}

// LRU Cache implementation
class LRUCache<K, V> {
    private final int capacity;
    private final Map<K, Node<K, V>> cache;
    private final Node<K, V> head;
    private final Node<K, V> tail;
    
    private static class Node<K, V> {
        K key;
        V value;
        Node<K, V> prev, next;
        
        Node(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.cache = new HashMap<>();
        this.head = new Node<>(null, null);
        this.tail = new Node<>(null, null);
        head.next = tail;
        tail.prev = head;
    }
    
    public V get(K key) {
        Node<K, V> node = cache.get(key);
        if (node == null) return null;
        
        moveToHead(node);
        return node.value;
    }
    
    public void put(K key, V value) {
        Node<K, V> node = cache.get(key);
        
        if (node != null) {
            node.value = value;
            moveToHead(node);
        } else {
            Node<K, V> newNode = new Node<>(key, value);
            
            if (cache.size() >= capacity) {
                Node<K, V> lastNode = removeTail();
                cache.remove(lastNode.key);
            }
            
            cache.put(key, newNode);
            addToHead(newNode);
        }
    }
    
    private void addToHead(Node<K, V> node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }
    
    private void removeNode(Node<K, V> node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void moveToHead(Node<K, V> node) {
        removeNode(node);
        addToHead(node);
    }
    
    private Node<K, V> removeTail() {
        Node<K, V> lastNode = tail.prev;
        removeNode(lastNode);
        return lastNode;
    }
    
    public int size() {
        return cache.size();
    }
    
    public boolean containsKey(K key) {
        return cache.containsKey(key);
    }
}

// Main demonstration class
public class DataStructuresDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Advanced Data Structures Demonstration ===\n");
        
        // Test Binary Search Tree
        System.out.println("=== Binary Search Tree (AVL) ===");
        BinarySearchTree<Integer> bst = new BinarySearchTree<>();
        int[] values = {10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8};
        
        for (int value : values) {
            bst.insert(value);
        }
        
        System.out.println("Inserted values: " + Arrays.toString(values));
        System.out.println("Inorder traversal: " + bst.inorderTraversal());
        System.out.println("Search 7: " + bst.search(7));
        System.out.println("Search 99: " + bst.search(99));
        
        bst.delete(10);
        System.out.println("After deleting 10: " + bst.inorderTraversal());
        
        // Test Heap
        System.out.println("\n=== Max Heap ===");
        Heap<Integer> maxHeap = new Heap<>(true);
        int[] heapValues = {4, 10, 3, 5, 1, 8, 15, 2, 7, 6};
        
        for (int value : heapValues) {
            maxHeap.insert(value);
        }
        
        System.out.println("Inserted values: " + Arrays.toString(heapValues));
        System.out.println("Max element: " + maxHeap.peek());
        
        System.out.print("Extracting elements: ");
        while (!maxHeap.isEmpty()) {
            System.out.print(maxHeap.extractTop() + " ");
        }
        System.out.println();
        
        // Test Trie
        System.out.println("\n=== Trie (Prefix Tree) ===");
        Trie trie = new Trie();
        String[] words = {"hello", "world", "help", "heap", "heat", "he"};
        
        for (String word : words) {
            trie.insert(word);
        }
        
        System.out.println("Inserted words: " + Arrays.toString(words));
        System.out.println("Search 'hello': " + trie.search("hello"));
        System.out.println("Search 'hell': " + trie.search("hell"));
        System.out.println("Starts with 'he': " + trie.startsWith("he"));
        System.out.println("Words with prefix 'he': " + trie.getAllWordsWithPrefix("he"));
        
        // Test Graph
        System.out.println("\n=== Graph Algorithms ===");
        Graph<String> graph = new Graph<>(true);
        
        graph.addEdge("A", "B", 4);
        graph.addEdge("A", "C", 2);
        graph.addEdge("B", "C", 1);
        graph.addEdge("B", "D", 5);
        graph.addEdge("C", "D", 8);
        graph.addEdge("C", "E", 10);
        graph.addEdge("D", "E", 2);
        
        System.out.println("DFS from A: " + graph.depthFirstSearch("A"));
        System.out.println("BFS from A: " + graph.breadthFirstSearch("A"));
        System.out.println("Shortest paths from A: " + graph.dijkstraShortestPath("A"));
        System.out.println("Has cycle: " + graph.hasCycle());
        
        // Test Hash Table
        System.out.println("\n=== Custom Hash Table ===");
        HashTable<String, Integer> hashTable = new HashTable<>();
        
        hashTable.put("apple", 10);
        hashTable.put("banana", 20);
        hashTable.put("cherry", 30);
        hashTable.put("date", 40);
        
        System.out.println("Get 'banana': " + hashTable.get("banana"));
        System.out.println("Contains 'apple': " + hashTable.containsKey("apple"));
        System.out.println("Size: " + hashTable.size());
        System.out.println("Load factor: " + String.format("%.2f", hashTable.getLoadFactor()));
        
        hashTable.remove("banana");
        System.out.println("After removing 'banana', size: " + hashTable.size());
        
        // Test Union-Find
        System.out.println("\n=== Union-Find ===");
        UnionFind uf = new UnionFind(6);
        
        System.out.println("Initial components: " + uf.getComponents());
        
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        
        System.out.println("After unions, components: " + uf.getComponents());
        System.out.println("0 and 2 connected: " + uf.isConnected(0, 2));
        System.out.println("2 and 4 connected: " + uf.isConnected(2, 4));
        
        // Test LRU Cache
        System.out.println("\n=== LRU Cache ===");
        LRUCache<String, String> cache = new LRUCache<>(3);
        
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");
        
        System.out.println("Cache size: " + cache.size());
        System.out.println("Get key2: " + cache.get("key2"));
        
        cache.put("key4", "value4"); // This should evict key1
        
        System.out.println("Contains key1: " + cache.containsKey("key1"));
        System.out.println("Contains key2: " + cache.containsKey("key2"));
        System.out.println("Contains key4: " + cache.containsKey("key4"));
        
        System.out.println("\nData structures demonstration completed!");
    }
}