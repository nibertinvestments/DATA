/// Dart Data Structures and Algorithms - Intermediate Examples
/// This file demonstrates various data structures and algorithms implemented in Dart
/// focusing on intermediate concepts and best practices

import 'dart:collection';
import 'dart:math' as math;

/// Generic Binary Search Tree implementation
class BinarySearchTree<T extends Comparable<T>> {
  TreeNode<T>? _root;

  /// Insert a value into the BST
  void insert(T value) {
    _root = _insertRecursive(_root, value);
  }

  TreeNode<T> _insertRecursive(TreeNode<T>? node, T value) {
    if (node == null) {
      return TreeNode<T>(value);
    }

    if (value.compareTo(node.value) < 0) {
      node.left = _insertRecursive(node.left, value);
    } else if (value.compareTo(node.value) > 0) {
      node.right = _insertRecursive(node.right, value);
    }

    return node;
  }

  /// Search for a value in the BST
  bool contains(T value) {
    return _searchRecursive(_root, value);
  }

  bool _searchRecursive(TreeNode<T>? node, T value) {
    if (node == null) return false;
    
    final comparison = value.compareTo(node.value);
    if (comparison == 0) return true;
    
    return comparison < 0
        ? _searchRecursive(node.left, value)
        : _searchRecursive(node.right, value);
  }

  /// In-order traversal of the BST
  List<T> inOrderTraversal() {
    final result = <T>[];
    _inOrderRecursive(_root, result);
    return result;
  }

  void _inOrderRecursive(TreeNode<T>? node, List<T> result) {
    if (node != null) {
      _inOrderRecursive(node.left, result);
      result.add(node.value);
      _inOrderRecursive(node.right, result);
    }
  }

  /// Get the height of the tree
  int get height => _getHeight(_root);

  int _getHeight(TreeNode<T>? node) {
    if (node == null) return 0;
    return 1 + math.max(_getHeight(node.left), _getHeight(node.right));
  }

  /// Delete a value from the BST
  void delete(T value) {
    _root = _deleteRecursive(_root, value);
  }

  TreeNode<T>? _deleteRecursive(TreeNode<T>? node, T value) {
    if (node == null) return null;

    final comparison = value.compareTo(node.value);
    if (comparison < 0) {
      node.left = _deleteRecursive(node.left, value);
    } else if (comparison > 0) {
      node.right = _deleteRecursive(node.right, value);
    } else {
      // Node to be deleted found
      if (node.left == null) return node.right;
      if (node.right == null) return node.left;

      // Node has two children
      final successor = _findMin(node.right!);
      node.value = successor.value;
      node.right = _deleteRecursive(node.right, successor.value);
    }

    return node;
  }

  TreeNode<T> _findMin(TreeNode<T> node) {
    while (node.left != null) {
      node = node.left!;
    }
    return node;
  }
}

class TreeNode<T> {
  T value;
  TreeNode<T>? left;
  TreeNode<T>? right;

  TreeNode(this.value);
}

/// Graph implementation with adjacency list
class Graph<T> {
  final Map<T, Set<T>> _adjacencyList = {};

  /// Add a vertex to the graph
  void addVertex(T vertex) {
    _adjacencyList.putIfAbsent(vertex, () => <T>{});
  }

  /// Add an edge between two vertices
  void addEdge(T from, T to, {bool directed = false}) {
    addVertex(from);
    addVertex(to);
    
    _adjacencyList[from]!.add(to);
    if (!directed) {
      _adjacencyList[to]!.add(from);
    }
  }

  /// Get neighbors of a vertex
  Set<T> getNeighbors(T vertex) {
    return _adjacencyList[vertex] ?? <T>{};
  }

  /// Breadth-First Search
  List<T> bfs(T start) {
    final visited = <T>{};
    final queue = Queue<T>();
    final result = <T>[];

    queue.add(start);
    visited.add(start);

    while (queue.isNotEmpty) {
      final current = queue.removeFirst();
      result.add(current);

      for (final neighbor in getNeighbors(current)) {
        if (!visited.contains(neighbor)) {
          visited.add(neighbor);
          queue.add(neighbor);
        }
      }
    }

    return result;
  }

  /// Depth-First Search
  List<T> dfs(T start) {
    final visited = <T>{};
    final result = <T>[];

    void dfsRecursive(T vertex) {
      visited.add(vertex);
      result.add(vertex);

      for (final neighbor in getNeighbors(vertex)) {
        if (!visited.contains(neighbor)) {
          dfsRecursive(neighbor);
        }
      }
    }

    dfsRecursive(start);
    return result;
  }

  /// Check if the graph has a cycle (for undirected graphs)
  bool hasCycle() {
    final visited = <T>{};

    bool hasCycleUtil(T vertex, T? parent) {
      visited.add(vertex);

      for (final neighbor in getNeighbors(vertex)) {
        if (!visited.contains(neighbor)) {
          if (hasCycleUtil(neighbor, vertex)) {
            return true;
          }
        } else if (neighbor != parent) {
          return true;
        }
      }

      return false;
    }

    for (final vertex in _adjacencyList.keys) {
      if (!visited.contains(vertex)) {
        if (hasCycleUtil(vertex, null)) {
          return true;
        }
      }
    }

    return false;
  }

  /// Find shortest path using Dijkstra's algorithm
  Map<T, int> dijkstra(T start, Map<String, int> weights) {
    final distances = <T, int>{};
    final visited = <T>{};
    final priorityQueue = PriorityQueue<MapEntry<T, int>>((a, b) => a.value - b.value);

    // Initialize distances
    for (final vertex in _adjacencyList.keys) {
      distances[vertex] = vertex == start ? 0 : double.maxFinite.toInt();
    }

    priorityQueue.add(MapEntry(start, 0));

    while (priorityQueue.isNotEmpty) {
      final current = priorityQueue.removeFirst();
      final currentVertex = current.key;

      if (visited.contains(currentVertex)) continue;
      visited.add(currentVertex);

      for (final neighbor in getNeighbors(currentVertex)) {
        if (!visited.contains(neighbor)) {
          final edgeWeight = weights['$currentVertex-$neighbor'] ?? 1;
          final newDistance = distances[currentVertex]! + edgeWeight;

          if (newDistance < distances[neighbor]!) {
            distances[neighbor] = newDistance;
            priorityQueue.add(MapEntry(neighbor, newDistance));
          }
        }
      }
    }

    return distances;
  }
}

/// Priority Queue implementation using a heap
class PriorityQueue<T> {
  final List<T> _heap = [];
  final int Function(T, T) _compare;

  PriorityQueue(this._compare);

  bool get isEmpty => _heap.isEmpty;
  bool get isNotEmpty => _heap.isNotEmpty;
  int get length => _heap.length;

  void add(T item) {
    _heap.add(item);
    _bubbleUp(_heap.length - 1);
  }

  T removeFirst() {
    if (_heap.isEmpty) {
      throw StateError('Priority queue is empty');
    }

    final result = _heap[0];
    final last = _heap.removeLast();

    if (_heap.isNotEmpty) {
      _heap[0] = last;
      _bubbleDown(0);
    }

    return result;
  }

  T get first {
    if (_heap.isEmpty) {
      throw StateError('Priority queue is empty');
    }
    return _heap[0];
  }

  void _bubbleUp(int index) {
    while (index > 0) {
      final parentIndex = (index - 1) ~/ 2;
      if (_compare(_heap[index], _heap[parentIndex]) >= 0) break;

      _swap(index, parentIndex);
      index = parentIndex;
    }
  }

  void _bubbleDown(int index) {
    while (true) {
      final leftChild = 2 * index + 1;
      final rightChild = 2 * index + 2;
      int smallest = index;

      if (leftChild < _heap.length &&
          _compare(_heap[leftChild], _heap[smallest]) < 0) {
        smallest = leftChild;
      }

      if (rightChild < _heap.length &&
          _compare(_heap[rightChild], _heap[smallest]) < 0) {
        smallest = rightChild;
      }

      if (smallest == index) break;

      _swap(index, smallest);
      index = smallest;
    }
  }

  void _swap(int i, int j) {
    final temp = _heap[i];
    _heap[i] = _heap[j];
    _heap[j] = temp;
  }
}

/// Trie (Prefix Tree) implementation for string operations
class Trie {
  final TrieNode _root = TrieNode();

  /// Insert a word into the trie
  void insert(String word) {
    TrieNode current = _root;
    
    for (final char in word.toLowerCase().split('')) {
      current.children.putIfAbsent(char, () => TrieNode());
      current = current.children[char]!;
    }
    
    current.isEndOfWord = true;
  }

  /// Search for a word in the trie
  bool search(String word) {
    final node = _searchNode(word);
    return node != null && node.isEndOfWord;
  }

  /// Check if any word starts with the given prefix
  bool startsWith(String prefix) {
    return _searchNode(prefix) != null;
  }

  TrieNode? _searchNode(String word) {
    TrieNode current = _root;
    
    for (final char in word.toLowerCase().split('')) {
      if (!current.children.containsKey(char)) {
        return null;
      }
      current = current.children[char]!;
    }
    
    return current;
  }

  /// Get all words with a given prefix
  List<String> getWordsWithPrefix(String prefix) {
    final node = _searchNode(prefix);
    if (node == null) return [];

    final words = <String>[];
    _collectWords(node, prefix, words);
    return words;
  }

  void _collectWords(TrieNode node, String prefix, List<String> words) {
    if (node.isEndOfWord) {
      words.add(prefix);
    }

    for (final entry in node.children.entries) {
      _collectWords(entry.value, prefix + entry.key, words);
    }
  }

  /// Delete a word from the trie
  void delete(String word) {
    _deleteRecursive(_root, word.toLowerCase(), 0);
  }

  bool _deleteRecursive(TrieNode node, String word, int index) {
    if (index == word.length) {
      if (!node.isEndOfWord) return false;
      
      node.isEndOfWord = false;
      return node.children.isEmpty;
    }

    final char = word[index];
    final childNode = node.children[char];
    if (childNode == null) return false;

    final shouldDeleteChild = _deleteRecursive(childNode, word, index + 1);

    if (shouldDeleteChild) {
      node.children.remove(char);
      return !node.isEndOfWord && node.children.isEmpty;
    }

    return false;
  }
}

class TrieNode {
  final Map<String, TrieNode> children = {};
  bool isEndOfWord = false;
}

/// Sorting algorithms implementation
class SortingAlgorithms {
  /// Quick Sort implementation
  static void quickSort<T extends Comparable<T>>(List<T> list, [int? low, int? high]) {
    low ??= 0;
    high ??= list.length - 1;

    if (low < high) {
      final pivotIndex = _partition(list, low, high);
      quickSort(list, low, pivotIndex - 1);
      quickSort(list, pivotIndex + 1, high);
    }
  }

  static int _partition<T extends Comparable<T>>(List<T> list, int low, int high) {
    final pivot = list[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
      if (list[j].compareTo(pivot) <= 0) {
        i++;
        _swap(list, i, j);
      }
    }

    _swap(list, i + 1, high);
    return i + 1;
  }

  /// Merge Sort implementation
  static List<T> mergeSort<T extends Comparable<T>>(List<T> list) {
    if (list.length <= 1) return List.from(list);

    final middle = list.length ~/ 2;
    final left = mergeSort(list.sublist(0, middle));
    final right = mergeSort(list.sublist(middle));

    return _merge(left, right);
  }

  static List<T> _merge<T extends Comparable<T>>(List<T> left, List<T> right) {
    final result = <T>[];
    int i = 0, j = 0;

    while (i < left.length && j < right.length) {
      if (left[i].compareTo(right[j]) <= 0) {
        result.add(left[i++]);
      } else {
        result.add(right[j++]);
      }
    }

    result.addAll(left.sublist(i));
    result.addAll(right.sublist(j));
    return result;
  }

  /// Heap Sort implementation
  static void heapSort<T extends Comparable<T>>(List<T> list) {
    // Build max heap
    for (int i = list.length ~/ 2 - 1; i >= 0; i--) {
      _heapify(list, list.length, i);
    }

    // Extract elements from heap one by one
    for (int i = list.length - 1; i > 0; i--) {
      _swap(list, 0, i);
      _heapify(list, i, 0);
    }
  }

  static void _heapify<T extends Comparable<T>>(List<T> list, int heapSize, int rootIndex) {
    int largest = rootIndex;
    final leftChild = 2 * rootIndex + 1;
    final rightChild = 2 * rootIndex + 2;

    if (leftChild < heapSize && list[leftChild].compareTo(list[largest]) > 0) {
      largest = leftChild;
    }

    if (rightChild < heapSize && list[rightChild].compareTo(list[largest]) > 0) {
      largest = rightChild;
    }

    if (largest != rootIndex) {
      _swap(list, rootIndex, largest);
      _heapify(list, heapSize, largest);
    }
  }

  static void _swap<T>(List<T> list, int i, int j) {
    final temp = list[i];
    list[i] = list[j];
    list[j] = temp;
  }
}

/// Dynamic Programming examples
class DynamicProgramming {
  /// Fibonacci with memoization
  static final Map<int, int> _fibCache = {};

  static int fibonacci(int n) {
    if (n <= 1) return n;
    
    if (_fibCache.containsKey(n)) {
      return _fibCache[n]!;
    }

    final result = fibonacci(n - 1) + fibonacci(n - 2);
    _fibCache[n] = result;
    return result;
  }

  /// Longest Common Subsequence
  static int longestCommonSubsequence(String text1, String text2) {
    final m = text1.length;
    final n = text2.length;
    final dp = List.generate(m + 1, (_) => List.filled(n + 1, 0));

    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        if (text1[i - 1] == text2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    return dp[m][n];
  }

  /// Knapsack problem (0/1)
  static int knapsack(List<int> weights, List<int> values, int capacity) {
    final n = weights.length;
    final dp = List.generate(n + 1, (_) => List.filled(capacity + 1, 0));

    for (int i = 1; i <= n; i++) {
      for (int w = 1; w <= capacity; w++) {
        if (weights[i - 1] <= w) {
          dp[i][w] = math.max(
            values[i - 1] + dp[i - 1][w - weights[i - 1]],
            dp[i - 1][w],
          );
        } else {
          dp[i][w] = dp[i - 1][w];
        }
      }
    }

    return dp[n][capacity];
  }

  /// Coin change problem
  static int coinChange(List<int> coins, int amount) {
    final dp = List.filled(amount + 1, amount + 1);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
      for (final coin in coins) {
        if (coin <= i) {
          dp[i] = math.min(dp[i], dp[i - coin] + 1);
        }
      }
    }

    return dp[amount] > amount ? -1 : dp[amount];
  }
}

/// Utility class for algorithm testing and demonstration
class AlgorithmUtils {
  /// Generate random list for testing
  static List<int> generateRandomList(int size, [int max = 1000]) {
    final random = math.Random();
    return List.generate(size, (_) => random.nextInt(max));
  }

  /// Measure execution time of a function
  static Duration measureExecutionTime(void Function() function) {
    final stopwatch = Stopwatch()..start();
    function();
    stopwatch.stop();
    return stopwatch.elapsed;
  }

  /// Test if a list is sorted
  static bool isSorted<T extends Comparable<T>>(List<T> list) {
    for (int i = 1; i < list.length; i++) {
      if (list[i].compareTo(list[i - 1]) < 0) {
        return false;
      }
    }
    return true;
  }

  /// Binary search implementation
  static int binarySearch<T extends Comparable<T>>(List<T> sortedList, T target) {
    int left = 0;
    int right = sortedList.length - 1;

    while (left <= right) {
      final mid = (left + right) ~/ 2;
      final comparison = sortedList[mid].compareTo(target);

      if (comparison == 0) {
        return mid;
      } else if (comparison < 0) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    return -1; // Not found
  }
}

/// Example usage and testing
void main() {
  print('=== Dart Data Structures and Algorithms Demo ===\n');

  // Binary Search Tree
  print('Binary Search Tree:');
  final bst = BinarySearchTree<int>();
  [50, 30, 70, 20, 40, 60, 80].forEach(bst.insert);
  print('In-order traversal: ${bst.inOrderTraversal()}');
  print('Contains 40: ${bst.contains(40)}');
  print('Tree height: ${bst.height}\n');

  // Graph
  print('Graph BFS/DFS:');
  final graph = Graph<String>();
  graph.addEdge('A', 'B');
  graph.addEdge('A', 'C');
  graph.addEdge('B', 'D');
  graph.addEdge('C', 'E');
  print('BFS from A: ${graph.bfs('A')}');
  print('DFS from A: ${graph.dfs('A')}\n');

  // Trie
  print('Trie (Prefix Tree):');
  final trie = Trie();
  ['apple', 'app', 'application', 'apply', 'banana'].forEach(trie.insert);
  print('Words with prefix "app": ${trie.getWordsWithPrefix('app')}');
  print('Contains "apple": ${trie.search('apple')}\n');

  // Sorting algorithms
  print('Sorting Algorithms:');
  final unsortedList = AlgorithmUtils.generateRandomList(10, 100);
  print('Original: $unsortedList');
  
  final quickSortList = List.from(unsortedList);
  SortingAlgorithms.quickSort(quickSortList);
  print('Quick Sort: $quickSortList');
  
  final mergeSortList = SortingAlgorithms.mergeSort(unsortedList);
  print('Merge Sort: $mergeSortList\n');

  // Dynamic Programming
  print('Dynamic Programming:');
  print('Fibonacci(20): ${DynamicProgramming.fibonacci(20)}');
  print('LCS("ABCDGH", "AEDFHR"): ${DynamicProgramming.longestCommonSubsequence("ABCDGH", "AEDFHR")}');
  print('Coin change for 11 with [1,2,5]: ${DynamicProgramming.coinChange([1, 2, 5], 11)}');

  // Performance testing
  print('\nPerformance Testing:');
  final largeList = AlgorithmUtils.generateRandomList(10000);
  
  final quickSortTime = AlgorithmUtils.measureExecutionTime(() {
    final testList = List.from(largeList);
    SortingAlgorithms.quickSort(testList);
  });
  
  final mergeSortTime = AlgorithmUtils.measureExecutionTime(() {
    SortingAlgorithms.mergeSort(largeList);
  });
  
  print('Quick Sort (10k elements): ${quickSortTime.inMicroseconds}μs');
  print('Merge Sort (10k elements): ${mergeSortTime.inMicroseconds}μs');
}