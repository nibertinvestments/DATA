// Advanced algorithms and data structures in JavaScript

console.log("=== Advanced Algorithms & Data Structures ===");

// Advanced Sorting Algorithms
class SortingAlgorithms {
    // Merge Sort - Stable, O(n log n)
    static mergeSort(arr) {
        if (arr.length <= 1) return arr;
        
        const mid = Math.floor(arr.length / 2);
        const left = this.mergeSort(arr.slice(0, mid));
        const right = this.mergeSort(arr.slice(mid));
        
        return this.merge(left, right);
    }
    
    static merge(left, right) {
        const result = [];
        let i = 0, j = 0;
        
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result.push(left[i++]);
            } else {
                result.push(right[j++]);
            }
        }
        
        return result.concat(left.slice(i)).concat(right.slice(j));
    }
    
    // Quick Sort - Average O(n log n), Worst O(n²)
    static quickSort(arr, low = 0, high = arr.length - 1) {
        if (low < high) {
            const pi = this.partition(arr, low, high);
            this.quickSort(arr, low, pi - 1);
            this.quickSort(arr, pi + 1, high);
        }
        return arr;
    }
    
    static partition(arr, low, high) {
        const pivot = arr[high];
        let i = low - 1;
        
        for (let j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
        }
        
        [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
        return i + 1;
    }
    
    // Heap Sort - O(n log n)
    static heapSort(arr) {
        const n = arr.length;
        
        // Build max heap
        for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
            this.heapify(arr, n, i);
        }
        
        // Extract elements one by one
        for (let i = n - 1; i > 0; i--) {
            [arr[0], arr[i]] = [arr[i], arr[0]];
            this.heapify(arr, i, 0);
        }
        
        return arr;
    }
    
    static heapify(arr, n, i) {
        let largest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest !== i) {
            [arr[i], arr[largest]] = [arr[largest], arr[i]];
            this.heapify(arr, n, largest);
        }
    }
    
    // Radix Sort - O(d * (n + k))
    static radixSort(arr) {
        const max = Math.max(...arr);
        
        for (let exp = 1; Math.floor(max / exp) > 0; exp *= 10) {
            this.countingSort(arr, exp);
        }
        
        return arr;
    }
    
    static countingSort(arr, exp) {
        const n = arr.length;
        const output = new Array(n);
        const count = new Array(10).fill(0);
        
        // Count occurrences of each digit
        for (let i = 0; i < n; i++) {
            count[Math.floor(arr[i] / exp) % 10]++;
        }
        
        // Change count[i] to actual position
        for (let i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        
        // Build output array
        for (let i = n - 1; i >= 0; i--) {
            const digit = Math.floor(arr[i] / exp) % 10;
            output[count[digit] - 1] = arr[i];
            count[digit]--;
        }
        
        // Copy output array to arr
        for (let i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }
}

// Advanced Tree Data Structures
class AVLNode {
    constructor(value) {
        this.value = value;
        this.left = null;
        this.right = null;
        this.height = 1;
    }
}

class AVLTree {
    constructor() {
        this.root = null;
    }
    
    getHeight(node) {
        return node ? node.height : 0;
    }
    
    getBalance(node) {
        return node ? this.getHeight(node.left) - this.getHeight(node.right) : 0;
    }
    
    updateHeight(node) {
        if (node) {
            node.height = 1 + Math.max(
                this.getHeight(node.left),
                this.getHeight(node.right)
            );
        }
    }
    
    rotateRight(y) {
        const x = y.left;
        const T2 = x.right;
        
        x.right = y;
        y.left = T2;
        
        this.updateHeight(y);
        this.updateHeight(x);
        
        return x;
    }
    
    rotateLeft(x) {
        const y = x.right;
        const T2 = y.left;
        
        y.left = x;
        x.right = T2;
        
        this.updateHeight(x);
        this.updateHeight(y);
        
        return y;
    }
    
    insert(value) {
        this.root = this.insertNode(this.root, value);
    }
    
    insertNode(node, value) {
        // Normal BST insertion
        if (!node) return new AVLNode(value);
        
        if (value < node.value) {
            node.left = this.insertNode(node.left, value);
        } else if (value > node.value) {
            node.right = this.insertNode(node.right, value);
        } else {
            return node; // Duplicate values not allowed
        }
        
        // Update height
        this.updateHeight(node);
        
        // Get balance factor
        const balance = this.getBalance(node);
        
        // Left Left Case
        if (balance > 1 && value < node.left.value) {
            return this.rotateRight(node);
        }
        
        // Right Right Case
        if (balance < -1 && value > node.right.value) {
            return this.rotateLeft(node);
        }
        
        // Left Right Case
        if (balance > 1 && value > node.left.value) {
            node.left = this.rotateLeft(node.left);
            return this.rotateRight(node);
        }
        
        // Right Left Case
        if (balance < -1 && value < node.right.value) {
            node.right = this.rotateRight(node.right);
            return this.rotateLeft(node);
        }
        
        return node;
    }
    
    inorderTraversal() {
        const result = [];
        this.inorder(this.root, result);
        return result;
    }
    
    inorder(node, result) {
        if (node) {
            this.inorder(node.left, result);
            result.push(node.value);
            this.inorder(node.right, result);
        }
    }
}

// Trie (Prefix Tree)
class TrieNode {
    constructor() {
        this.children = {};
        this.isEndOfWord = false;
        this.wordCount = 0; // For counting word occurrences
    }
}

class Trie {
    constructor() {
        this.root = new TrieNode();
    }
    
    insert(word) {
        let current = this.root;
        
        for (const char of word.toLowerCase()) {
            if (!current.children[char]) {
                current.children[char] = new TrieNode();
            }
            current = current.children[char];
        }
        
        current.isEndOfWord = true;
        current.wordCount++;
    }
    
    search(word) {
        let current = this.root;
        
        for (const char of word.toLowerCase()) {
            if (!current.children[char]) {
                return false;
            }
            current = current.children[char];
        }
        
        return current.isEndOfWord;
    }
    
    startsWith(prefix) {
        let current = this.root;
        
        for (const char of prefix.toLowerCase()) {
            if (!current.children[char]) {
                return false;
            }
            current = current.children[char];
        }
        
        return true;
    }
    
    getAllWordsWithPrefix(prefix) {
        const words = [];
        let current = this.root;
        
        // Navigate to the prefix
        for (const char of prefix.toLowerCase()) {
            if (!current.children[char]) {
                return words;
            }
            current = current.children[char];
        }
        
        // DFS to find all words
        this.dfsWords(current, prefix.toLowerCase(), words);
        return words;
    }
    
    dfsWords(node, currentWord, words) {
        if (node.isEndOfWord) {
            words.push(currentWord);
        }
        
        for (const [char, childNode] of Object.entries(node.children)) {
            this.dfsWords(childNode, currentWord + char, words);
        }
    }
    
    delete(word) {
        this.deleteHelper(this.root, word.toLowerCase(), 0);
    }
    
    deleteHelper(node, word, index) {
        if (index === word.length) {
            if (!node.isEndOfWord) return false;
            
            node.isEndOfWord = false;
            node.wordCount = 0;
            
            // Return true if current has no children
            return Object.keys(node.children).length === 0;
        }
        
        const char = word[index];
        const childNode = node.children[char];
        
        if (!childNode) return false;
        
        const shouldDeleteChild = this.deleteHelper(childNode, word, index + 1);
        
        if (shouldDeleteChild) {
            delete node.children[char];
            
            // Return true if current node has no children and is not end of word
            return !node.isEndOfWord && Object.keys(node.children).length === 0;
        }
        
        return false;
    }
}

// Graph Algorithms
class Graph {
    constructor(directed = false) {
        this.adjacencyList = new Map();
        this.directed = directed;
    }
    
    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }
    
    addEdge(vertex1, vertex2, weight = 1) {
        this.addVertex(vertex1);
        this.addVertex(vertex2);
        
        this.adjacencyList.get(vertex1).push({ vertex: vertex2, weight });
        
        if (!this.directed) {
            this.adjacencyList.get(vertex2).push({ vertex: vertex1, weight });
        }
    }
    
    // Depth-First Search
    dfs(startVertex, visited = new Set()) {
        const result = [];
        
        const dfsHelper = (vertex) => {
            visited.add(vertex);
            result.push(vertex);
            
            const neighbors = this.adjacencyList.get(vertex) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor.vertex)) {
                    dfsHelper(neighbor.vertex);
                }
            }
        };
        
        dfsHelper(startVertex);
        return result;
    }
    
    // Breadth-First Search
    bfs(startVertex) {
        const visited = new Set();
        const queue = [startVertex];
        const result = [];
        
        visited.add(startVertex);
        
        while (queue.length > 0) {
            const vertex = queue.shift();
            result.push(vertex);
            
            const neighbors = this.adjacencyList.get(vertex) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor.vertex)) {
                    visited.add(neighbor.vertex);
                    queue.push(neighbor.vertex);
                }
            }
        }
        
        return result;
    }
    
    // Dijkstra's Shortest Path Algorithm
    dijkstra(startVertex) {
        const distances = new Map();
        const previous = new Map();
        const visited = new Set();
        const priorityQueue = new MinHeap();
        
        // Initialize distances
        for (const vertex of this.adjacencyList.keys()) {
            distances.set(vertex, vertex === startVertex ? 0 : Infinity);
            priorityQueue.insert({ vertex, distance: distances.get(vertex) });
        }
        
        while (!priorityQueue.isEmpty()) {
            const { vertex: currentVertex } = priorityQueue.extractMin();
            
            if (visited.has(currentVertex)) continue;
            visited.add(currentVertex);
            
            const neighbors = this.adjacencyList.get(currentVertex) || [];
            
            for (const neighbor of neighbors) {
                const { vertex: neighborVertex, weight } = neighbor;
                const newDistance = distances.get(currentVertex) + weight;
                
                if (newDistance < distances.get(neighborVertex)) {
                    distances.set(neighborVertex, newDistance);
                    previous.set(neighborVertex, currentVertex);
                    priorityQueue.insert({ vertex: neighborVertex, distance: newDistance });
                }
            }
        }
        
        return { distances, previous };
    }
    
    // Topological Sort (for DAG)
    topologicalSort() {
        const visited = new Set();
        const stack = [];
        
        const topologicalSortHelper = (vertex) => {
            visited.add(vertex);
            
            const neighbors = this.adjacencyList.get(vertex) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor.vertex)) {
                    topologicalSortHelper(neighbor.vertex);
                }
            }
            
            stack.push(vertex);
        };
        
        for (const vertex of this.adjacencyList.keys()) {
            if (!visited.has(vertex)) {
                topologicalSortHelper(vertex);
            }
        }
        
        return stack.reverse();
    }
    
    // Detect cycle in directed graph
    hasCycle() {
        const visited = new Set();
        const recursionStack = new Set();
        
        const hasCycleHelper = (vertex) => {
            visited.add(vertex);
            recursionStack.add(vertex);
            
            const neighbors = this.adjacencyList.get(vertex) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor.vertex)) {
                    if (hasCycleHelper(neighbor.vertex)) {
                        return true;
                    }
                } else if (recursionStack.has(neighbor.vertex)) {
                    return true;
                }
            }
            
            recursionStack.delete(vertex);
            return false;
        };
        
        for (const vertex of this.adjacencyList.keys()) {
            if (!visited.has(vertex)) {
                if (hasCycleHelper(vertex)) {
                    return true;
                }
            }
        }
        
        return false;
    }
}

// Min Heap for Dijkstra's algorithm
class MinHeap {
    constructor() {
        this.heap = [];
    }
    
    insert(element) {
        this.heap.push(element);
        this.heapifyUp(this.heap.length - 1);
    }
    
    extractMin() {
        if (this.heap.length === 0) return null;
        if (this.heap.length === 1) return this.heap.pop();
        
        const min = this.heap[0];
        this.heap[0] = this.heap.pop();
        this.heapifyDown(0);
        
        return min;
    }
    
    heapifyUp(index) {
        const parentIndex = Math.floor((index - 1) / 2);
        
        if (parentIndex >= 0 && this.heap[parentIndex].distance > this.heap[index].distance) {
            [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
            this.heapifyUp(parentIndex);
        }
    }
    
    heapifyDown(index) {
        const leftChild = 2 * index + 1;
        const rightChild = 2 * index + 2;
        let smallest = index;
        
        if (leftChild < this.heap.length && 
            this.heap[leftChild].distance < this.heap[smallest].distance) {
            smallest = leftChild;
        }
        
        if (rightChild < this.heap.length && 
            this.heap[rightChild].distance < this.heap[smallest].distance) {
            smallest = rightChild;
        }
        
        if (smallest !== index) {
            [this.heap[index], this.heap[smallest]] = [this.heap[smallest], this.heap[index]];
            this.heapifyDown(smallest);
        }
    }
    
    isEmpty() {
        return this.heap.length === 0;
    }
}

// Dynamic Programming Examples
class DynamicProgramming {
    // Fibonacci with memoization
    static fibonacciMemo() {
        const memo = {};
        
        function fib(n) {
            if (n in memo) return memo[n];
            if (n <= 2) return 1;
            
            memo[n] = fib(n - 1) + fib(n - 2);
            return memo[n];
        }
        
        return fib;
    }
    
    // Longest Common Subsequence
    static longestCommonSubsequence(text1, text2) {
        const m = text1.length;
        const n = text2.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
        
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (text1[i - 1] === text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    // Knapsack Problem
    static knapsack(capacity, weights, values, n) {
        const dp = Array(n + 1).fill(null).map(() => Array(capacity + 1).fill(0));
        
        for (let i = 1; i <= n; i++) {
            for (let w = 1; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        return dp[n][capacity];
    }
    
    // Edit Distance (Levenshtein Distance)
    static editDistance(str1, str2) {
        const m = str1.length;
        const n = str2.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
        
        // Initialize first row and column
        for (let i = 0; i <= m; i++) dp[i][0] = i;
        for (let j = 0; j <= n; j++) dp[0][j] = j;
        
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (str1[i - 1] === str2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(
                        dp[i - 1][j],     // deletion
                        dp[i][j - 1],     // insertion
                        dp[i - 1][j - 1]  // substitution
                    );
                }
            }
        }
        
        return dp[m][n];
    }
}

// String Algorithms
class StringAlgorithms {
    // KMP Pattern Matching
    static kmpSearch(text, pattern) {
        const lps = this.computeLPS(pattern);
        const matches = [];
        let i = 0; // index for text
        let j = 0; // index for pattern
        
        while (i < text.length) {
            if (pattern[j] === text[i]) {
                i++;
                j++;
            }
            
            if (j === pattern.length) {
                matches.push(i - j);
                j = lps[j - 1];
            } else if (i < text.length && pattern[j] !== text[i]) {
                if (j !== 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return matches;
    }
    
    static computeLPS(pattern) {
        const lps = new Array(pattern.length).fill(0);
        let len = 0;
        let i = 1;
        
        while (i < pattern.length) {
            if (pattern[i] === pattern[len]) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len !== 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }
    
    // Rabin-Karp Algorithm
    static rabinKarp(text, pattern, prime = 101) {
        const matches = [];
        const m = pattern.length;
        const n = text.length;
        const d = 256; // number of characters in alphabet
        
        let patternHash = 0;
        let textHash = 0;
        let h = 1;
        
        // Calculate h = pow(d, m-1) % prime
        for (let i = 0; i < m - 1; i++) {
            h = (h * d) % prime;
        }
        
        // Calculate hash value for pattern and first window of text
        for (let i = 0; i < m; i++) {
            patternHash = (d * patternHash + pattern.charCodeAt(i)) % prime;
            textHash = (d * textHash + text.charCodeAt(i)) % prime;
        }
        
        // Slide the pattern over text one by one
        for (let i = 0; i <= n - m; i++) {
            if (patternHash === textHash) {
                // Check characters one by one
                let j;
                for (j = 0; j < m; j++) {
                    if (text[i + j] !== pattern[j]) break;
                }
                
                if (j === m) {
                    matches.push(i);
                }
            }
            
            // Calculate hash value for next window
            if (i < n - m) {
                textHash = (d * (textHash - text.charCodeAt(i) * h) + text.charCodeAt(i + m)) % prime;
                
                // Convert negative value to positive
                if (textHash < 0) {
                    textHash += prime;
                }
            }
        }
        
        return matches;
    }
}

// Demonstration and testing
function demonstrateAlgorithms() {
    console.log("\n=== Sorting Algorithms ===");
    
    const testArray = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42];
    
    console.log("Original:", testArray);
    console.log("Merge Sort:", SortingAlgorithms.mergeSort([...testArray]));
    console.log("Quick Sort:", SortingAlgorithms.quickSort([...testArray]));
    console.log("Heap Sort:", SortingAlgorithms.heapSort([...testArray]));
    console.log("Radix Sort:", SortingAlgorithms.radixSort([...testArray]));
    
    console.log("\n=== AVL Tree ===");
    
    const avl = new AVLTree();
    [10, 20, 30, 40, 50, 25].forEach(val => avl.insert(val));
    console.log("AVL Tree inorder:", avl.inorderTraversal());
    
    console.log("\n=== Trie ===");
    
    const trie = new Trie();
    ['hello', 'world', 'help', 'heap', 'heat'].forEach(word => trie.insert(word));
    
    console.log("Search 'hello':", trie.search('hello'));
    console.log("Search 'hell':", trie.search('hell'));
    console.log("Starts with 'he':", trie.startsWith('he'));
    console.log("Words with prefix 'he':", trie.getAllWordsWithPrefix('he'));
    
    console.log("\n=== Graph Algorithms ===");
    
    const graph = new Graph(true);
    ['A', 'B', 'C', 'D', 'E', 'F'].forEach(v => graph.addVertex(v));
    graph.addEdge('A', 'B', 4);
    graph.addEdge('A', 'C', 2);
    graph.addEdge('B', 'C', 1);
    graph.addEdge('B', 'D', 5);
    graph.addEdge('C', 'D', 8);
    graph.addEdge('C', 'E', 10);
    graph.addEdge('D', 'E', 2);
    
    console.log("DFS from A:", graph.dfs('A'));
    console.log("BFS from A:", graph.bfs('A'));
    
    const { distances } = graph.dijkstra('A');
    console.log("Shortest distances from A:", Object.fromEntries(distances));
    
    console.log("\n=== Dynamic Programming ===");
    
    const fib = DynamicProgramming.fibonacciMemo();
    console.log("Fibonacci(10):", fib(10));
    console.log("LCS of 'ABCDGH' and 'AEDFHR':", 
        DynamicProgramming.longestCommonSubsequence('ABCDGH', 'AEDFHR'));
    
    const weights = [10, 20, 30];
    const values = [60, 100, 120];
    console.log("Knapsack (capacity=50):", 
        DynamicProgramming.knapsack(50, weights, values, 3));
    
    console.log("Edit distance 'kitten' -> 'sitting':", 
        DynamicProgramming.editDistance('kitten', 'sitting'));
    
    console.log("\n=== String Algorithms ===");
    
    const text = "ABABDABACDABABCABCABCABCABC";
    const pattern = "ABABCABCABCABC";
    
    console.log("KMP matches:", StringAlgorithms.kmpSearch(text, pattern));
    console.log("Rabin-Karp matches:", StringAlgorithms.rabinKarp(text, pattern));
    
    console.log("\nAll algorithm demonstrations completed!");
}

// Run the demonstration
demonstrateAlgorithms();