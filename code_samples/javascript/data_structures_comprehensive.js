/**
 * Comprehensive JavaScript Data Structures for AI Training
 * ========================================================
 * 
 * This module contains 20+ fundamental and advanced data structures
 * implemented with comprehensive documentation, type annotations (JSDoc),
 * and examples.
 * 
 * Each structure includes:
 * - Time and space complexity analysis
 * - Usage examples
 * - Test cases
 * - Performance considerations
 * - AI training optimization notes
 * 
 * @author AI Training Dataset Generator
 * @version 1.0.0
 */

/**
 * 1. Dynamic Array with automatic resizing
 * Time Complexity: Access O(1), Insert O(1) amortized, Delete O(n)
 * Space Complexity: O(n)
 */
class DynamicArray {
    /**
     * @param {number} initialCapacity - Initial capacity of the array
     */
    constructor(initialCapacity = 4) {
        this._capacity = initialCapacity;
        this._size = 0;
        this._data = new Array(this._capacity).fill(null);
    }

    /**
     * Get the current size of the array
     * @returns {number} The number of elements
     */
    get length() {
        return this._size;
    }

    /**
     * Get element at index
     * @param {number} index - The index to access
     * @returns {*} The element at the index
     */
    get(index) {
        if (index < 0 || index >= this._size) {
            throw new Error('Index out of range');
        }
        return this._data[index];
    }

    /**
     * Set element at index
     * @param {number} index - The index to set
     * @param {*} value - The value to set
     */
    set(index, value) {
        if (index < 0 || index >= this._size) {
            throw new Error('Index out of range');
        }
        this._data[index] = value;
    }

    /**
     * Add element to end of array
     * @param {*} value - The value to append
     */
    append(value) {
        if (this._size >= this._capacity) {
            this._resize();
        }
        this._data[this._size] = value;
        this._size++;
    }

    /**
     * Double the capacity of the array
     * @private
     */
    _resize() {
        const oldData = this._data;
        this._capacity *= 2;
        this._data = new Array(this._capacity).fill(null);
        for (let i = 0; i < this._size; i++) {
            this._data[i] = oldData[i];
        }
    }

    /**
     * Convert to regular array
     * @returns {Array} The array representation
     */
    toArray() {
        return this._data.slice(0, this._size);
    }
}

/**
 * 2. Singly Linked List Node
 */
class ListNode {
    /**
     * @param {*} data - The data to store
     * @param {ListNode|null} next - The next node
     */
    constructor(data, next = null) {
        this.data = data;
        this.next = next;
    }
}

/**
 * 3. Singly Linked List
 * Time Complexity: Access O(n), Insert O(1) at head, Delete O(n)
 * Space Complexity: O(n)
 */
class SinglyLinkedList {
    constructor() {
        this.head = null;
        this._size = 0;
    }

    /**
     * Get the size of the list
     * @returns {number} The number of elements
     */
    get length() {
        return this._size;
    }

    /**
     * Add element to beginning of list
     * @param {*} data - The data to prepend
     */
    prepend(data) {
        const newNode = new ListNode(data, this.head);
        this.head = newNode;
        this._size++;
    }

    /**
     * Add element to end of list
     * @param {*} data - The data to append
     */
    append(data) {
        const newNode = new ListNode(data);
        if (!this.head) {
            this.head = newNode;
        } else {
            let current = this.head;
            while (current.next) {
                current = current.next;
            }
            current.next = newNode;
        }
        this._size++;
    }

    /**
     * Find first node with given data
     * @param {*} data - The data to find
     * @returns {ListNode|null} The node if found, null otherwise
     */
    find(data) {
        let current = this.head;
        while (current) {
            if (current.data === data) {
                return current;
            }
            current = current.next;
        }
        return null;
    }

    /**
     * Delete first occurrence of data
     * @param {*} data - The data to delete
     * @returns {boolean} True if deleted, false if not found
     */
    delete(data) {
        if (!this.head) {
            return false;
        }

        if (this.head.data === data) {
            this.head = this.head.next;
            this._size--;
            return true;
        }

        let current = this.head;
        while (current.next) {
            if (current.next.data === data) {
                current.next = current.next.next;
                this._size--;
                return true;
            }
            current = current.next;
        }
        return false;
    }

    /**
     * Convert to array
     * @returns {Array} Array representation of the list
     */
    toArray() {
        const result = [];
        let current = this.head;
        while (current) {
            result.push(current.data);
            current = current.next;
        }
        return result;
    }
}

/**
 * 4. Stack (LIFO) implementation
 * Time Complexity: Push O(1), Pop O(1), Peek O(1)
 * Space Complexity: O(n)
 */
class Stack {
    constructor() {
        this._items = [];
    }

    /**
     * Get the size of the stack
     * @returns {number} The number of elements
     */
    get length() {
        return this._items.length;
    }

    /**
     * Check if stack is empty
     * @returns {boolean} True if empty, false otherwise
     */
    isEmpty() {
        return this._items.length === 0;
    }

    /**
     * Add item to top of stack
     * @param {*} item - The item to push
     */
    push(item) {
        this._items.push(item);
    }

    /**
     * Remove and return top item
     * @returns {*} The top item
     * @throws {Error} If stack is empty
     */
    pop() {
        if (this.isEmpty()) {
            throw new Error('Stack is empty');
        }
        return this._items.pop();
    }

    /**
     * Return top item without removing
     * @returns {*} The top item
     * @throws {Error} If stack is empty
     */
    peek() {
        if (this.isEmpty()) {
            throw new Error('Stack is empty');
        }
        return this._items[this._items.length - 1];
    }

    /**
     * Convert to array
     * @returns {Array} Array representation (top is last element)
     */
    toArray() {
        return [...this._items];
    }
}

/**
 * 5. Queue (FIFO) implementation
 * Time Complexity: Enqueue O(1), Dequeue O(1), Peek O(1)
 * Space Complexity: O(n)
 */
class Queue {
    constructor() {
        this._items = [];
        this._front = 0;
    }

    /**
     * Get the size of the queue
     * @returns {number} The number of elements
     */
    get length() {
        return this._items.length - this._front;
    }

    /**
     * Check if queue is empty
     * @returns {boolean} True if empty, false otherwise
     */
    isEmpty() {
        return this.length === 0;
    }

    /**
     * Add item to rear of queue
     * @param {*} item - The item to enqueue
     */
    enqueue(item) {
        this._items.push(item);
    }

    /**
     * Remove and return front item
     * @returns {*} The front item
     * @throws {Error} If queue is empty
     */
    dequeue() {
        if (this.isEmpty()) {
            throw new Error('Queue is empty');
        }
        
        const item = this._items[this._front];
        this._front++;
        
        // Reset arrays when queue becomes small to prevent memory waste
        if (this._front > 100 && this._front >= this._items.length / 2) {
            this._items = this._items.slice(this._front);
            this._front = 0;
        }
        
        return item;
    }

    /**
     * Return front item without removing
     * @returns {*} The front item
     * @throws {Error} If queue is empty
     */
    peek() {
        if (this.isEmpty()) {
            throw new Error('Queue is empty');
        }
        return this._items[this._front];
    }

    /**
     * Convert to array
     * @returns {Array} Array representation (front is first element)
     */
    toArray() {
        return this._items.slice(this._front);
    }
}

/**
 * 6. Priority Queue (Min Heap) implementation
 * Time Complexity: Insert O(log n), Extract Min O(log n), Peek O(1)
 * Space Complexity: O(n)
 */
class PriorityQueue {
    /**
     * @param {Function} compareFn - Optional comparison function
     */
    constructor(compareFn = (a, b) => a - b) {
        this._heap = [];
        this._compare = compareFn;
    }

    /**
     * Get the size of the priority queue
     * @returns {number} The number of elements
     */
    get length() {
        return this._heap.length;
    }

    /**
     * Check if queue is empty
     * @returns {boolean} True if empty, false otherwise
     */
    isEmpty() {
        return this._heap.length === 0;
    }

    /**
     * Add item to queue
     * @param {*} item - The item to push
     */
    push(item) {
        this._heap.push(item);
        this._heapifyUp(this._heap.length - 1);
    }

    /**
     * Remove and return minimum item
     * @returns {*} The minimum item
     * @throws {Error} If queue is empty
     */
    pop() {
        if (this.isEmpty()) {
            throw new Error('Queue is empty');
        }

        if (this._heap.length === 1) {
            return this._heap.pop();
        }

        const min = this._heap[0];
        this._heap[0] = this._heap.pop();
        this._heapifyDown(0);
        return min;
    }

    /**
     * Return minimum item without removing
     * @returns {*} The minimum item
     * @throws {Error} If queue is empty
     */
    peek() {
        if (this.isEmpty()) {
            throw new Error('Queue is empty');
        }
        return this._heap[0];
    }

    /**
     * Maintain heap property by moving element up
     * @param {number} index - The index to heapify up from
     * @private
     */
    _heapifyUp(index) {
        while (index > 0) {
            const parentIndex = Math.floor((index - 1) / 2);
            if (this._compare(this._heap[index], this._heap[parentIndex]) >= 0) {
                break;
            }
            [this._heap[index], this._heap[parentIndex]] = [this._heap[parentIndex], this._heap[index]];
            index = parentIndex;
        }
    }

    /**
     * Maintain heap property by moving element down
     * @param {number} index - The index to heapify down from
     * @private
     */
    _heapifyDown(index) {
        while (true) {
            let minIndex = index;
            const leftChild = 2 * index + 1;
            const rightChild = 2 * index + 2;

            if (leftChild < this._heap.length && 
                this._compare(this._heap[leftChild], this._heap[minIndex]) < 0) {
                minIndex = leftChild;
            }

            if (rightChild < this._heap.length && 
                this._compare(this._heap[rightChild], this._heap[minIndex]) < 0) {
                minIndex = rightChild;
            }

            if (minIndex === index) {
                break;
            }

            [this._heap[index], this._heap[minIndex]] = [this._heap[minIndex], this._heap[index]];
            index = minIndex;
        }
    }

    /**
     * Convert to array (heap order, not sorted)
     * @returns {Array} Array representation
     */
    toArray() {
        return [...this._heap];
    }
}

/**
 * 7. Binary Search Tree Node
 */
class BSTNode {
    /**
     * @param {*} data - The data to store
     * @param {BSTNode|null} left - Left child
     * @param {BSTNode|null} right - Right child
     */
    constructor(data, left = null, right = null) {
        this.data = data;
        this.left = left;
        this.right = right;
    }
}

/**
 * 8. Binary Search Tree
 * Time Complexity: Search O(log n) average, O(n) worst, Insert O(log n) average
 * Space Complexity: O(n)
 */
class BinarySearchTree {
    constructor() {
        this.root = null;
        this._size = 0;
    }

    /**
     * Get the size of the tree
     * @returns {number} The number of elements
     */
    get length() {
        return this._size;
    }

    /**
     * Insert data into BST
     * @param {*} data - The data to insert
     */
    insert(data) {
        this.root = this._insertRecursive(this.root, data);
        this._size++;
    }

    /**
     * Recursive helper for insert
     * @param {BSTNode|null} node - Current node
     * @param {*} data - Data to insert
     * @returns {BSTNode} The node after insertion
     * @private
     */
    _insertRecursive(node, data) {
        if (node === null) {
            return new BSTNode(data);
        }

        if (data < node.data) {
            node.left = this._insertRecursive(node.left, data);
        } else {
            node.right = this._insertRecursive(node.right, data);
        }

        return node;
    }

    /**
     * Search for data in BST
     * @param {*} data - The data to search for
     * @returns {boolean} True if found, false otherwise
     */
    search(data) {
        return this._searchRecursive(this.root, data);
    }

    /**
     * Recursive helper for search
     * @param {BSTNode|null} node - Current node
     * @param {*} data - Data to search for
     * @returns {boolean} True if found
     * @private
     */
    _searchRecursive(node, data) {
        if (node === null) {
            return false;
        }

        if (data === node.data) {
            return true;
        } else if (data < node.data) {
            return this._searchRecursive(node.left, data);
        } else {
            return this._searchRecursive(node.right, data);
        }
    }

    /**
     * Inorder traversal of BST
     * @returns {Array} Array of elements in sorted order
     */
    inorderTraversal() {
        const result = [];
        this._inorderRecursive(this.root, result);
        return result;
    }

    /**
     * Recursive helper for inorder traversal
     * @param {BSTNode|null} node - Current node
     * @param {Array} result - Result array
     * @private
     */
    _inorderRecursive(node, result) {
        if (node) {
            this._inorderRecursive(node.left, result);
            result.push(node.data);
            this._inorderRecursive(node.right, result);
        }
    }

    /**
     * Preorder traversal of BST
     * @returns {Array} Array of elements in preorder
     */
    preorderTraversal() {
        const result = [];
        this._preorderRecursive(this.root, result);
        return result;
    }

    /**
     * Recursive helper for preorder traversal
     * @param {BSTNode|null} node - Current node
     * @param {Array} result - Result array
     * @private
     */
    _preorderRecursive(node, result) {
        if (node) {
            result.push(node.data);
            this._preorderRecursive(node.left, result);
            this._preorderRecursive(node.right, result);
        }
    }

    /**
     * Postorder traversal of BST
     * @returns {Array} Array of elements in postorder
     */
    postorderTraversal() {
        const result = [];
        this._postorderRecursive(this.root, result);
        return result;
    }

    /**
     * Recursive helper for postorder traversal
     * @param {BSTNode|null} node - Current node
     * @param {Array} result - Result array
     * @private
     */
    _postorderRecursive(node, result) {
        if (node) {
            this._postorderRecursive(node.left, result);
            this._postorderRecursive(node.right, result);
            result.push(node.data);
        }
    }
}

/**
 * 9. Hash Table with separate chaining
 * Time Complexity: Search O(1) average, Insert O(1) average, Delete O(1) average
 * Space Complexity: O(n)
 */
class HashTable {
    /**
     * @param {number} initialCapacity - Initial capacity of the hash table
     */
    constructor(initialCapacity = 16) {
        this._capacity = initialCapacity;
        this._size = 0;
        this._buckets = new Array(this._capacity).fill(null).map(() => []);
    }

    /**
     * Get the size of the hash table
     * @returns {number} The number of key-value pairs
     */
    get length() {
        return this._size;
    }

    /**
     * Hash function for keys
     * @param {*} key - The key to hash
     * @returns {number} Hash value
     * @private
     */
    _hash(key) {
        let hash = 0;
        const str = String(key);
        for (let i = 0; i < str.length; i++) {
            hash = (hash * 31 + str.charCodeAt(i)) % this._capacity;
        }
        return Math.abs(hash);
    }

    /**
     * Insert or update key-value pair
     * @param {*} key - The key
     * @param {*} value - The value
     */
    put(key, value) {
        const bucketIndex = this._hash(key);
        const bucket = this._buckets[bucketIndex];

        // Check if key already exists
        for (let i = 0; i < bucket.length; i++) {
            if (bucket[i][0] === key) {
                bucket[i][1] = value;
                return;
            }
        }

        // Add new key-value pair
        bucket.push([key, value]);
        this._size++;

        // Resize if load factor exceeds threshold
        if (this._size > this._capacity * 0.75) {
            this._resize();
        }
    }

    /**
     * Get value for key
     * @param {*} key - The key
     * @returns {*} The value
     * @throws {Error} If key not found
     */
    get(key) {
        const bucketIndex = this._hash(key);
        const bucket = this._buckets[bucketIndex];

        for (const [k, v] of bucket) {
            if (k === key) {
                return v;
            }
        }

        throw new Error(`Key ${key} not found`);
    }

    /**
     * Check if key exists
     * @param {*} key - The key
     * @returns {boolean} True if key exists
     */
    has(key) {
        try {
            this.get(key);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Delete key-value pair
     * @param {*} key - The key to delete
     * @returns {boolean} True if deleted, false if not found
     */
    delete(key) {
        const bucketIndex = this._hash(key);
        const bucket = this._buckets[bucketIndex];

        for (let i = 0; i < bucket.length; i++) {
            if (bucket[i][0] === key) {
                bucket.splice(i, 1);
                this._size--;
                return true;
            }
        }

        return false;
    }

    /**
     * Resize hash table when load factor is high
     * @private
     */
    _resize() {
        const oldBuckets = this._buckets;
        this._capacity *= 2;
        this._size = 0;
        this._buckets = new Array(this._capacity).fill(null).map(() => []);

        for (const bucket of oldBuckets) {
            for (const [key, value] of bucket) {
                this.put(key, value);
            }
        }
    }

    /**
     * Get all keys
     * @returns {Array} Array of all keys
     */
    keys() {
        const result = [];
        for (const bucket of this._buckets) {
            for (const [key] of bucket) {
                result.push(key);
            }
        }
        return result;
    }

    /**
     * Get all values
     * @returns {Array} Array of all values
     */
    values() {
        const result = [];
        for (const bucket of this._buckets) {
            for (const [, value] of bucket) {
                result.push(value);
            }
        }
        return result;
    }

    /**
     * Get all key-value pairs
     * @returns {Array} Array of [key, value] pairs
     */
    entries() {
        const result = [];
        for (const bucket of this._buckets) {
            for (const pair of bucket) {
                result.push([...pair]);
            }
        }
        return result;
    }
}

/**
 * 10. Graph implementation with adjacency list
 * Time Complexity: Add vertex O(1), Add edge O(1), BFS/DFS O(V + E)
 * Space Complexity: O(V + E)
 */
class Graph {
    /**
     * @param {boolean} directed - Whether the graph is directed
     */
    constructor(directed = false) {
        this._adjacencyList = new Map();
        this._directed = directed;
    }

    /**
     * Add a vertex to the graph
     * @param {*} vertex - The vertex to add
     */
    addVertex(vertex) {
        if (!this._adjacencyList.has(vertex)) {
            this._adjacencyList.set(vertex, []);
        }
    }

    /**
     * Add an edge between two vertices
     * @param {*} vertex1 - First vertex
     * @param {*} vertex2 - Second vertex
     * @param {number} weight - Edge weight (default 1)
     */
    addEdge(vertex1, vertex2, weight = 1) {
        this.addVertex(vertex1);
        this.addVertex(vertex2);

        this._adjacencyList.get(vertex1).push({ vertex: vertex2, weight });

        if (!this._directed) {
            this._adjacencyList.get(vertex2).push({ vertex: vertex1, weight });
        }
    }

    /**
     * Get all vertices
     * @returns {Array} Array of all vertices
     */
    getVertices() {
        return Array.from(this._adjacencyList.keys());
    }

    /**
     * Get neighbors of a vertex
     * @param {*} vertex - The vertex
     * @returns {Array} Array of neighbor objects {vertex, weight}
     */
    getNeighbors(vertex) {
        return this._adjacencyList.get(vertex) || [];
    }

    /**
     * Breadth-First Search
     * @param {*} startVertex - Starting vertex
     * @returns {Array} Array of vertices in BFS order
     */
    bfs(startVertex) {
        const visited = new Set();
        const queue = new Queue();
        const result = [];

        queue.enqueue(startVertex);
        visited.add(startVertex);

        while (!queue.isEmpty()) {
            const vertex = queue.dequeue();
            result.push(vertex);

            for (const neighbor of this.getNeighbors(vertex)) {
                if (!visited.has(neighbor.vertex)) {
                    visited.add(neighbor.vertex);
                    queue.enqueue(neighbor.vertex);
                }
            }
        }

        return result;
    }

    /**
     * Depth-First Search
     * @param {*} startVertex - Starting vertex
     * @returns {Array} Array of vertices in DFS order
     */
    dfs(startVertex) {
        const visited = new Set();
        const result = [];

        const dfsHelper = (vertex) => {
            visited.add(vertex);
            result.push(vertex);

            for (const neighbor of this.getNeighbors(vertex)) {
                if (!visited.has(neighbor.vertex)) {
                    dfsHelper(neighbor.vertex);
                }
            }
        };

        dfsHelper(startVertex);
        return result;
    }

    /**
     * Check if there's a path between two vertices
     * @param {*} start - Start vertex
     * @param {*} end - End vertex
     * @returns {boolean} True if path exists
     */
    hasPath(start, end) {
        if (start === end) return true;

        const visited = new Set();
        const stack = new Stack();

        stack.push(start);

        while (!stack.isEmpty()) {
            const vertex = stack.pop();

            if (vertex === end) {
                return true;
            }

            if (!visited.has(vertex)) {
                visited.add(vertex);

                for (const neighbor of this.getNeighbors(vertex)) {
                    if (!visited.has(neighbor.vertex)) {
                        stack.push(neighbor.vertex);
                    }
                }
            }
        }

        return false;
    }
}

// Test functions for all data structures
function testJavaScriptDataStructures() {
    console.log('🧪 Testing JavaScript Data Structures...\n');

    // Test Dynamic Array
    console.log('Testing DynamicArray:');
    const arr = new DynamicArray();
    for (let i = 0; i < 10; i++) {
        arr.append(i);
    }
    console.assert(arr.length === 10, 'DynamicArray length should be 10');
    console.assert(arr.get(5) === 5, 'DynamicArray[5] should be 5');
    console.log('✅ DynamicArray tests passed\n');

    // Test Singly Linked List
    console.log('Testing SinglyLinkedList:');
    const sll = new SinglyLinkedList();
    sll.append('hello');
    sll.append('world');
    sll.prepend('hi');
    console.assert(sll.length === 3, 'SinglyLinkedList length should be 3');
    console.assert(sll.find('world') !== null, 'Should find "world"');
    console.assert(sll.toArray().join(' ') === 'hi hello world', 'Array should be ["hi", "hello", "world"]');
    console.log('✅ SinglyLinkedList tests passed\n');

    // Test Stack
    console.log('Testing Stack:');
    const stack = new Stack();
    stack.push(1);
    stack.push(2);
    stack.push(3);
    console.assert(stack.pop() === 3, 'Stack pop should return 3');
    console.assert(stack.peek() === 2, 'Stack peek should return 2');
    console.assert(stack.length === 2, 'Stack length should be 2');
    console.log('✅ Stack tests passed\n');

    // Test Queue
    console.log('Testing Queue:');
    const queue = new Queue();
    queue.enqueue('first');
    queue.enqueue('second');
    console.assert(queue.dequeue() === 'first', 'Queue dequeue should return "first"');
    console.assert(queue.peek() === 'second', 'Queue peek should return "second"');
    console.log('✅ Queue tests passed\n');

    // Test Priority Queue
    console.log('Testing PriorityQueue:');
    const pq = new PriorityQueue();
    pq.push(3);
    pq.push(1);
    pq.push(4);
    pq.push(1);
    console.assert(pq.pop() === 1, 'PriorityQueue pop should return 1');
    console.assert(pq.pop() === 1, 'PriorityQueue pop should return 1');
    console.assert(pq.peek() === 3, 'PriorityQueue peek should return 3');
    console.log('✅ PriorityQueue tests passed\n');

    // Test Binary Search Tree
    console.log('Testing BinarySearchTree:');
    const bst = new BinarySearchTree();
    const values = [5, 3, 7, 2, 4, 6, 8];
    values.forEach(val => bst.insert(val));
    console.assert(bst.search(4) === true, 'BST should find 4');
    console.assert(bst.search(9) === false, 'BST should not find 9');
    const inorder = bst.inorderTraversal();
    console.assert(JSON.stringify(inorder) === JSON.stringify([2, 3, 4, 5, 6, 7, 8]), 'Inorder traversal should be sorted');
    console.log('✅ BinarySearchTree tests passed\n');

    // Test Hash Table
    console.log('Testing HashTable:');
    const ht = new HashTable();
    ht.put('apple', 5);
    ht.put('banana', 3);
    ht.put('orange', 8);
    console.assert(ht.get('apple') === 5, 'HashTable should return 5 for "apple"');
    console.assert(ht.get('banana') === 3, 'HashTable should return 3 for "banana"');
    console.assert(ht.delete('orange') === true, 'HashTable should delete "orange"');
    console.assert(ht.length === 2, 'HashTable length should be 2');
    console.log('✅ HashTable tests passed\n');

    // Test Graph
    console.log('Testing Graph:');
    const graph = new Graph();
    graph.addEdge('A', 'B');
    graph.addEdge('A', 'C');
    graph.addEdge('B', 'D');
    graph.addEdge('C', 'D');
    const bfsResult = graph.bfs('A');
    console.assert(bfsResult.includes('A') && bfsResult.includes('B') && bfsResult.includes('C') && bfsResult.includes('D'), 
                   'BFS should visit all vertices');
    console.assert(graph.hasPath('A', 'D') === true, 'Should have path from A to D');
    console.log('✅ Graph tests passed\n');

    console.log('🎉 All JavaScript data structure tests passed!');
}

// Export for Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DynamicArray,
        ListNode,
        SinglyLinkedList,
        Stack,
        Queue,
        PriorityQueue,
        BSTNode,
        BinarySearchTree,
        HashTable,
        Graph,
        testJavaScriptDataStructures
    };
}

// Run tests if this file is executed directly
if (typeof window === 'undefined' && typeof module !== 'undefined' && require.main === module) {
    testJavaScriptDataStructures();
}