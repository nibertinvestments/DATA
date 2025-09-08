// TypeScript data structures implementation for AI training dataset.
// Demonstrates various data structures with strong typing.

/**
 * Generic Stack implementation using array.
 */
class Stack<T> {
    private items: T[] = [];

    push(item: T): void {
        this.items.push(item);
    }

    pop(): T | undefined {
        return this.items.pop();
    }

    peek(): T | undefined {
        return this.items[this.items.length - 1];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }

    clear(): void {
        this.items = [];
    }

    toArray(): T[] {
        return [...this.items];
    }
}

/**
 * Generic Queue implementation using array.
 */
class Queue<T> {
    private items: T[] = [];

    enqueue(item: T): void {
        this.items.push(item);
    }

    dequeue(): T | undefined {
        return this.items.shift();
    }

    front(): T | undefined {
        return this.items[0];
    }

    back(): T | undefined {
        return this.items[this.items.length - 1];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }

    clear(): void {
        this.items = [];
    }

    toArray(): T[] {
        return [...this.items];
    }
}

/**
 * Binary Tree Node class.
 */
class TreeNode<T> {
    value: T;
    left: TreeNode<T> | null = null;
    right: TreeNode<T> | null = null;

    constructor(value: T) {
        this.value = value;
    }
}

/**
 * Binary Search Tree implementation.
 */
class BinarySearchTree<T> {
    private root: TreeNode<T> | null = null;
    private compareFn: (a: T, b: T) => number;

    constructor(compareFn: (a: T, b: T) => number) {
        this.compareFn = compareFn;
    }

    insert(value: T): void {
        this.root = this.insertRecursive(this.root, value);
    }

    private insertRecursive(node: TreeNode<T> | null, value: T): TreeNode<T> {
        if (node === null) {
            return new TreeNode(value);
        }

        const comparison = this.compareFn(value, node.value);
        if (comparison < 0) {
            node.left = this.insertRecursive(node.left, value);
        } else if (comparison > 0) {
            node.right = this.insertRecursive(node.right, value);
        }

        return node;
    }

    search(value: T): boolean {
        return this.searchRecursive(this.root, value);
    }

    private searchRecursive(node: TreeNode<T> | null, value: T): boolean {
        if (node === null) {
            return false;
        }

        const comparison = this.compareFn(value, node.value);
        if (comparison === 0) {
            return true;
        } else if (comparison < 0) {
            return this.searchRecursive(node.left, value);
        } else {
            return this.searchRecursive(node.right, value);
        }
    }

    inorderTraversal(): T[] {
        const result: T[] = [];
        this.inorderRecursive(this.root, result);
        return result;
    }

    private inorderRecursive(node: TreeNode<T> | null, result: T[]): void {
        if (node !== null) {
            this.inorderRecursive(node.left, result);
            result.push(node.value);
            this.inorderRecursive(node.right, result);
        }
    }

    preorderTraversal(): T[] {
        const result: T[] = [];
        this.preorderRecursive(this.root, result);
        return result;
    }

    private preorderRecursive(node: TreeNode<T> | null, result: T[]): void {
        if (node !== null) {
            result.push(node.value);
            this.preorderRecursive(node.left, result);
            this.preorderRecursive(node.right, result);
        }
    }

    postorderTraversal(): T[] {
        const result: T[] = [];
        this.postorderRecursive(this.root, result);
        return result;
    }

    private postorderRecursive(node: TreeNode<T> | null, result: T[]): void {
        if (node !== null) {
            this.postorderRecursive(node.left, result);
            this.postorderRecursive(node.right, result);
            result.push(node.value);
        }
    }

    levelOrderTraversal(): T[] {
        if (this.root === null) return [];

        const result: T[] = [];
        const queue: TreeNode<T>[] = [this.root];

        while (queue.length > 0) {
            const node = queue.shift()!;
            result.push(node.value);

            if (node.left) queue.push(node.left);
            if (node.right) queue.push(node.right);
        }

        return result;
    }
}

/**
 * Linked List Node class.
 */
class ListNode<T> {
    value: T;
    next: ListNode<T> | null = null;

    constructor(value: T) {
        this.value = value;
    }
}

/**
 * Singly Linked List implementation.
 */
class LinkedList<T> {
    private head: ListNode<T> | null = null;
    private size: number = 0;

    prepend(value: T): void {
        const newNode = new ListNode(value);
        newNode.next = this.head;
        this.head = newNode;
        this.size++;
    }

    append(value: T): void {
        const newNode = new ListNode(value);
        
        if (this.head === null) {
            this.head = newNode;
        } else {
            let current = this.head;
            while (current.next !== null) {
                current = current.next;
            }
            current.next = newNode;
        }
        this.size++;
    }

    insertAt(index: number, value: T): boolean {
        if (index < 0 || index > this.size) {
            return false;
        }

        if (index === 0) {
            this.prepend(value);
            return true;
        }

        const newNode = new ListNode(value);
        let current = this.head;
        for (let i = 0; i < index - 1; i++) {
            current = current!.next;
        }

        newNode.next = current!.next;
        current!.next = newNode;
        this.size++;
        return true;
    }

    delete(value: T): boolean {
        if (this.head === null) return false;

        if (this.head.value === value) {
            this.head = this.head.next;
            this.size--;
            return true;
        }

        let current = this.head;
        while (current.next !== null && current.next.value !== value) {
            current = current.next;
        }

        if (current.next !== null) {
            current.next = current.next.next;
            this.size--;
            return true;
        }

        return false;
    }

    deleteAt(index: number): boolean {
        if (index < 0 || index >= this.size) {
            return false;
        }

        if (index === 0) {
            this.head = this.head!.next;
            this.size--;
            return true;
        }

        let current = this.head;
        for (let i = 0; i < index - 1; i++) {
            current = current!.next;
        }

        current!.next = current!.next!.next;
        this.size--;
        return true;
    }

    get(index: number): T | null {
        if (index < 0 || index >= this.size) {
            return null;
        }

        let current = this.head;
        for (let i = 0; i < index; i++) {
            current = current!.next;
        }

        return current!.value;
    }

    indexOf(value: T): number {
        let current = this.head;
        for (let i = 0; i < this.size; i++) {
            if (current!.value === value) {
                return i;
            }
            current = current!.next;
        }
        return -1;
    }

    contains(value: T): boolean {
        return this.indexOf(value) !== -1;
    }

    getSize(): number {
        return this.size;
    }

    isEmpty(): boolean {
        return this.size === 0;
    }

    toArray(): T[] {
        const result: T[] = [];
        let current = this.head;
        while (current !== null) {
            result.push(current.value);
            current = current.next;
        }
        return result;
    }

    clear(): void {
        this.head = null;
        this.size = 0;
    }
}

/**
 * Hash Table implementation with chaining for collision resolution.
 */
class HashTable<K, V> {
    private buckets: Array<Array<{ key: K; value: V }>>;
    private size: number = 0;
    private capacity: number;

    constructor(capacity: number = 16) {
        this.capacity = capacity;
        this.buckets = new Array(capacity);
        for (let i = 0; i < capacity; i++) {
            this.buckets[i] = [];
        }
    }

    private hash(key: K): number {
        const stringKey = String(key);
        let hash = 0;
        for (let i = 0; i < stringKey.length; i++) {
            const char = stringKey.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash) % this.capacity;
    }

    set(key: K, value: V): void {
        const index = this.hash(key);
        const bucket = this.buckets[index];

        for (let i = 0; i < bucket.length; i++) {
            if (bucket[i].key === key) {
                bucket[i].value = value;
                return;
            }
        }

        bucket.push({ key, value });
        this.size++;

        // Resize if load factor exceeds 0.75
        if (this.size > this.capacity * 0.75) {
            this.resize();
        }
    }

    get(key: K): V | undefined {
        const index = this.hash(key);
        const bucket = this.buckets[index];

        for (const item of bucket) {
            if (item.key === key) {
                return item.value;
            }
        }

        return undefined;
    }

    has(key: K): boolean {
        return this.get(key) !== undefined;
    }

    delete(key: K): boolean {
        const index = this.hash(key);
        const bucket = this.buckets[index];

        for (let i = 0; i < bucket.length; i++) {
            if (bucket[i].key === key) {
                bucket.splice(i, 1);
                this.size--;
                return true;
            }
        }

        return false;
    }

    private resize(): void {
        const oldBuckets = this.buckets;
        this.capacity *= 2;
        this.size = 0;
        this.buckets = new Array(this.capacity);
        for (let i = 0; i < this.capacity; i++) {
            this.buckets[i] = [];
        }

        for (const bucket of oldBuckets) {
            for (const item of bucket) {
                this.set(item.key, item.value);
            }
        }
    }

    keys(): K[] {
        const keys: K[] = [];
        for (const bucket of this.buckets) {
            for (const item of bucket) {
                keys.push(item.key);
            }
        }
        return keys;
    }

    values(): V[] {
        const values: V[] = [];
        for (const bucket of this.buckets) {
            for (const item of bucket) {
                values.push(item.value);
            }
        }
        return values;
    }

    entries(): Array<[K, V]> {
        const entries: Array<[K, V]> = [];
        for (const bucket of this.buckets) {
            for (const item of bucket) {
                entries.push([item.key, item.value]);
            }
        }
        return entries;
    }

    getSize(): number {
        return this.size;
    }

    isEmpty(): boolean {
        return this.size === 0;
    }

    clear(): void {
        this.buckets = new Array(this.capacity);
        for (let i = 0; i < this.capacity; i++) {
            this.buckets[i] = [];
        }
        this.size = 0;
    }
}

/**
 * Priority Queue implementation using binary heap.
 */
class PriorityQueue<T> {
    private heap: T[] = [];
    private compare: (a: T, b: T) => number;

    constructor(compareFn: (a: T, b: T) => number) {
        this.compare = compareFn;
    }

    private getParentIndex(index: number): number {
        return Math.floor((index - 1) / 2);
    }

    private getLeftChildIndex(index: number): number {
        return 2 * index + 1;
    }

    private getRightChildIndex(index: number): number {
        return 2 * index + 2;
    }

    private swap(index1: number, index2: number): void {
        [this.heap[index1], this.heap[index2]] = [this.heap[index2], this.heap[index1]];
    }

    private heapifyUp(): void {
        let index = this.heap.length - 1;
        while (index > 0) {
            const parentIndex = this.getParentIndex(index);
            if (this.compare(this.heap[index], this.heap[parentIndex]) >= 0) {
                break;
            }
            this.swap(index, parentIndex);
            index = parentIndex;
        }
    }

    private heapifyDown(): void {
        let index = 0;
        while (this.getLeftChildIndex(index) < this.heap.length) {
            const leftChildIndex = this.getLeftChildIndex(index);
            const rightChildIndex = this.getRightChildIndex(index);
            let smallestIndex = leftChildIndex;

            if (
                rightChildIndex < this.heap.length &&
                this.compare(this.heap[rightChildIndex], this.heap[leftChildIndex]) < 0
            ) {
                smallestIndex = rightChildIndex;
            }

            if (this.compare(this.heap[index], this.heap[smallestIndex]) <= 0) {
                break;
            }

            this.swap(index, smallestIndex);
            index = smallestIndex;
        }
    }

    enqueue(item: T): void {
        this.heap.push(item);
        this.heapifyUp();
    }

    dequeue(): T | undefined {
        if (this.heap.length === 0) return undefined;
        if (this.heap.length === 1) return this.heap.pop();

        const min = this.heap[0];
        this.heap[0] = this.heap.pop()!;
        this.heapifyDown();
        return min;
    }

    peek(): T | undefined {
        return this.heap[0];
    }

    size(): number {
        return this.heap.length;
    }

    isEmpty(): boolean {
        return this.heap.length === 0;
    }
}

/**
 * Trie (Prefix Tree) implementation.
 */
class TrieNode {
    children: Map<string, TrieNode> = new Map();
    isEndOfWord: boolean = false;
}

class Trie {
    private root: TrieNode = new TrieNode();

    insert(word: string): void {
        let current = this.root;
        for (const char of word) {
            if (!current.children.has(char)) {
                current.children.set(char, new TrieNode());
            }
            current = current.children.get(char)!;
        }
        current.isEndOfWord = true;
    }

    search(word: string): boolean {
        let current = this.root;
        for (const char of word) {
            if (!current.children.has(char)) {
                return false;
            }
            current = current.children.get(char)!;
        }
        return current.isEndOfWord;
    }

    startsWith(prefix: string): boolean {
        let current = this.root;
        for (const char of prefix) {
            if (!current.children.has(char)) {
                return false;
            }
            current = current.children.get(char)!;
        }
        return true;
    }

    getAllWordsWithPrefix(prefix: string): string[] {
        let current = this.root;
        for (const char of prefix) {
            if (!current.children.has(char)) {
                return [];
            }
            current = current.children.get(char)!;
        }

        const words: string[] = [];
        this.dfsWords(current, prefix, words);
        return words;
    }

    private dfsWords(node: TrieNode, currentWord: string, words: string[]): void {
        if (node.isEndOfWord) {
            words.push(currentWord);
        }

        for (const [char, childNode] of node.children) {
            this.dfsWords(childNode, currentWord + char, words);
        }
    }
}

/**
 * Graph implementation using adjacency list.
 */
class Graph<T> {
    private adjacencyList: Map<T, Set<T>> = new Map();

    addVertex(vertex: T): void {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, new Set());
        }
    }

    addEdge(vertex1: T, vertex2: T): void {
        this.addVertex(vertex1);
        this.addVertex(vertex2);
        this.adjacencyList.get(vertex1)!.add(vertex2);
        this.adjacencyList.get(vertex2)!.add(vertex1);
    }

    removeEdge(vertex1: T, vertex2: T): void {
        this.adjacencyList.get(vertex1)?.delete(vertex2);
        this.adjacencyList.get(vertex2)?.delete(vertex1);
    }

    removeVertex(vertex: T): void {
        if (this.adjacencyList.has(vertex)) {
            for (const neighbor of this.adjacencyList.get(vertex)!) {
                this.adjacencyList.get(neighbor)!.delete(vertex);
            }
            this.adjacencyList.delete(vertex);
        }
    }

    getNeighbors(vertex: T): T[] {
        return Array.from(this.adjacencyList.get(vertex) || []);
    }

    hasVertex(vertex: T): boolean {
        return this.adjacencyList.has(vertex);
    }

    hasEdge(vertex1: T, vertex2: T): boolean {
        return this.adjacencyList.get(vertex1)?.has(vertex2) || false;
    }

    dfs(startVertex: T): T[] {
        const visited = new Set<T>();
        const result: T[] = [];

        const dfsHelper = (vertex: T) => {
            visited.add(vertex);
            result.push(vertex);

            for (const neighbor of this.adjacencyList.get(vertex) || []) {
                if (!visited.has(neighbor)) {
                    dfsHelper(neighbor);
                }
            }
        };

        dfsHelper(startVertex);
        return result;
    }

    bfs(startVertex: T): T[] {
        const visited = new Set<T>();
        const result: T[] = [];
        const queue: T[] = [startVertex];

        visited.add(startVertex);

        while (queue.length > 0) {
            const vertex = queue.shift()!;
            result.push(vertex);

            for (const neighbor of this.adjacencyList.get(vertex) || []) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        return result;
    }

    getAllVertices(): T[] {
        return Array.from(this.adjacencyList.keys());
    }

    size(): number {
        return this.adjacencyList.size;
    }
}

// Example usage and testing
function runDataStructureTests(): void {
    console.log("=== Data Structure Tests ===");

    // Test Stack
    console.log("\n--- Stack Test ---");
    const stack = new Stack<number>();
    stack.push(1);
    stack.push(2);
    stack.push(3);
    console.log("Stack after pushes:", stack.toArray());
    console.log("Popped:", stack.pop());
    console.log("Peek:", stack.peek());

    // Test Queue
    console.log("\n--- Queue Test ---");
    const queue = new Queue<string>();
    queue.enqueue("first");
    queue.enqueue("second");
    queue.enqueue("third");
    console.log("Queue after enqueues:", queue.toArray());
    console.log("Dequeued:", queue.dequeue());

    // Test BST
    console.log("\n--- BST Test ---");
    const bst = new BinarySearchTree<number>((a, b) => a - b);
    [5, 3, 7, 2, 4, 6, 8].forEach(val => bst.insert(val));
    console.log("Inorder traversal:", bst.inorderTraversal());
    console.log("Search 4:", bst.search(4));

    // Test LinkedList
    console.log("\n--- LinkedList Test ---");
    const list = new LinkedList<number>();
    list.append(1);
    list.append(2);
    list.prepend(0);
    console.log("List:", list.toArray());
    console.log("Delete 1:", list.delete(1));
    console.log("List after delete:", list.toArray());

    // Test HashTable
    console.log("\n--- HashTable Test ---");
    const hashTable = new HashTable<string, number>();
    hashTable.set("apple", 5);
    hashTable.set("banana", 3);
    hashTable.set("orange", 8);
    console.log("Get banana:", hashTable.get("banana"));
    console.log("Keys:", hashTable.keys());

    // Test Trie
    console.log("\n--- Trie Test ---");
    const trie = new Trie();
    ["apple", "app", "application", "apply"].forEach(word => trie.insert(word));
    console.log("Search 'app':", trie.search("app"));
    console.log("StartsWith 'app':", trie.startsWith("app"));
    console.log("Words with prefix 'app':", trie.getAllWordsWithPrefix("app"));

    // Test Graph
    console.log("\n--- Graph Test ---");
    const graph = new Graph<string>();
    ["A", "B", "C", "D"].forEach(vertex => graph.addVertex(vertex));
    graph.addEdge("A", "B");
    graph.addEdge("A", "C");
    graph.addEdge("B", "D");
    console.log("DFS from A:", graph.dfs("A"));
    console.log("BFS from A:", graph.bfs("A"));
}

// Export classes and functions
export {
    Stack,
    Queue,
    TreeNode,
    BinarySearchTree,
    LinkedList,
    HashTable,
    PriorityQueue,
    Trie,
    Graph,
    runDataStructureTests
};