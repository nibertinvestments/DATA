// Go data structures implementation for AI training dataset.
// Demonstrates various data structures and their operations.

package main

import (
	"container/heap"
	"errors"
	"fmt"
)

// Stack implementation using slice
type Stack[T any] struct {
	items []T
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{items: make([]T, 0)}
}

func (s *Stack[T]) Push(item T) {
	s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, error) {
	var zero T
	if len(s.items) == 0 {
		return zero, errors.New("stack is empty")
	}
	index := len(s.items) - 1
	item := s.items[index]
	s.items = s.items[:index]
	return item, nil
}

func (s *Stack[T]) Peek() (T, error) {
	var zero T
	if len(s.items) == 0 {
		return zero, errors.New("stack is empty")
	}
	return s.items[len(s.items)-1], nil
}

func (s *Stack[T]) IsEmpty() bool {
	return len(s.items) == 0
}

func (s *Stack[T]) Size() int {
	return len(s.items)
}

// Queue implementation using slice
type Queue[T any] struct {
	items []T
}

func NewQueue[T any]() *Queue[T] {
	return &Queue[T]{items: make([]T, 0)}
}

func (q *Queue[T]) Enqueue(item T) {
	q.items = append(q.items, item)
}

func (q *Queue[T]) Dequeue() (T, error) {
	var zero T
	if len(q.items) == 0 {
		return zero, errors.New("queue is empty")
	}
	item := q.items[0]
	q.items = q.items[1:]
	return item, nil
}

func (q *Queue[T]) Front() (T, error) {
	var zero T
	if len(q.items) == 0 {
		return zero, errors.New("queue is empty")
	}
	return q.items[0], nil
}

func (q *Queue[T]) IsEmpty() bool {
	return len(q.items) == 0
}

func (q *Queue[T]) Size() int {
	return len(q.items)
}

// Binary Tree Node
type TreeNode[T any] struct {
	Value T
	Left  *TreeNode[T]
	Right *TreeNode[T]
}

func NewTreeNode[T any](value T) *TreeNode[T] {
	return &TreeNode[T]{Value: value}
}

// Binary Search Tree
type BinarySearchTree[T comparable] struct {
	root *TreeNode[T]
	less func(a, b T) bool
}

func NewBST[T comparable](lessFn func(a, b T) bool) *BinarySearchTree[T] {
	return &BinarySearchTree[T]{less: lessFn}
}

func (bst *BinarySearchTree[T]) Insert(value T) {
	bst.root = bst.insertRecursive(bst.root, value)
}

func (bst *BinarySearchTree[T]) insertRecursive(node *TreeNode[T], value T) *TreeNode[T] {
	if node == nil {
		return NewTreeNode(value)
	}

	if bst.less(value, node.Value) {
		node.Left = bst.insertRecursive(node.Left, value)
	} else if value != node.Value {
		node.Right = bst.insertRecursive(node.Right, value)
	}

	return node
}

func (bst *BinarySearchTree[T]) Search(value T) bool {
	return bst.searchRecursive(bst.root, value)
}

func (bst *BinarySearchTree[T]) searchRecursive(node *TreeNode[T], value T) bool {
	if node == nil {
		return false
	}

	if value == node.Value {
		return true
	} else if bst.less(value, node.Value) {
		return bst.searchRecursive(node.Left, value)
	} else {
		return bst.searchRecursive(node.Right, value)
	}
}

func (bst *BinarySearchTree[T]) InorderTraversal() []T {
	var result []T
	bst.inorderRecursive(bst.root, &result)
	return result
}

func (bst *BinarySearchTree[T]) inorderRecursive(node *TreeNode[T], result *[]T) {
	if node != nil {
		bst.inorderRecursive(node.Left, result)
		*result = append(*result, node.Value)
		bst.inorderRecursive(node.Right, result)
	}
}

func (bst *BinarySearchTree[T]) PreorderTraversal() []T {
	var result []T
	bst.preorderRecursive(bst.root, &result)
	return result
}

func (bst *BinarySearchTree[T]) preorderRecursive(node *TreeNode[T], result *[]T) {
	if node != nil {
		*result = append(*result, node.Value)
		bst.preorderRecursive(node.Left, result)
		bst.preorderRecursive(node.Right, result)
	}
}

func (bst *BinarySearchTree[T]) PostorderTraversal() []T {
	var result []T
	bst.postorderRecursive(bst.root, &result)
	return result
}

func (bst *BinarySearchTree[T]) postorderRecursive(node *TreeNode[T], result *[]T) {
	if node != nil {
		bst.postorderRecursive(node.Left, result)
		bst.postorderRecursive(node.Right, result)
		*result = append(*result, node.Value)
	}
}

// Linked List implementation
type ListNode[T any] struct {
	Value T
	Next  *ListNode[T]
}

type LinkedList[T any] struct {
	head *ListNode[T]
	size int
}

func NewLinkedList[T any]() *LinkedList[T] {
	return &LinkedList[T]{}
}

func (ll *LinkedList[T]) Prepend(value T) {
	newNode := &ListNode[T]{Value: value, Next: ll.head}
	ll.head = newNode
	ll.size++
}

func (ll *LinkedList[T]) Append(value T) {
	newNode := &ListNode[T]{Value: value}
	
	if ll.head == nil {
		ll.head = newNode
	} else {
		current := ll.head
		for current.Next != nil {
			current = current.Next
		}
		current.Next = newNode
	}
	ll.size++
}

func (ll *LinkedList[T]) Delete(index int) error {
	if index < 0 || index >= ll.size {
		return errors.New("index out of bounds")
	}

	if index == 0 {
		ll.head = ll.head.Next
		ll.size--
		return nil
	}

	current := ll.head
	for i := 0; i < index-1; i++ {
		current = current.Next
	}
	current.Next = current.Next.Next
	ll.size--
	return nil
}

func (ll *LinkedList[T]) Get(index int) (T, error) {
	var zero T
	if index < 0 || index >= ll.size {
		return zero, errors.New("index out of bounds")
	}

	current := ll.head
	for i := 0; i < index; i++ {
		current = current.Next
	}
	return current.Value, nil
}

func (ll *LinkedList[T]) Size() int {
	return ll.size
}

func (ll *LinkedList[T]) IsEmpty() bool {
	return ll.size == 0
}

func (ll *LinkedList[T]) ToSlice() []T {
	result := make([]T, 0, ll.size)
	current := ll.head
	for current != nil {
		result = append(result, current.Value)
		current = current.Next
	}
	return result
}

// Hash Table implementation with chaining
type HashTable[K comparable, V any] struct {
	buckets [][]KeyValue[K, V]
	size    int
	capacity int
}

type KeyValue[K comparable, V any] struct {
	Key   K
	Value V
}

func NewHashTable[K comparable, V any](capacity int) *HashTable[K, V] {
	return &HashTable[K, V]{
		buckets:  make([][]KeyValue[K, V], capacity),
		capacity: capacity,
	}
}

func (ht *HashTable[K, V]) hash(key K) int {
	// Simple hash function - in practice, use a better one
	return int(fmt.Sprintf("%v", key)[0]) % ht.capacity
}

func (ht *HashTable[K, V]) Put(key K, value V) {
	index := ht.hash(key)
	bucket := ht.buckets[index]

	for i, kv := range bucket {
		if kv.Key == key {
			bucket[i].Value = value
			return
		}
	}

	ht.buckets[index] = append(bucket, KeyValue[K, V]{Key: key, Value: value})
	ht.size++
}

func (ht *HashTable[K, V]) Get(key K) (V, bool) {
	var zero V
	index := ht.hash(key)
	bucket := ht.buckets[index]

	for _, kv := range bucket {
		if kv.Key == key {
			return kv.Value, true
		}
	}

	return zero, false
}

func (ht *HashTable[K, V]) Delete(key K) bool {
	index := ht.hash(key)
	bucket := ht.buckets[index]

	for i, kv := range bucket {
		if kv.Key == key {
			ht.buckets[index] = append(bucket[:i], bucket[i+1:]...)
			ht.size--
			return true
		}
	}

	return false
}

func (ht *HashTable[K, V]) Size() int {
	return ht.size
}

func (ht *HashTable[K, V]) IsEmpty() bool {
	return ht.size == 0
}

// Priority Queue implementation using heap
type PriorityQueue[T any] struct {
	items []T
	less  func(i, j int) bool
}

func NewPriorityQueue[T any](lessFn func(i, j int) bool) *PriorityQueue[T] {
	pq := &PriorityQueue[T]{
		items: make([]T, 0),
		less:  lessFn,
	}
	heap.Init(pq)
	return pq
}

func (pq *PriorityQueue[T]) Len() int {
	return len(pq.items)
}

func (pq *PriorityQueue[T]) Less(i, j int) bool {
	return pq.less(i, j)
}

func (pq *PriorityQueue[T]) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
}

func (pq *PriorityQueue[T]) Push(x interface{}) {
	pq.items = append(pq.items, x.(T))
}

func (pq *PriorityQueue[T]) Pop() interface{} {
	old := pq.items
	n := len(old)
	item := old[n-1]
	pq.items = old[0 : n-1]
	return item
}

func (pq *PriorityQueue[T]) Enqueue(item T) {
	heap.Push(pq, item)
}

func (pq *PriorityQueue[T]) Dequeue() (T, error) {
	var zero T
	if pq.Len() == 0 {
		return zero, errors.New("priority queue is empty")
	}
	return heap.Pop(pq).(T), nil
}

func (pq *PriorityQueue[T]) Peek() (T, error) {
	var zero T
	if pq.Len() == 0 {
		return zero, errors.New("priority queue is empty")
	}
	return pq.items[0], nil
}

func (pq *PriorityQueue[T]) IsEmpty() bool {
	return pq.Len() == 0
}

// Trie (Prefix Tree) implementation
type TrieNode struct {
	children map[rune]*TrieNode
	isEnd    bool
}

type Trie struct {
	root *TrieNode
}

func NewTrie() *Trie {
	return &Trie{
		root: &TrieNode{
			children: make(map[rune]*TrieNode),
		},
	}
}

func (t *Trie) Insert(word string) {
	current := t.root
	for _, char := range word {
		if _, exists := current.children[char]; !exists {
			current.children[char] = &TrieNode{
				children: make(map[rune]*TrieNode),
			}
		}
		current = current.children[char]
	}
	current.isEnd = true
}

func (t *Trie) Search(word string) bool {
	current := t.root
	for _, char := range word {
		if _, exists := current.children[char]; !exists {
			return false
		}
		current = current.children[char]
	}
	return current.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
	current := t.root
	for _, char := range prefix {
		if _, exists := current.children[char]; !exists {
			return false
		}
		current = current.children[char]
	}
	return true
}

// Example usage and testing
func main() {
	// Test Stack
	fmt.Println("=== Stack Test ===")
	stack := NewStack[int]()
	stack.Push(1)
	stack.Push(2)
	stack.Push(3)
	fmt.Printf("Stack size: %d\n", stack.Size())
	
	if val, err := stack.Pop(); err == nil {
		fmt.Printf("Popped: %d\n", val)
	}
	
	if val, err := stack.Peek(); err == nil {
		fmt.Printf("Top: %d\n", val)
	}

	// Test Queue
	fmt.Println("\n=== Queue Test ===")
	queue := NewQueue[string]()
	queue.Enqueue("first")
	queue.Enqueue("second")
	queue.Enqueue("third")
	
	if val, err := queue.Dequeue(); err == nil {
		fmt.Printf("Dequeued: %s\n", val)
	}

	// Test Binary Search Tree
	fmt.Println("\n=== BST Test ===")
	bst := NewBST(func(a, b int) bool { return a < b })
	values := []int{5, 3, 7, 2, 4, 6, 8}
	for _, val := range values {
		bst.Insert(val)
	}
	
	fmt.Printf("Inorder traversal: %v\n", bst.InorderTraversal())
	fmt.Printf("Search 4: %t\n", bst.Search(4))
	fmt.Printf("Search 10: %t\n", bst.Search(10))

	// Test LinkedList
	fmt.Println("\n=== LinkedList Test ===")
	list := NewLinkedList[int]()
	list.Append(1)
	list.Append(2)
	list.Prepend(0)
	fmt.Printf("List: %v\n", list.ToSlice())
	list.Delete(1)
	fmt.Printf("After deleting index 1: %v\n", list.ToSlice())

	// Test HashTable
	fmt.Println("\n=== HashTable Test ===")
	ht := NewHashTable[string, int](10)
	ht.Put("apple", 5)
	ht.Put("banana", 3)
	ht.Put("orange", 8)
	
	if val, found := ht.Get("banana"); found {
		fmt.Printf("banana: %d\n", val)
	}

	// Test Trie
	fmt.Println("\n=== Trie Test ===")
	trie := NewTrie()
	words := []string{"apple", "app", "application", "apply"}
	for _, word := range words {
		trie.Insert(word)
	}
	
	fmt.Printf("Search 'app': %t\n", trie.Search("app"))
	fmt.Printf("Search 'appl': %t\n", trie.Search("appl"))
	fmt.Printf("StartsWith 'app': %t\n", trie.StartsWith("app"))
}