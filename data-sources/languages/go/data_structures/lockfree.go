// Package lockfree provides lock-free data structures using atomic operations
// 
// This implementation demonstrates advanced concurrent programming in Go,
// utilizing atomic operations to create thread-safe data structures without
// traditional mutex locks.
//
// Performance Benefits:
// - No lock contention or blocking
// - Better scalability with multiple goroutines
// - Reduced context switching overhead
// - Cache-friendly operations
//
// Trade-offs:
// - More complex implementation
// - Memory ordering considerations
// - ABA problem mitigation required
//
// Author: AI Training Dataset
// Version: 1.0

package lockfree

import (
	"runtime"
	"sync/atomic"
	"unsafe"
)

// Node represents a node in the lock-free data structures
type Node[T any] struct {
	value T
	next  unsafe.Pointer // *Node[T]
}

// newNode creates a new node with the given value
func newNode[T any](value T) *Node[T] {
	return &Node[T]{
		value: value,
		next:  nil,
	}
}

// getNext atomically loads the next pointer
func (n *Node[T]) getNext() *Node[T] {
	return (*Node[T])(atomic.LoadPointer(&n.next))
}

// setNext atomically stores the next pointer
func (n *Node[T]) setNext(next *Node[T]) {
	atomic.StorePointer(&n.next, unsafe.Pointer(next))
}

// compareAndSwapNext atomically compares and swaps the next pointer
func (n *Node[T]) compareAndSwapNext(old, new *Node[T]) bool {
	return atomic.CompareAndSwapPointer(&n.next, unsafe.Pointer(old), unsafe.Pointer(new))
}

// LockFreeStack implements a lock-free stack using the Treiber stack algorithm
//
// The Treiber stack is a classic lock-free data structure that uses
// compare-and-swap (CAS) operations to maintain consistency.
//
// Time Complexity:
// - Push: O(1) amortized (may retry due to contention)
// - Pop: O(1) amortized (may retry due to contention)
//
// Space Complexity: O(n) where n is the number of elements
type LockFreeStack[T any] struct {
	head unsafe.Pointer // *Node[T]
	size int64
}

// NewLockFreeStack creates a new lock-free stack
func NewLockFreeStack[T any]() *LockFreeStack[T] {
	return &LockFreeStack[T]{
		head: nil,
		size: 0,
	}
}

// Push adds an element to the top of the stack
//
// Uses the Treiber stack algorithm:
// 1. Create new node with value
// 2. Load current head
// 3. Set new node's next to current head
// 4. Try to CAS head from old to new
// 5. Retry on failure (ABA problem is handled by pointer comparison)
func (s *LockFreeStack[T]) Push(value T) {
	newNode := newNode(value)
	
	for {
		head := (*Node[T])(atomic.LoadPointer(&s.head))
		newNode.setNext(head)
		
		if atomic.CompareAndSwapPointer(&s.head, unsafe.Pointer(head), unsafe.Pointer(newNode)) {
			atomic.AddInt64(&s.size, 1)
			return
		}
		
		// Backoff to reduce contention
		runtime.Gosched()
	}
}

// Pop removes and returns the top element from the stack
//
// Returns the value and true if successful, zero value and false if empty
func (s *LockFreeStack[T]) Pop() (T, bool) {
	for {
		head := (*Node[T])(atomic.LoadPointer(&s.head))
		if head == nil {
			var zero T
			return zero, false
		}
		
		next := head.getNext()
		if atomic.CompareAndSwapPointer(&s.head, unsafe.Pointer(head), unsafe.Pointer(next)) {
			atomic.AddInt64(&s.size, -1)
			return head.value, true
		}
		
		// Backoff to reduce contention
		runtime.Gosched()
	}
}

// Size returns the approximate size of the stack
//
// Note: The size is approximate due to concurrent operations
func (s *LockFreeStack[T]) Size() int64 {
	return atomic.LoadInt64(&s.size)
}

// IsEmpty checks if the stack is empty
func (s *LockFreeStack[T]) IsEmpty() bool {
	return atomic.LoadPointer(&s.head) == nil
}

// LockFreeQueue implements a lock-free queue using the Michael & Scott algorithm
//
// This is one of the most well-known lock-free queue algorithms,
// using dummy nodes to simplify the implementation.
//
// Time Complexity:
// - Enqueue: O(1) amortized
// - Dequeue: O(1) amortized
//
// Space Complexity: O(n) where n is the number of elements
type LockFreeQueue[T any] struct {
	head unsafe.Pointer // *Node[T]
	tail unsafe.Pointer // *Node[T]
	size int64
}

// NewLockFreeQueue creates a new lock-free queue
func NewLockFreeQueue[T any]() *LockFreeQueue[T] {
	dummy := newNode[T](*new(T)) // Create dummy node with zero value
	q := &LockFreeQueue[T]{
		head: unsafe.Pointer(dummy),
		tail: unsafe.Pointer(dummy),
		size: 0,
	}
	return q
}

// Enqueue adds an element to the rear of the queue
//
// Michael & Scott algorithm:
// 1. Create new node
// 2. Loop: try to link new node at end of list
// 3. Advance tail pointer
func (q *LockFreeQueue[T]) Enqueue(value T) {
	newNode := newNode(value)
	
	for {
		tail := (*Node[T])(atomic.LoadPointer(&q.tail))
		next := tail.getNext()
		
		// Check if tail is still the last node
		if tail == (*Node[T])(atomic.LoadPointer(&q.tail)) {
			if next == nil {
				// Try to link new node at end of list
				if tail.compareAndSwapNext(nil, newNode) {
					break // Successfully linked
				}
			} else {
				// Tail is lagging, try to advance it
				atomic.CompareAndSwapPointer(&q.tail, unsafe.Pointer(tail), unsafe.Pointer(next))
			}
		}
		
		runtime.Gosched()
	}
	
	// Try to advance tail pointer
	tail := (*Node[T])(atomic.LoadPointer(&q.tail))
	atomic.CompareAndSwapPointer(&q.tail, unsafe.Pointer(tail), unsafe.Pointer(newNode))
	atomic.AddInt64(&q.size, 1)
}

// Dequeue removes and returns the front element from the queue
//
// Returns the value and true if successful, zero value and false if empty
func (q *LockFreeQueue[T]) Dequeue() (T, bool) {
	for {
		head := (*Node[T])(atomic.LoadPointer(&q.head))
		tail := (*Node[T])(atomic.LoadPointer(&q.tail))
		next := head.getNext()
		
		// Check if head is still the first node
		if head == (*Node[T])(atomic.LoadPointer(&q.head)) {
			if head == tail {
				if next == nil {
					// Queue is empty
					var zero T
					return zero, false
				}
				// Tail is lagging, try to advance it
				atomic.CompareAndSwapPointer(&q.tail, unsafe.Pointer(tail), unsafe.Pointer(next))
			} else {
				if next == nil {
					// Inconsistent state, retry
					continue
				}
				
				// Read value before CAS
				value := next.value
				
				// Try to advance head
				if atomic.CompareAndSwapPointer(&q.head, unsafe.Pointer(head), unsafe.Pointer(next)) {
					atomic.AddInt64(&q.size, -1)
					return value, true
				}
			}
		}
		
		runtime.Gosched()
	}
}

// Size returns the approximate size of the queue
func (q *LockFreeQueue[T]) Size() int64 {
	return atomic.LoadInt64(&q.size)
}

// IsEmpty checks if the queue is empty
func (q *LockFreeQueue[T]) IsEmpty() bool {
	head := (*Node[T])(atomic.LoadPointer(&q.head))
	tail := (*Node[T])(atomic.LoadPointer(&q.tail))
	return head == tail && head.getNext() == nil
}

// LockFreeSet implements a lock-free set using a sorted linked list
//
// Based on the Harris-Herlihy algorithm with logical deletion
// using mark bits to handle concurrent deletions safely.
//
// Time Complexity:
// - Insert: O(n) worst case, O(log n) average with good distribution
// - Delete: O(n) worst case, O(log n) average
// - Contains: O(n) worst case, O(log n) average
type LockFreeSet[T comparable] struct {
	head unsafe.Pointer // *Node[T]
	less func(T, T) bool
}

// marked returns a marked pointer (lowest bit set)
func marked[T any](ptr *Node[T]) *Node[T] {
	return (*Node[T])(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) | 1))
}

// unmarked returns an unmarked pointer (lowest bit cleared)
func unmarked[T any](ptr *Node[T]) *Node[T] {
	return (*Node[T])(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) &^ 1))
}

// isMarked checks if a pointer is marked
func isMarked[T any](ptr *Node[T]) bool {
	return uintptr(unsafe.Pointer(ptr))&1 == 1
}

// NewLockFreeSet creates a new lock-free set with a comparison function
func NewLockFreeSet[T comparable](less func(T, T) bool) *LockFreeSet[T] {
	// Create sentinel nodes for head and tail
	var minVal, maxVal T
	head := newNode(minVal)
	tail := newNode(maxVal)
	head.setNext(tail)
	
	return &LockFreeSet[T]{
		head: unsafe.Pointer(head),
		less: less,
	}
}

// search finds the position for a given value
// Returns (predecessor, current) where value should be between them
func (s *LockFreeSet[T]) search(value T) (*Node[T], *Node[T]) {
retry:
	for {
		head := (*Node[T])(atomic.LoadPointer(&s.head))
		pred := head
		curr := pred.getNext()
		
		for {
			if curr == nil {
				break
			}
			
			succ := curr.getNext()
			
			// If current node is marked for deletion
			if isMarked(succ) {
				// Try to physically remove the marked node
				if !pred.compareAndSwapNext(curr, unmarked(succ)) {
					goto retry // Restart if CAS failed
				}
				curr = unmarked(succ)
			} else {
				// Compare values
				if !s.less(curr.value, value) {
					break // Found position
				}
				pred = curr
				curr = succ
			}
		}
		
		return pred, curr
	}
}

// Insert adds a value to the set
// Returns true if inserted, false if already exists
func (s *LockFreeSet[T]) Insert(value T) bool {
	for {
		pred, curr := s.search(value)
		
		// Check if value already exists
		if curr != nil && !s.less(value, curr.value) && !s.less(curr.value, value) {
			return false // Value already exists
		}
		
		// Create new node
		newNode := newNode(value)
		newNode.setNext(curr)
		
		// Try to insert
		if pred.compareAndSwapNext(curr, newNode) {
			return true
		}
		
		runtime.Gosched()
	}
}

// Delete removes a value from the set
// Returns true if deleted, false if not found
func (s *LockFreeSet[T]) Delete(value T) bool {
	for {
		pred, curr := s.search(value)
		
		// Check if value exists
		if curr == nil || s.less(value, curr.value) || s.less(curr.value, value) {
			return false // Value not found
		}
		
		succ := curr.getNext()
		
		// Logically delete by marking
		if !curr.compareAndSwapNext(succ, marked(succ)) {
			continue // Retry if marking failed
		}
		
		// Try to physically delete
		pred.compareAndSwapNext(curr, succ)
		return true
	}
}

// Contains checks if a value exists in the set
func (s *LockFreeSet[T]) Contains(value T) bool {
	curr := (*Node[T])(atomic.LoadPointer(&s.head)).getNext()
	
	for curr != nil && s.less(curr.value, value) {
		curr = unmarked(curr.getNext())
	}
	
	return curr != nil && !s.less(value, curr.value) && !s.less(curr.value, value) && !isMarked(curr.getNext())
}

// ToSlice returns all values in the set as a slice
// Note: This is a snapshot and may not reflect concurrent modifications
func (s *LockFreeSet[T]) ToSlice() []T {
	var result []T
	curr := (*Node[T])(atomic.LoadPointer(&s.head)).getNext()
	
	for curr != nil {
		if !isMarked(curr.getNext()) {
			result = append(result, curr.value)
		}
		curr = unmarked(curr.getNext())
	}
	
	return result
}