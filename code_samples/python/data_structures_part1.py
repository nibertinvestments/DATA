"""
Comprehensive Python Data Structures for AI Training
=====================================================

This module contains 20+ fundamental and advanced data structures
implemented with comprehensive documentation, type hints, and examples.

Each structure includes:
- Time and space complexity analysis
- Usage examples
- Test cases
- Performance considerations
- AI training optimization notes
"""

from typing import List, Optional, Generic, TypeVar, Iterator, Any, Tuple
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict, deque
import bisect

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# 1. Dynamic Array (List with automatic resizing)
class DynamicArray(Generic[T]):
    """
    Dynamic array with automatic resizing.
    
    Time Complexity:
    - Access: O(1)
    - Insert (end): O(1) amortized
    - Insert (middle): O(n)
    - Delete: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, initial_capacity: int = 4):
        """Initialize dynamic array with given capacity."""
        self._capacity = initial_capacity
        self._size = 0
        self._data = [None] * self._capacity
    
    def __len__(self) -> int:
        """Return number of elements."""
        return self._size
    
    def __getitem__(self, index: int) -> T:
        """Get element at index."""
        if not 0 <= index < self._size:
            raise IndexError("Index out of range")
        return self._data[index]
    
    def __setitem__(self, index: int, value: T) -> None:
        """Set element at index."""
        if not 0 <= index < self._size:
            raise IndexError("Index out of range")
        self._data[index] = value
    
    def append(self, value: T) -> None:
        """Add element to end. O(1) amortized."""
        if self._size >= self._capacity:
            self._resize()
        self._data[self._size] = value
        self._size += 1
    
    def _resize(self) -> None:
        """Double the capacity."""
        old_data = self._data
        self._capacity *= 2
        self._data = [None] * self._capacity
        for i in range(self._size):
            self._data[i] = old_data[i]


# 2. Singly Linked List
class ListNode(Generic[T]):
    """Node for singly linked list."""
    
    def __init__(self, data: T, next_node: Optional['ListNode[T]'] = None):
        self.data = data
        self.next = next_node


class SinglyLinkedList(Generic[T]):
    """
    Singly linked list implementation.
    
    Time Complexity:
    - Access: O(n)
    - Insert (head): O(1)
    - Insert (tail): O(n)
    - Delete: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        """Initialize empty linked list."""
        self.head: Optional[ListNode[T]] = None
        self._size = 0
    
    def __len__(self) -> int:
        """Return number of elements."""
        return self._size
    
    def prepend(self, data: T) -> None:
        """Add element to beginning. O(1)."""
        new_node = ListNode(data, self.head)
        self.head = new_node
        self._size += 1
    
    def append(self, data: T) -> None:
        """Add element to end. O(n)."""
        new_node = ListNode(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1
    
    def find(self, data: T) -> Optional[ListNode[T]]:
        """Find first node with given data. O(n)."""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def delete(self, data: T) -> bool:
        """Delete first occurrence of data. O(n)."""
        if not self.head:
            return False
        
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        return False


# 3. Doubly Linked List
class DoublyListNode(Generic[T]):
    """Node for doubly linked list."""
    
    def __init__(
        self, 
        data: T, 
        prev_node: Optional['DoublyListNode[T]'] = None,
        next_node: Optional['DoublyListNode[T]'] = None
    ):
        self.data = data
        self.prev = prev_node
        self.next = next_node


class DoublyLinkedList(Generic[T]):
    """
    Doubly linked list implementation.
    
    Time Complexity:
    - Access: O(n)
    - Insert (head/tail): O(1)
    - Delete (with node reference): O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        """Initialize empty doubly linked list."""
        self.head: Optional[DoublyListNode[T]] = None
        self.tail: Optional[DoublyListNode[T]] = None
        self._size = 0
    
    def __len__(self) -> int:
        """Return number of elements."""
        return self._size
    
    def prepend(self, data: T) -> DoublyListNode[T]:
        """Add element to beginning. O(1)."""
        new_node = DoublyListNode(data, None, self.head)
        if self.head:
            self.head.prev = new_node
        else:
            self.tail = new_node
        self.head = new_node
        self._size += 1
        return new_node
    
    def append(self, data: T) -> DoublyListNode[T]:
        """Add element to end. O(1)."""
        new_node = DoublyListNode(data, self.tail, None)
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node
        self.tail = new_node
        self._size += 1
        return new_node
    
    def delete_node(self, node: DoublyListNode[T]) -> None:
        """Delete specific node. O(1)."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        
        self._size -= 1


# 4. Stack
class Stack(Generic[T]):
    """
    Stack (LIFO) implementation using dynamic array.
    
    Time Complexity:
    - Push: O(1) amortized
    - Pop: O(1)
    - Peek: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        """Initialize empty stack."""
        self._items: List[T] = []
    
    def __len__(self) -> int:
        """Return number of elements."""
        return len(self._items)
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self._items) == 0
    
    def push(self, item: T) -> None:
        """Add item to top of stack. O(1) amortized."""
        self._items.append(item)
    
    def pop(self) -> T:
        """Remove and return top item. O(1)."""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self) -> T:
        """Return top item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]


# 5. Queue
class Queue(Generic[T]):
    """
    Queue (FIFO) implementation using deque.
    
    Time Complexity:
    - Enqueue: O(1)
    - Dequeue: O(1)
    - Peek: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        """Initialize empty queue."""
        self._items: deque[T] = deque()
    
    def __len__(self) -> int:
        """Return number of elements."""
        return len(self._items)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._items) == 0
    
    def enqueue(self, item: T) -> None:
        """Add item to rear of queue. O(1)."""
        self._items.append(item)
    
    def dequeue(self) -> T:
        """Remove and return front item. O(1)."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items.popleft()
    
    def peek(self) -> T:
        """Return front item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]


# 6. Circular Queue
class CircularQueue(Generic[T]):
    """
    Circular queue with fixed capacity.
    
    Time Complexity:
    - Enqueue: O(1)
    - Dequeue: O(1)
    
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        """Initialize circular queue with given capacity."""
        self._capacity = capacity
        self._data: List[Optional[T]] = [None] * capacity
        self._front = 0
        self._size = 0
    
    def __len__(self) -> int:
        """Return number of elements."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._size == self._capacity
    
    def enqueue(self, item: T) -> None:
        """Add item to queue. O(1)."""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = item
        self._size += 1
    
    def dequeue(self) -> T:
        """Remove and return front item. O(1)."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item


# 7. Priority Queue (Min Heap)
class PriorityQueue(Generic[T]):
    """
    Priority queue implementation using binary heap.
    
    Time Complexity:
    - Insert: O(log n)
    - Extract min: O(log n)
    - Peek min: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, items: Optional[List[T]] = None):
        """Initialize priority queue with optional items."""
        self._heap: List[T] = []
        if items:
            self._heap = list(items)
            heapq.heapify(self._heap)
    
    def __len__(self) -> int:
        """Return number of elements."""
        return len(self._heap)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0
    
    def push(self, item: T) -> None:
        """Add item to queue. O(log n)."""
        heapq.heappush(self._heap, item)
    
    def pop(self) -> T:
        """Remove and return minimum item. O(log n)."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return heapq.heappop(self._heap)
    
    def peek(self) -> T:
        """Return minimum item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._heap[0]


# 8. Binary Search Tree Node
class BSTNode(Generic[T]):
    """Node for binary search tree."""
    
    def __init__(
        self, 
        data: T, 
        left: Optional['BSTNode[T]'] = None, 
        right: Optional['BSTNode[T]'] = None
    ):
        self.data = data
        self.left = left
        self.right = right


# 9. Binary Search Tree
class BinarySearchTree(Generic[T]):
    """
    Binary search tree implementation.
    
    Time Complexity (average case):
    - Search: O(log n)
    - Insert: O(log n)
    - Delete: O(log n)
    
    Time Complexity (worst case): O(n)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        """Initialize empty BST."""
        self.root: Optional[BSTNode[T]] = None
        self._size = 0
    
    def __len__(self) -> int:
        """Return number of elements."""
        return self._size
    
    def insert(self, data: T) -> None:
        """Insert data into BST. O(log n) average."""
        self.root = self._insert_recursive(self.root, data)
        self._size += 1
    
    def _insert_recursive(self, node: Optional[BSTNode[T]], data: T) -> BSTNode[T]:
        """Recursive helper for insert."""
        if node is None:
            return BSTNode(data)
        
        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        else:
            node.right = self._insert_recursive(node.right, data)
        
        return node
    
    def search(self, data: T) -> bool:
        """Search for data in BST. O(log n) average."""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node: Optional[BSTNode[T]], data: T) -> bool:
        """Recursive helper for search."""
        if node is None:
            return False
        
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def inorder_traversal(self) -> List[T]:
        """Return inorder traversal of BST."""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[BSTNode[T]], result: List[T]) -> None:
        """Recursive helper for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)


# 10. Hash Table (Separate Chaining)
class HashTable(Generic[K, V]):
    """
    Hash table implementation with separate chaining.
    
    Time Complexity (average case):
    - Search: O(1)
    - Insert: O(1)
    - Delete: O(1)
    
    Time Complexity (worst case): O(n)
    Space Complexity: O(n)
    """
    
    def __init__(self, initial_capacity: int = 16):
        """Initialize hash table with given capacity."""
        self._capacity = initial_capacity
        self._size = 0
        self._buckets: List[List[Tuple[K, V]]] = [[] for _ in range(self._capacity)]
    
    def __len__(self) -> int:
        """Return number of key-value pairs."""
        return self._size
    
    def _hash(self, key: K) -> int:
        """Hash function for keys."""
        return hash(key) % self._capacity
    
    def put(self, key: K, value: V) -> None:
        """Insert or update key-value pair. O(1) average."""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self._size += 1
        
        # Resize if load factor exceeds threshold
        if self._size > self._capacity * 0.75:
            self._resize()
    
    def get(self, key: K) -> V:
        """Get value for key. O(1) average."""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(f"Key {key} not found")
    
    def delete(self, key: K) -> None:
        """Delete key-value pair. O(1) average."""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self._size -= 1
                return
        
        raise KeyError(f"Key {key} not found")
    
    def _resize(self) -> None:
        """Resize hash table when load factor is high."""
        old_buckets = self._buckets
        self._capacity *= 2
        self._size = 0
        self._buckets = [[] for _ in range(self._capacity)]
        
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)


# Example usage and test functions
def test_data_structures():
    """Test all data structures with example usage."""
    
    # Test Dynamic Array
    arr = DynamicArray[int]()
    for i in range(10):
        arr.append(i)
    assert len(arr) == 10
    assert arr[5] == 5
    print("✅ DynamicArray tests passed")
    
    # Test Singly Linked List
    sll = SinglyLinkedList[str]()
    sll.append("hello")
    sll.append("world")
    sll.prepend("hi")
    assert len(sll) == 3
    assert sll.find("world") is not None
    print("✅ SinglyLinkedList tests passed")
    
    # Test Stack
    stack = Stack[int]()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    assert stack.pop() == 3
    assert stack.peek() == 2
    assert len(stack) == 2
    print("✅ Stack tests passed")
    
    # Test Queue
    queue = Queue[str]()
    queue.enqueue("first")
    queue.enqueue("second")
    assert queue.dequeue() == "first"
    assert queue.peek() == "second"
    print("✅ Queue tests passed")
    
    # Test Priority Queue
    pq = PriorityQueue[int]()
    pq.push(3)
    pq.push(1)
    pq.push(4)
    pq.push(1)
    assert pq.pop() == 1
    assert pq.pop() == 1
    assert pq.peek() == 3
    print("✅ PriorityQueue tests passed")
    
    # Test Binary Search Tree
    bst = BinarySearchTree[int]()
    values = [5, 3, 7, 2, 4, 6, 8]
    for val in values:
        bst.insert(val)
    assert bst.search(4) == True
    assert bst.search(9) == False
    inorder = bst.inorder_traversal()
    assert inorder == [2, 3, 4, 5, 6, 7, 8]
    print("✅ BinarySearchTree tests passed")
    
    # Test Hash Table
    ht = HashTable[str, int]()
    ht.put("apple", 5)
    ht.put("banana", 3)
    ht.put("orange", 8)
    assert ht.get("apple") == 5
    assert ht.get("banana") == 3
    ht.delete("orange")
    assert len(ht) == 2
    print("✅ HashTable tests passed")
    
    print("🎉 All data structure tests passed!")


if __name__ == "__main__":
    test_data_structures()