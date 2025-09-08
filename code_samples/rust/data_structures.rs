// Rust data structures implementation for AI training dataset.
// Demonstrates various data structures and their operations.

use std::collections::HashMap;
use std::fmt::Display;

/// Generic Stack implementation using Vec.
#[derive(Debug, Clone)]
pub struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    /// Create a new empty stack.
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }

    /// Push an item onto the stack.
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Pop an item from the stack.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    /// Peek at the top item without removing it.
    pub fn peek(&self) -> Option<&T> {
        self.items.last()
    }

    /// Check if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the size of the stack.
    pub fn size(&self) -> usize {
        self.items.len()
    }
}

/// Generic Queue implementation using VecDeque.
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Queue<T> {
    items: VecDeque<T>,
}

impl<T> Queue<T> {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Queue {
            items: VecDeque::new(),
        }
    }

    /// Enqueue an item to the back of the queue.
    pub fn enqueue(&mut self, item: T) {
        self.items.push_back(item);
    }

    /// Dequeue an item from the front of the queue.
    pub fn dequeue(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    /// Peek at the front item without removing it.
    pub fn front(&self) -> Option<&T> {
        self.items.front()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the size of the queue.
    pub fn size(&self) -> usize {
        self.items.len()
    }
}

/// Binary Tree Node structure.
#[derive(Debug, Clone)]
pub struct TreeNode<T> {
    pub value: T,
    pub left: Option<Box<TreeNode<T>>>,
    pub right: Option<Box<TreeNode<T>>>,
}

impl<T> TreeNode<T> {
    /// Create a new tree node.
    pub fn new(value: T) -> Self {
        TreeNode {
            value,
            left: None,
            right: None,
        }
    }

    /// Create a new tree node with children.
    pub fn with_children(
        value: T,
        left: Option<Box<TreeNode<T>>>,
        right: Option<Box<TreeNode<T>>>,
    ) -> Self {
        TreeNode { value, left, right }
    }
}

/// Binary Search Tree implementation.
#[derive(Debug, Clone)]
pub struct BinarySearchTree<T: PartialOrd> {
    root: Option<Box<TreeNode<T>>>,
}

impl<T: PartialOrd> BinarySearchTree<T> {
    /// Create a new empty BST.
    pub fn new() -> Self {
        BinarySearchTree { root: None }
    }

    /// Insert a value into the BST.
    pub fn insert(&mut self, value: T) {
        self.root = Self::insert_recursive(self.root.take(), value);
    }

    fn insert_recursive(node: Option<Box<TreeNode<T>>>, value: T) -> Option<Box<TreeNode<T>>> {
        match node {
            None => Some(Box::new(TreeNode::new(value))),
            Some(mut node) => {
                if value < node.value {
                    node.left = Self::insert_recursive(node.left.take(), value);
                } else if value > node.value {
                    node.right = Self::insert_recursive(node.right.take(), value);
                }
                Some(node)
            }
        }
    }

    /// Search for a value in the BST.
    pub fn search(&self, value: &T) -> bool {
        Self::search_recursive(&self.root, value)
    }

    fn search_recursive(node: &Option<Box<TreeNode<T>>>, value: &T) -> bool {
        match node {
            None => false,
            Some(node) => {
                if *value == node.value {
                    true
                } else if *value < node.value {
                    Self::search_recursive(&node.left, value)
                } else {
                    Self::search_recursive(&node.right, value)
                }
            }
        }
    }

    /// In-order traversal of the BST.
    pub fn inorder_traversal(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::inorder_recursive(&self.root, &mut result);
        result
    }

    fn inorder_recursive<'a>(node: &'a Option<Box<TreeNode<T>>>, result: &mut Vec<&'a T>) {
        if let Some(node) = node {
            Self::inorder_recursive(&node.left, result);
            result.push(&node.value);
            Self::inorder_recursive(&node.right, result);
        }
    }
}

/// Linked List implementation.
#[derive(Debug, Clone)]
struct ListNode<T> {
    value: T,
    next: Option<Box<ListNode<T>>>,
}

#[derive(Debug, Clone)]
pub struct LinkedList<T> {
    head: Option<Box<ListNode<T>>>,
    size: usize,
}

impl<T> LinkedList<T> {
    /// Create a new empty linked list.
    pub fn new() -> Self {
        LinkedList { head: None, size: 0 }
    }

    /// Insert a value at the beginning of the list.
    pub fn prepend(&mut self, value: T) {
        let new_node = Box::new(ListNode {
            value,
            next: self.head.take(),
        });
        self.head = Some(new_node);
        self.size += 1;
    }

    /// Remove and return the first element.
    pub fn pop(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next;
            self.size -= 1;
            node.value
        })
    }

    /// Get the size of the list.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Peek at the first element without removing it.
    pub fn peek(&self) -> Option<&T> {
        self.head.as_ref().map(|node| &node.value)
    }
}

/// Hash Table implementation using chaining for collision resolution.
#[derive(Debug)]
pub struct HashTable<K, V> {
    buckets: Vec<Vec<(K, V)>>,
    size: usize,
    capacity: usize,
}

impl<K: Clone + PartialEq + std::hash::Hash, V: Clone> HashTable<K, V> {
    /// Create a new hash table with given capacity.
    pub fn new(capacity: usize) -> Self {
        let mut buckets = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buckets.push(Vec::new());
        }
        HashTable {
            buckets,
            size: 0,
            capacity,
        }
    }

    /// Hash function to get bucket index.
    fn hash(&self, key: &K) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.capacity
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: K, value: V) {
        let index = self.hash(&key);
        let bucket = &mut self.buckets[index];
        
        for (existing_key, existing_value) in bucket.iter_mut() {
            if *existing_key == key {
                *existing_value = value;
                return;
            }
        }
        
        bucket.push((key, value));
        self.size += 1;
    }

    /// Get a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self.hash(key);
        let bucket = &self.buckets[index];
        
        for (existing_key, value) in bucket {
            if *existing_key == *key {
                return Some(value);
            }
        }
        
        None
    }

    /// Remove a key-value pair.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index = self.hash(key);
        let bucket = &mut self.buckets[index];
        
        for (i, (existing_key, _)) in bucket.iter().enumerate() {
            if *existing_key == *key {
                let (_, value) = bucket.remove(i);
                self.size -= 1;
                return Some(value);
            }
        }
        
        None
    }

    /// Get the number of key-value pairs.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the hash table is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack() {
        let mut stack = Stack::new();
        assert!(stack.is_empty());
        
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        assert_eq!(stack.size(), 3);
        assert_eq!(stack.peek(), Some(&3));
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.size(), 1);
    }

    #[test]
    fn test_queue() {
        let mut queue = Queue::new();
        assert!(queue.is_empty());
        
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);
        
        assert_eq!(queue.size(), 3);
        assert_eq!(queue.front(), Some(&1));
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.size(), 1);
    }

    #[test]
    fn test_binary_search_tree() {
        let mut bst = BinarySearchTree::new();
        bst.insert(5);
        bst.insert(3);
        bst.insert(7);
        bst.insert(2);
        bst.insert(4);
        
        assert!(bst.search(&5));
        assert!(bst.search(&3));
        assert!(!bst.search(&10));
        
        let inorder = bst.inorder_traversal();
        assert_eq!(inorder, vec![&2, &3, &4, &5, &7]);
    }

    #[test]
    fn test_linked_list() {
        let mut list = LinkedList::new();
        assert!(list.is_empty());
        
        list.prepend(1);
        list.prepend(2);
        list.prepend(3);
        
        assert_eq!(list.len(), 3);
        assert_eq!(list.peek(), Some(&3));
        assert_eq!(list.pop(), Some(3));
        assert_eq!(list.pop(), Some(2));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_hash_table() {
        let mut ht = HashTable::new(10);
        assert!(ht.is_empty());
        
        ht.insert("key1", "value1");
        ht.insert("key2", "value2");
        ht.insert("key3", "value3");
        
        assert_eq!(ht.len(), 3);
        assert_eq!(ht.get(&"key1"), Some(&"value1"));
        assert_eq!(ht.get(&"key2"), Some(&"value2"));
        assert_eq!(ht.get(&"nonexistent"), None);
        
        assert_eq!(ht.remove(&"key2"), Some("value2"));
        assert_eq!(ht.len(), 2);
        assert_eq!(ht.get(&"key2"), None);
    }
}