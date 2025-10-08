#!/usr/bin/env python3
"""
Linked List Implementation
Singly and Doubly Linked Lists for AI Training
"""

class Node:
    """Node for singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """Singly Linked List implementation."""
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, data):
        """Add element to the end."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, data):
        """Add element to the beginning."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, data):
        """Delete first occurrence of data."""
        if not self.head:
            return False
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return True
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        return False
    
    def search(self, data):
        """Search for data in list."""
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return -1
    
    def reverse(self):
        """Reverse the linked list."""
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
    
    def to_list(self):
        """Convert to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return ' -> '.join(map(str, self.to_list()))

class DoublyNode:
    """Node for doubly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    """Doubly Linked List implementation."""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, data):
        """Add element to the end."""
        new_node = DoublyNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def prepend(self, data):
        """Add element to the beginning."""
        new_node = DoublyNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
    
    def delete(self, data):
        """Delete first occurrence of data."""
        current = self.head
        while current:
            if current.data == data:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                self.size -= 1
                return True
            current = current.next
        return False
    
    def __len__(self):
        return self.size

if __name__ == "__main__":
    # Test singly linked list
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print(f"Linked List: {ll}")
    ll.reverse()
    print(f"Reversed: {ll}")
    
    # Test doubly linked list
    dll = DoublyLinkedList()
    dll.append(10)
    dll.append(20)
    dll.prepend(5)
    print(f"Doubly Linked List size: {len(dll)}")
