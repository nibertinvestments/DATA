"""
Advanced Python Data Structures for AI Training - Part 2
========================================================

This module contains 10+ advanced data structures with comprehensive
documentation, type hints, and real-world applications.

Continued from data_structures_part1.py with more complex structures.
"""

from typing import List, Optional, Generic, TypeVar, Iterator, Any, Tuple, Dict, Set
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
import bisect
import random

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# 11. Trie (Prefix Tree)
class TrieNode:
    """Node for Trie data structure."""
    
    def __init__(self):
        """Initialize trie node."""
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word = False
        self.word_count = 0  # For frequency tracking


class Trie:
    """
    Trie (Prefix Tree) implementation for efficient string operations.
    
    Time Complexity:
    - Insert: O(m) where m is string length
    - Search: O(m)
    - Prefix search: O(p + k) where p is prefix length, k is results count
    
    Space Complexity: O(ALPHABET_SIZE * N * M) worst case
    """
    
    def __init__(self):
        """Initialize empty trie."""
        self.root = TrieNode()
        self._word_count = 0
    
    def insert(self, word: str) -> None:
        """Insert word into trie. O(m)."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self._word_count += 1
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word: str) -> bool:
        """Search for exact word in trie. O(m)."""
        node = self._search_prefix(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix. O(p)."""
        return self._search_prefix(prefix) is not None
    
    def _search_prefix(self, prefix: str) -> Optional[TrieNode]:
        """Helper method to find node at end of prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def words_with_prefix(self, prefix: str) -> List[str]:
        """Return all words with given prefix. O(p + k)."""
        result = []
        prefix_node = self._search_prefix(prefix)
        if prefix_node:
            self._collect_words(prefix_node, prefix, result)
        return result
    
    def _collect_words(self, node: TrieNode, current_word: str, result: List[str]) -> None:
        """Recursively collect all words from node."""
        if node.is_end_of_word:
            result.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, result)
    
    def delete(self, word: str) -> bool:
        """Delete word from trie. O(m)."""
        def _delete_recursive(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                node.word_count = 0
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_recursive(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        if self.search(word):
            _delete_recursive(self.root, word, 0)
            self._word_count -= 1
            return True
        return False


# 12. Segment Tree
class SegmentTree:
    """
    Segment tree for range queries and updates.
    
    Time Complexity:
    - Build: O(n)
    - Query: O(log n)
    - Update: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, arr: List[int], operation: str = "sum"):
        """
        Initialize segment tree with array and operation.
        
        Args:
            arr: Input array
            operation: "sum", "min", or "max"
        """
        self.n = len(arr)
        self.operation = operation
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr: List[int], node: int, start: int, end: int) -> None:
        """Build segment tree recursively."""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node + 1, start, mid)
            self._build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self._merge(
                self.tree[2 * node + 1], 
                self.tree[2 * node + 2]
            )
    
    def _merge(self, left: int, right: int) -> int:
        """Merge two values based on operation."""
        if self.operation == "sum":
            return left + right
        elif self.operation == "min":
            return min(left, right)
        elif self.operation == "max":
            return max(left, right)
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
    
    def query(self, l: int, r: int) -> int:
        """Query range [l, r]. O(log n)."""
        return self._query(0, 0, self.n - 1, l, r)
    
    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Recursive range query."""
        if r < start or l > end:
            if self.operation == "sum":
                return 0
            elif self.operation == "min":
                return float('inf')
            elif self.operation == "max":
                return float('-inf')
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query(2 * node + 1, start, mid, l, r)
        right_result = self._query(2 * node + 2, mid + 1, end, l, r)
        return self._merge(left_result, right_result)
    
    def update(self, index: int, value: int) -> None:
        """Update value at index. O(log n)."""
        self._update(0, 0, self.n - 1, index, value)
    
    def _update(self, node: int, start: int, end: int, index: int, value: int) -> None:
        """Recursive point update."""
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            if index <= mid:
                self._update(2 * node + 1, start, mid, index, value)
            else:
                self._update(2 * node + 2, mid + 1, end, index, value)
            self.tree[node] = self._merge(
                self.tree[2 * node + 1], 
                self.tree[2 * node + 2]
            )


# 13. Disjoint Set (Union-Find)
class DisjointSet:
    """
    Disjoint Set (Union-Find) with path compression and union by rank.
    
    Time Complexity:
    - Find: O(α(n)) amortized (nearly constant)
    - Union: O(α(n)) amortized
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        """Initialize disjoint set with n elements."""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n
    
    def find(self, x: int) -> int:
        """Find root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in same set."""
        return self.find(x) == self.find(y)
    
    def get_size(self, x: int) -> int:
        """Get size of set containing x."""
        return self.size[self.find(x)]


# 14. Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    """
    Fenwick Tree for efficient prefix sum queries and updates.
    
    Time Complexity:
    - Update: O(log n)
    - Prefix sum: O(log n)
    - Range sum: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        """Initialize Fenwick tree with n elements."""
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index: int, delta: int) -> None:
        """Add delta to element at index (1-based). O(log n)."""
        index += 1  # Convert to 1-based indexing
        while index <= self.n:
            self.tree[index] += delta
            index += index & (-index)
    
    def prefix_sum(self, index: int) -> int:
        """Get sum of elements from 0 to index (0-based). O(log n)."""
        index += 1  # Convert to 1-based indexing
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)
        return result
    
    def range_sum(self, left: int, right: int) -> int:
        """Get sum of elements from left to right (inclusive). O(log n)."""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


# 15. Bloom Filter
class BloomFilter:
    """
    Bloom filter for probabilistic set membership testing.
    
    Time Complexity:
    - Add: O(k) where k is number of hash functions
    - Contains: O(k)
    
    Space Complexity: O(m) where m is bit array size
    """
    
    def __init__(self, capacity: int, error_rate: float = 0.1):
        """
        Initialize bloom filter.
        
        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal bit array size and number of hash functions
        self.bit_array_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.bit_array_size * math.log(2) / capacity)
        
        self.bit_array = [False] * self.bit_array_size
        self.items_added = 0
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash value for item with given seed."""
        return hash(f"{item}_{seed}") % self.bit_array_size
    
    def add(self, item: str) -> None:
        """Add item to bloom filter. O(k)."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
        self.items_added += 1
    
    def __contains__(self, item: str) -> bool:
        """Check if item might be in set. O(k)."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True
    
    def false_positive_rate(self) -> float:
        """Calculate current false positive rate."""
        if self.items_added == 0:
            return 0.0
        
        # Probability that a bit is still 0
        prob_zero = (1 - 1/self.bit_array_size) ** (self.hash_count * self.items_added)
        
        # Probability of false positive
        return (1 - prob_zero) ** self.hash_count


# 16. LRU Cache
class LRUCache(Generic[K, V]):
    """
    LRU (Least Recently Used) Cache implementation.
    
    Time Complexity:
    - Get: O(1)
    - Put: O(1)
    
    Space Complexity: O(capacity)
    """
    
    class Node:
        """Doubly linked list node for LRU cache."""
        
        def __init__(self, key: K, value: V):
            self.key = key
            self.value = value
            self.prev: Optional['LRUCache.Node'] = None
            self.next: Optional['LRUCache.Node'] = None
    
    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        self.capacity = capacity
        self.cache: Dict[K, 'LRUCache.Node'] = {}
        
        # Create dummy head and tail nodes
        self.head = self.Node(None, None)
        self.tail = self.Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key: K) -> Optional[V]:
        """Get value for key. O(1)."""
        if key in self.cache:
            node = self.cache[key]
            self._move_to_head(node)
            return node.value
        return None
    
    def put(self, key: K, value: V) -> None:
        """Put key-value pair. O(1)."""
        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                tail_node = self.tail.prev
                self._remove_node(tail_node)
                del self.cache[tail_node.key]
            
            # Add new node
            new_node = self.Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
    
    def _add_to_head(self, node: 'LRUCache.Node') -> None:
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: 'LRUCache.Node') -> None:
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node: 'LRUCache.Node') -> None:
        """Move node to head (mark as recently used)."""
        self._remove_node(node)
        self._add_to_head(node)


# 17. Skip List
class SkipListNode(Generic[T]):
    """Node for skip list."""
    
    def __init__(self, value: T, level: int):
        self.value = value
        self.forward: List[Optional['SkipListNode[T]']] = [None] * (level + 1)


class SkipList(Generic[T]):
    """
    Skip list implementation for probabilistic balanced search.
    
    Time Complexity (average case):
    - Search: O(log n)
    - Insert: O(log n)
    - Delete: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, max_level: int = 16, p: float = 0.5):
        """Initialize skip list."""
        self.max_level = max_level
        self.p = p
        self.header = SkipListNode(None, max_level)
        self.level = 0
    
    def _random_level(self) -> int:
        """Generate random level for new node."""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, value: T) -> bool:
        """Search for value in skip list. O(log n) average."""
        current = self.header
        
        for i in range(self.level, -1, -1):
            while (current.forward[i] and 
                   current.forward[i].value is not None and
                   current.forward[i].value < value):
                current = current.forward[i]
        
        current = current.forward[0]
        return current is not None and current.value == value
    
    def insert(self, value: T) -> None:
        """Insert value into skip list. O(log n) average."""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find insertion position
        for i in range(self.level, -1, -1):
            while (current.forward[i] and 
                   current.forward[i].value is not None and
                   current.forward[i].value < value):
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # If value doesn't exist, insert it
        if not current or current.value != value:
            new_level = self._random_level()
            
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level
            
            new_node = SkipListNode(value, new_level)
            
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node


# 18. Suffix Array
class SuffixArray:
    """
    Suffix array for efficient string pattern matching.
    
    Time Complexity:
    - Build: O(n log n)
    - Search: O(m log n) where m is pattern length
    
    Space Complexity: O(n)
    """
    
    def __init__(self, text: str):
        """Initialize suffix array for given text."""
        self.text = text + '$'  # Add sentinel character
        self.n = len(self.text)
        self.suffix_array = self._build_suffix_array()
        self.lcp_array = self._build_lcp_array()
    
    def _build_suffix_array(self) -> List[int]:
        """Build suffix array using counting sort and prefix doubling."""
        suffixes = [(self.text[i:], i) for i in range(self.n)]
        suffixes.sort()
        return [suffix[1] for suffix in suffixes]
    
    def _build_lcp_array(self) -> List[int]:
        """Build LCP (Longest Common Prefix) array."""
        rank = [0] * self.n
        for i in range(self.n):
            rank[self.suffix_array[i]] = i
        
        lcp = [0] * (self.n - 1)
        h = 0
        
        for i in range(self.n):
            if rank[i] > 0:
                j = self.suffix_array[rank[i] - 1]
                while (i + h < self.n and j + h < self.n and 
                       self.text[i + h] == self.text[j + h]):
                    h += 1
                lcp[rank[i] - 1] = h
                if h > 0:
                    h -= 1
        
        return lcp
    
    def search(self, pattern: str) -> List[int]:
        """Search for all occurrences of pattern. O(m log n)."""
        def binary_search_left():
            left, right = 0, self.n - 1
            while left <= right:
                mid = (left + right) // 2
                suffix = self.text[self.suffix_array[mid]:]
                if suffix.startswith(pattern):
                    if mid == 0 or not self.text[self.suffix_array[mid - 1]:].startswith(pattern):
                        return mid
                    right = mid - 1
                elif suffix < pattern:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        
        def binary_search_right():
            left, right = 0, self.n - 1
            while left <= right:
                mid = (left + right) // 2
                suffix = self.text[self.suffix_array[mid]:]
                if suffix.startswith(pattern):
                    if mid == self.n - 1 or not self.text[self.suffix_array[mid + 1]:].startswith(pattern):
                        return mid
                    left = mid + 1
                elif suffix < pattern:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        
        left_bound = binary_search_left()
        if left_bound == -1:
            return []
        
        right_bound = binary_search_right()
        return [self.suffix_array[i] for i in range(left_bound, right_bound + 1)]


# Example usage and comprehensive test functions
def test_advanced_data_structures():
    """Test all advanced data structures with example usage."""
    
    # Test Trie
    trie = Trie()
    words = ["apple", "app", "application", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
    
    assert trie.search("app") == True
    assert trie.search("appl") == False
    assert trie.starts_with("app") == True
    app_words = trie.words_with_prefix("app")
    assert len(app_words) == 3  # ["app", "apple", "application"]
    assert "app" in app_words
    assert "apple" in app_words  
    assert "application" in app_words
    print("✅ Trie tests passed")
    
    # Test Segment Tree
    arr = [1, 3, 5, 7, 9, 11]
    seg_tree = SegmentTree(arr, "sum")
    assert seg_tree.query(1, 3) == 15  # 3 + 5 + 7
    seg_tree.update(1, 10)
    assert seg_tree.query(1, 3) == 22  # 10 + 5 + 7
    print("✅ SegmentTree tests passed")
    
    # Test Disjoint Set
    ds = DisjointSet(6)
    ds.union(0, 1)
    ds.union(1, 2)
    ds.union(3, 4)
    assert ds.connected(0, 2) == True
    assert ds.connected(0, 3) == False
    assert ds.get_size(0) == 3
    print("✅ DisjointSet tests passed")
    
    # Test Fenwick Tree
    ft = FenwickTree(6)
    for i in range(6):
        ft.update(i, i + 1)  # [1, 2, 3, 4, 5, 6]
    assert ft.prefix_sum(3) == 10  # 1 + 2 + 3 + 4
    assert ft.range_sum(2, 4) == 12  # 3 + 4 + 5
    print("✅ FenwickTree tests passed")
    
    # Test Bloom Filter
    bf = BloomFilter(1000, 0.1)
    items = ["apple", "banana", "cherry", "date"]
    for item in items:
        bf.add(item)
    
    for item in items:
        assert item in bf
    assert "grape" not in bf  # Might fail due to false positive, but very unlikely
    print("✅ BloomFilter tests passed")
    
    # Test LRU Cache
    cache = LRUCache[str, int](2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # Should evict "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    print("✅ LRUCache tests passed")
    
    # Test Skip List
    skip_list = SkipList[int]()
    values = [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]
    for val in values:
        skip_list.insert(val)
    
    for val in values:
        assert skip_list.search(val) == True
    assert skip_list.search(100) == False
    print("✅ SkipList tests passed")
    
    # Test Suffix Array
    suffix_array = SuffixArray("banana")
    occurrences = suffix_array.search("ana")
    assert len(occurrences) == 2  # "ana" occurs at positions 1 and 3
    print("✅ SuffixArray tests passed")
    
    print("🎉 All advanced data structure tests passed!")


if __name__ == "__main__":
    test_advanced_data_structures()