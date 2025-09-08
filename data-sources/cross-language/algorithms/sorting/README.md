# Quick Sort Algorithm Implementations

This directory contains quick sort algorithm implementations across multiple programming languages, demonstrating consistent algorithmic thinking while respecting language-specific conventions.

## Algorithm Overview

Quick Sort is a divide-and-conquer algorithm that:
1. Chooses a 'pivot' element from the array
2. Partitions other elements into two sub-arrays according to whether they are less than or greater than the pivot
3. Recursively sorts the sub-arrays

**Time Complexity**: 
- Average: O(n log n)
- Worst: O(n²)
- Best: O(n log n)

**Space Complexity**: O(log n) due to recursion stack

## Implementation Files

- `quicksort.py` - Python implementation with type hints
- `quicksort.js` - JavaScript implementation with modern ES6+ syntax
- `quicksort.java` - Java implementation with generics
- `quicksort.cpp` - C++ implementation with templates
- `quicksort.go` - Go implementation with slices
- `quicksort.rs` - Rust implementation with ownership patterns

## Common Variations

1. **Pivot Selection**:
   - First element
   - Last element
   - Random element
   - Median-of-three

2. **Partitioning Schemes**:
   - Lomuto partition
   - Hoare partition

3. **Optimizations**:
   - Hybrid with insertion sort for small arrays
   - Three-way partitioning for arrays with many duplicates
   - Iterative implementation to avoid stack overflow