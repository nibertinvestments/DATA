"""
Quick Sort Algorithm - Python Implementation

This module provides a comprehensive quick sort implementation with
multiple variations and optimizations, following Python best practices.
"""

from typing import List, TypeVar, Callable, Optional
import random

T = TypeVar('T')


def quicksort_basic(arr: List[T]) -> List[T]:
    """
    Basic quick sort implementation using list comprehensions.
    
    Args:
        arr: List of comparable elements to sort
        
    Returns:
        New sorted list (does not modify original)
        
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(n) due to creating new lists
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # Choose middle element as pivot
    
    # Partition into three parts: less, equal, greater
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort_basic(left) + middle + quicksort_basic(right)


def quicksort_inplace(arr: List[T], low: int = 0, high: Optional[int] = None) -> None:
    """
    In-place quick sort implementation using Lomuto partition scheme.
    
    Args:
        arr: List to sort (modified in place)
        low: Starting index (default 0)
        high: Ending index (default len(arr) - 1)
        
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) due to recursion stack
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pivot_index = _lomuto_partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)


def _lomuto_partition(arr: List[T], low: int, high: int) -> int:
    """
    Lomuto partition scheme: pivot is the last element.
    
    Args:
        arr: Array to partition
        low: Starting index
        high: Ending index
        
    Returns:
        Final position of pivot element
    """
    pivot = arr[high]  # Choose last element as pivot
    i = low - 1        # Index of smaller element
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # Swap elements
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort_randomized(arr: List[T]) -> List[T]:
    """
    Randomized quick sort to avoid worst-case performance on sorted arrays.
    
    Args:
        arr: List of comparable elements to sort
        
    Returns:
        New sorted list
        
    Expected Time Complexity: O(n log n)
    """
    if len(arr) <= 1:
        return arr
    
    # Randomly choose pivot to avoid worst case
    pivot_index = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_index]
    
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort_randomized(left) + middle + quicksort_randomized(right)


def quicksort_three_way(arr: List[T]) -> List[T]:
    """
    Three-way quick sort (Dutch National Flag algorithm).
    Efficient for arrays with many duplicate elements.
    
    Args:
        arr: List of comparable elements to sort
        
    Returns:
        New sorted list
        
    Time Complexity: O(n log n), but O(n) for arrays with many duplicates
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    
    less = []
    equal = []
    greater = []
    
    for element in arr:
        if element < pivot:
            less.append(element)
        elif element == pivot:
            equal.append(element)
        else:
            greater.append(element)
    
    return quicksort_three_way(less) + equal + quicksort_three_way(greater)


def quicksort_hybrid(arr: List[T], threshold: int = 10) -> List[T]:
    """
    Hybrid quick sort that switches to insertion sort for small arrays.
    
    Args:
        arr: List of comparable elements to sort
        threshold: Size below which to use insertion sort
        
    Returns:
        New sorted list
        
    Performance: Faster for real-world data due to insertion sort optimization
    """
    if len(arr) <= threshold:
        return _insertion_sort(arr)
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort_hybrid(left, threshold) + middle + quicksort_hybrid(right, threshold)


def _insertion_sort(arr: List[T]) -> List[T]:
    """
    Simple insertion sort for small arrays.
    
    Args:
        arr: Small list to sort
        
    Returns:
        New sorted list
    """
    result = arr.copy()
    
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        
        result[j + 1] = key
    
    return result


def quicksort_iterative(arr: List[T]) -> None:
    """
    Iterative implementation of quick sort to avoid stack overflow.
    
    Args:
        arr: List to sort (modified in place)
        
    Space Complexity: O(log n) for explicit stack
    """
    if len(arr) <= 1:
        return
    
    # Create stack for storing start and end indices
    stack = [(0, len(arr) - 1)]
    
    while stack:
        low, high = stack.pop()
        
        if low < high:
            # Partition and get pivot index
            pivot_index = _lomuto_partition(arr, low, high)
            
            # Push left and right subarrays to stack
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))


def quicksort_with_key(arr: List[T], key: Callable[[T], any] = None, reverse: bool = False) -> List[T]:
    """
    Quick sort with custom key function and reverse option.
    
    Args:
        arr: List of elements to sort
        key: Function to extract comparison key from each element
        reverse: If True, sort in descending order
        
    Returns:
        New sorted list
        
    Example:
        # Sort strings by length
        quicksort_with_key(['apple', 'pie', 'washington', 'book'], key=len)
        
        # Sort tuples by second element
        quicksort_with_key([(1, 3), (2, 1), (3, 2)], key=lambda x: x[1])
    """
    if len(arr) <= 1:
        return arr
    
    # Default key function
    if key is None:
        key_func = lambda x: x
    else:
        key_func = key
    
    pivot = arr[len(arr) // 2]
    pivot_key = key_func(pivot)
    
    if reverse:
        left = [x for x in arr if key_func(x) > pivot_key]
        middle = [x for x in arr if key_func(x) == pivot_key]
        right = [x for x in arr if key_func(x) < pivot_key]
    else:
        left = [x for x in arr if key_func(x) < pivot_key]
        middle = [x for x in arr if key_func(x) == pivot_key]
        right = [x for x in arr if key_func(x) > pivot_key]
    
    return (quicksort_with_key(left, key, reverse) + 
            middle + 
            quicksort_with_key(right, key, reverse))


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_arrays = [
        [3, 6, 8, 10, 1, 2, 1],
        [1],
        [],
        [5, 5, 5, 5],
        [9, 8, 7, 6, 5, 4, 3, 2, 1],
        list(range(1000, 0, -1))  # Large reverse-sorted array
    ]
    
    print("Quick Sort Algorithm Demonstrations")
    print("=" * 50)
    
    for i, arr in enumerate(test_arrays[:4]):  # Skip large array for display
        print(f"\nTest Case {i + 1}: {arr}")
        
        # Test different implementations
        print(f"Basic:      {quicksort_basic(arr)}")
        
        arr_copy = arr.copy()
        quicksort_inplace(arr_copy)
        print(f"In-place:   {arr_copy}")
        
        print(f"Randomized: {quicksort_randomized(arr)}")
        print(f"Three-way:  {quicksort_three_way(arr)}")
        print(f"Hybrid:     {quicksort_hybrid(arr)}")
    
    # Performance test with large array (use randomized to avoid worst case)
    print("\nPerformance Test (1000 elements):")
    import random
    large_array = list(range(1000))
    random.shuffle(large_array)  # Avoid worst-case scenario
    
    import time
    
    # Test iterative sort (safer for large arrays)
    start_time = time.time()
    test_array = large_array.copy()
    quicksort_iterative(test_array)
    print(f"Iterative sort: {time.time() - start_time:.4f} seconds")
    
    # Test hybrid sort
    start_time = time.time()
    quicksort_hybrid(large_array)
    print(f"Hybrid sort: {time.time() - start_time:.4f} seconds")
    
    # Demonstrate custom key sorting
    print("\nCustom Key Sorting:")
    words = ["apple", "pie", "washington", "book", "a"]
    print(f"Original: {words}")
    print(f"By length: {quicksort_with_key(words, key=len)}")
    print(f"By length (reverse): {quicksort_with_key(words, key=len, reverse=True)}")
    
    # Demonstrate sorting complex objects
    students = [
        {"name": "Alice", "grade": 85},
        {"name": "Bob", "grade": 90},
        {"name": "Charlie", "grade": 78},
        {"name": "Diana", "grade": 92}
    ]
    
    sorted_students = quicksort_with_key(students, key=lambda s: s["grade"], reverse=True)
    print(f"\nStudents by grade (highest first):")
    for student in sorted_students:
        print(f"  {student['name']}: {student['grade']}")