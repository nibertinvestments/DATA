#!/usr/bin/env python3
"""
Searching Algorithms
Binary, Linear, and Advanced Search Techniques
"""

def linear_search(arr, target):
    """Linear search O(n)."""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def binary_search(arr, target):
    """Binary search O(log n) on sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binary_search_recursive(arr, target, left, right):
    """Recursive binary search."""
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def jump_search(arr, target):
    """Jump search algorithm."""
    n = len(arr)
    step = int(n ** 0.5)
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(n ** 0.5)
        if prev >= n:
            return -1
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

def interpolation_search(arr, target):
    """Interpolation search for uniformly distributed data."""
    low, high = 0, len(arr) - 1
    while low <= high and target >= arr[low] and target <= arr[high]:
        if low == high:
            return low if arr[low] == target else -1
        pos = low + int(((high - low) / (arr[high] - arr[low])) * (target - arr[low]))
        if arr[pos] == target:
            return pos
        if arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    return -1

if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target = 7
    print(f"Array: {arr}")
    print(f"Linear Search for {target}: {linear_search(arr, target)}")
    print(f"Binary Search for {target}: {binary_search(arr, target)}")
    print(f"Jump Search for {target}: {jump_search(arr, target)}")
