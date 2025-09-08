"""
Sample Python code for AI training dataset.
Demonstrates basic algorithms and patterns.
"""


def bubble_sort(arr):
    """
    Implementation of bubble sort algorithm.
    Time complexity: O(n^2)
    Space complexity: O(1)
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def binary_search(arr, target):
    """
    Binary search implementation for sorted arrays.
    Time complexity: O(log n)
    Space complexity: O(1)
    """
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


class Node:
    """Simple linked list node."""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Basic linked list implementation."""

    def __init__(self):
        self.head = None

    def append(self, data):
        """Add a new node to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        """Display all elements in the list."""
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements


if __name__ == "__main__":
    # Test bubble sort
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = bubble_sort(test_arr.copy())
    print(f"Original: {test_arr}")
    print(f"Sorted: {sorted_arr}")

    # Test binary search
    sorted_array = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(sorted_array, target)
    print(f"Binary search for {target}: index {result}")

    # Test linked list
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print(f"Linked list: {ll.display()}")
