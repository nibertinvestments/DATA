#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Bubble Sort Algorithm
void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Quick Sort Algorithm
void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    
    return (i + 1);
}

// Merge Sort Algorithm
void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));
    
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    
    free(L);
    free(R);
}

// Binary Search Algorithm
int binary_search(int arr[], int n, int target) {
    int left = 0, right = n - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target)
            return mid;
        
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    
    return -1; // Element not found
}

// Linear Search Algorithm
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target)
            return i;
    }
    return -1; // Element not found
}

// Utility function to print array
void print_array(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Utility function to copy array
void copy_array(int source[], int dest[], int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = source[i];
    }
}

// Function to measure execution time
double measure_time(void (*func)(int[], int), int arr[], int n) {
    clock_t start = clock();
    func(arr, n);
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Special wrapper for quick_sort to match function pointer signature
void quick_sort_wrapper(int arr[], int n) {
    quick_sort(arr, 0, n - 1);
}

void merge_sort_wrapper(int arr[], int n) {
    merge_sort(arr, 0, n - 1);
}

int main() {
    printf("C Sorting and Searching Algorithms\n");
    printf("==================================\n\n");
    
    // Create test array
    int original[] = {64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42};
    int n = sizeof(original) / sizeof(original[0]);
    
    printf("Original array: ");
    print_array(original, n);
    printf("\n");
    
    // Test sorting algorithms
    int test_array[11];
    double execution_time;
    
    // Bubble Sort
    copy_array(original, test_array, n);
    printf("Bubble Sort:\n");
    printf("Before: ");
    print_array(test_array, n);
    execution_time = measure_time(bubble_sort, test_array, n);
    printf("After:  ");
    print_array(test_array, n);
    printf("Time: %.6f seconds\n\n", execution_time);
    
    // Quick Sort
    copy_array(original, test_array, n);
    printf("Quick Sort:\n");
    printf("Before: ");
    print_array(test_array, n);
    execution_time = measure_time(quick_sort_wrapper, test_array, n);
    printf("After:  ");
    print_array(test_array, n);
    printf("Time: %.6f seconds\n\n", execution_time);
    
    // Merge Sort
    copy_array(original, test_array, n);
    printf("Merge Sort:\n");
    printf("Before: ");
    print_array(test_array, n);
    execution_time = measure_time(merge_sort_wrapper, test_array, n);
    printf("After:  ");
    print_array(test_array, n);
    printf("Time: %.6f seconds\n\n", execution_time);
    
    // Test searching algorithms
    int sorted_array[] = {11, 12, 22, 25, 34, 42, 50, 64, 76, 88, 90};
    int target = 42;
    
    printf("Searching for %d in sorted array: ", target);
    print_array(sorted_array, n);
    
    // Linear Search
    clock_t start = clock();
    int linear_result = linear_search(sorted_array, n, target);
    clock_t end = clock();
    double linear_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Linear Search: ");
    if (linear_result != -1) {
        printf("Found at index %d", linear_result);
    } else {
        printf("Not found");
    }
    printf(" (Time: %.6f seconds)\n", linear_time);
    
    // Binary Search
    start = clock();
    int binary_result = binary_search(sorted_array, n, target);
    end = clock();
    double binary_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Binary Search: ");
    if (binary_result != -1) {
        printf("Found at index %d", binary_result);
    } else {
        printf("Not found");
    }
    printf(" (Time: %.6f seconds)\n", binary_time);
    
    printf("\nAll algorithms tested successfully!\n");
    return 0;
}