// Sample Go code for AI training dataset.
// Demonstrates basic algorithms and patterns.

package main

import (
	"fmt"
	"sort"
)

// BubbleSort implements bubble sort algorithm.
// Time complexity: O(n^2)
// Space complexity: O(1)
func BubbleSort(arr []int) {
	n := len(arr)
	for i := 0; i < n; i++ {
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}

// BinarySearch implements binary search for sorted arrays.
// Time complexity: O(log n)
// Space complexity: O(1)
func BinarySearch(arr []int, target int) int {
	left, right := 0, len(arr)-1

	for left <= right {
		mid := left + (right-left)/2
		if arr[mid] == target {
			return mid
		} else if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

// QuickSort implements quicksort algorithm.
// Time complexity: O(n log n) average, O(n^2) worst case
// Space complexity: O(log n)
func QuickSort(arr []int) {
	if len(arr) <= 1 {
		return
	}
	quickSortHelper(arr, 0, len(arr)-1)
}

func quickSortHelper(arr []int, low, high int) {
	if low < high {
		pi := partition(arr, low, high)
		quickSortHelper(arr, low, pi-1)
		quickSortHelper(arr, pi+1, high)
	}
}

func partition(arr []int, low, high int) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// MergeSort implements merge sort algorithm.
// Time complexity: O(n log n)
// Space complexity: O(n)
func MergeSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}

	mid := len(arr) / 2
	left := MergeSort(arr[:mid])
	right := MergeSort(arr[mid:])

	return merge(left, right)
}

func merge(left, right []int) []int {
	result := make([]int, 0, len(left)+len(right))
	i, j := 0, 0

	for i < len(left) && j < len(right) {
		if left[i] <= right[j] {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}

	result = append(result, left[i:]...)
	result = append(result, right[j:]...)
	return result
}

// LinearSearch implements linear search algorithm.
// Time complexity: O(n)
// Space complexity: O(1)
func LinearSearch(arr []int, target int) int {
	for i, value := range arr {
		if value == target {
			return i
		}
	}
	return -1
}

// Fibonacci calculates nth Fibonacci number iteratively.
// Time complexity: O(n)
// Space complexity: O(1)
func Fibonacci(n int) int {
	if n <= 1 {
		return n
	}

	prev, curr := 0, 1
	for i := 2; i <= n; i++ {
		prev, curr = curr, prev+curr
	}
	return curr
}

// FibonacciRecursive calculates nth Fibonacci number recursively.
// Time complexity: O(2^n)
// Space complexity: O(n)
func FibonacciRecursive(n int) int {
	if n <= 1 {
		return n
	}
	return FibonacciRecursive(n-1) + FibonacciRecursive(n-2)
}

// FibonacciMemoized calculates nth Fibonacci number with memoization.
// Time complexity: O(n)
// Space complexity: O(n)
func FibonacciMemoized(n int) int {
	memo := make(map[int]int)
	return fibMemoHelper(n, memo)
}

func fibMemoHelper(n int, memo map[int]int) int {
	if n <= 1 {
		return n
	}
	if val, exists := memo[n]; exists {
		return val
	}
	memo[n] = fibMemoHelper(n-1, memo) + fibMemoHelper(n-2, memo)
	return memo[n]
}

// GCD calculates Greatest Common Divisor using Euclidean algorithm.
// Time complexity: O(log min(a, b))
// Space complexity: O(1)
func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// LCM calculates Least Common Multiple.
// Time complexity: O(log min(a, b))
// Space complexity: O(1)
func LCM(a, b int) int {
	return (a * b) / GCD(a, b)
}

// Factorial calculates factorial of n iteratively.
// Time complexity: O(n)
// Space complexity: O(1)
func Factorial(n int) int {
	if n < 0 {
		return -1 // Invalid input
	}
	result := 1
	for i := 2; i <= n; i++ {
		result *= i
	}
	return result
}

// FactorialRecursive calculates factorial of n recursively.
// Time complexity: O(n)
// Space complexity: O(n)
func FactorialRecursive(n int) int {
	if n < 0 {
		return -1 // Invalid input
	}
	if n <= 1 {
		return 1
	}
	return n * FactorialRecursive(n-1)
}

// IsPrime checks if a number is prime.
// Time complexity: O(sqrt(n))
// Space complexity: O(1)
func IsPrime(n int) bool {
	if n < 2 {
		return false
	}
	if n == 2 {
		return true
	}
	if n%2 == 0 {
		return false
	}

	for i := 3; i*i <= n; i += 2 {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// SieveOfEratosthenes finds all prime numbers up to n.
// Time complexity: O(n log log n)
// Space complexity: O(n)
func SieveOfEratosthenes(n int) []int {
	if n < 2 {
		return []int{}
	}

	isPrime := make([]bool, n+1)
	for i := 2; i <= n; i++ {
		isPrime[i] = true
	}

	for i := 2; i*i <= n; i++ {
		if isPrime[i] {
			for j := i * i; j <= n; j += i {
				isPrime[j] = false
			}
		}
	}

	var primes []int
	for i := 2; i <= n; i++ {
		if isPrime[i] {
			primes = append(primes, i)
		}
	}
	return primes
}

// ReverseString reverses a string.
// Time complexity: O(n)
// Space complexity: O(n)
func ReverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

// IsPalindrome checks if a string is a palindrome.
// Time complexity: O(n)
// Space complexity: O(1)
func IsPalindrome(s string) bool {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		if runes[i] != runes[j] {
			return false
		}
	}
	return true
}

// TwoSum finds two numbers in array that add up to target.
// Time complexity: O(n)
// Space complexity: O(n)
func TwoSum(nums []int, target int) []int {
	numMap := make(map[int]int)
	
	for i, num := range nums {
		complement := target - num
		if j, exists := numMap[complement]; exists {
			return []int{j, i}
		}
		numMap[num] = i
	}
	return nil
}

// MaxSubarraySum finds maximum sum of contiguous subarray (Kadane's algorithm).
// Time complexity: O(n)
// Space complexity: O(1)
func MaxSubarraySum(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	
	maxSum := nums[0]
	currentSum := nums[0]
	
	for i := 1; i < len(nums); i++ {
		currentSum = max(nums[i], currentSum+nums[i])
		maxSum = max(maxSum, currentSum)
	}
	
	return maxSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ValidParentheses checks if parentheses are valid.
// Time complexity: O(n)
// Space complexity: O(n)
func ValidParentheses(s string) bool {
	stack := []rune{}
	mapping := map[rune]rune{
		')': '(',
		'}': '{',
		']': '[',
	}
	
	for _, char := range s {
		if char == '(' || char == '{' || char == '[' {
			stack = append(stack, char)
		} else if char == ')' || char == '}' || char == ']' {
			if len(stack) == 0 || stack[len(stack)-1] != mapping[char] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	
	return len(stack) == 0
}

// Example usage and testing
func main() {
	// Test bubble sort
	arr := []int{64, 34, 25, 12, 22, 11, 90}
	fmt.Printf("Original array: %v\n", arr)
	BubbleSort(arr)
	fmt.Printf("Sorted array: %v\n", arr)

	// Test binary search
	sortedArr := []int{1, 3, 5, 7, 9, 11, 13}
	target := 7
	index := BinarySearch(sortedArr, target)
	fmt.Printf("Binary search for %d in %v: index %d\n", target, sortedArr, index)

	// Test Fibonacci
	n := 10
	fmt.Printf("Fibonacci(%d) = %d\n", n, Fibonacci(n))

	// Test prime checking
	num := 17
	fmt.Printf("Is %d prime? %t\n", num, IsPrime(num))

	// Test Sieve of Eratosthenes
	primes := SieveOfEratosthenes(30)
	fmt.Printf("Primes up to 30: %v\n", primes)

	// Test palindrome checking
	testStr := "racecar"
	fmt.Printf("Is '%s' a palindrome? %t\n", testStr, IsPalindrome(testStr))

	// Test Two Sum
	nums := []int{2, 7, 11, 15}
	targetSum := 9
	result := TwoSum(nums, targetSum)
	fmt.Printf("Two sum indices for target %d in %v: %v\n", targetSum, nums, result)

	// Test Maximum Subarray Sum
	subarrayNums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
	maxSum := MaxSubarraySum(subarrayNums)
	fmt.Printf("Maximum subarray sum of %v: %d\n", subarrayNums, maxSum)

	// Test Valid Parentheses
	parentheses := "()[]{}"
	fmt.Printf("Are parentheses '%s' valid? %t\n", parentheses, ValidParentheses(parentheses))
}