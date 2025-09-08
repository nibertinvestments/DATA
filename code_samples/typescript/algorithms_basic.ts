// Sample TypeScript code for AI training dataset.
// Demonstrates basic algorithms and patterns with type safety.

/**
 * Implementation of bubble sort algorithm.
 * Time complexity: O(n^2)
 * Space complexity: O(1)
 */
function bubbleSort(arr: number[]): number[] {
    const n = arr.length;
    const result = [...arr]; // Create a copy to avoid mutating original
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (result[j] > result[j + 1]) {
                [result[j], result[j + 1]] = [result[j + 1], result[j]];
            }
        }
    }
    return result;
}

/**
 * Generic bubble sort for any comparable type.
 */
function bubbleSortGeneric<T>(arr: T[], compareFn: (a: T, b: T) => number): T[] {
    const n = arr.length;
    const result = [...arr];
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (compareFn(result[j], result[j + 1]) > 0) {
                [result[j], result[j + 1]] = [result[j + 1], result[j]];
            }
        }
    }
    return result;
}

/**
 * Binary search implementation for sorted arrays.
 * Time complexity: O(log n)
 * Space complexity: O(1)
 */
function binarySearch(arr: number[], target: number): number {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
        const mid = Math.floor(left + (right - left) / 2);
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

/**
 * Generic binary search function.
 */
function binarySearchGeneric<T>(
    arr: T[], 
    target: T, 
    compareFn: (a: T, b: T) => number
): number {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
        const mid = Math.floor(left + (right - left) / 2);
        const comparison = compareFn(arr[mid], target);
        
        if (comparison === 0) {
            return mid;
        } else if (comparison < 0) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

/**
 * Quick sort implementation using recursion.
 * Time complexity: O(n log n) average, O(n^2) worst case
 * Space complexity: O(log n)
 */
function quickSort(arr: number[]): number[] {
    if (arr.length <= 1) {
        return arr;
    }

    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);

    return [...quickSort(left), ...middle, ...quickSort(right)];
}

/**
 * Merge sort implementation.
 * Time complexity: O(n log n)
 * Space complexity: O(n)
 */
function mergeSort(arr: number[]): number[] {
    if (arr.length <= 1) {
        return arr;
    }

    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));

    return merge(left, right);
}

function merge(left: number[], right: number[]): number[] {
    const result: number[] = [];
    let i = 0, j = 0;

    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            result.push(left[i]);
            i++;
        } else {
            result.push(right[j]);
            j++;
        }
    }

    return result.concat(left.slice(i)).concat(right.slice(j));
}

/**
 * Fibonacci sequence implementation using iteration.
 * Time complexity: O(n)
 * Space complexity: O(1)
 */
function fibonacci(n: number): number {
    if (n <= 1) return n;

    let prev = 0;
    let curr = 1;

    for (let i = 2; i <= n; i++) {
        [prev, curr] = [curr, prev + curr];
    }

    return curr;
}

/**
 * Fibonacci with memoization.
 * Time complexity: O(n)
 * Space complexity: O(n)
 */
function fibonacciMemo(n: number, memo: Map<number, number> = new Map()): number {
    if (n <= 1) return n;
    if (memo.has(n)) return memo.get(n)!;

    const result = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo);
    memo.set(n, result);
    return result;
}

/**
 * Greatest Common Divisor using Euclidean algorithm.
 * Time complexity: O(log min(a, b))
 * Space complexity: O(1)
 */
function gcd(a: number, b: number): number {
    while (b !== 0) {
        [a, b] = [b, a % b];
    }
    return a;
}

/**
 * Least Common Multiple.
 */
function lcm(a: number, b: number): number {
    return (a * b) / gcd(a, b);
}

/**
 * Factorial implementation.
 * Time complexity: O(n)
 * Space complexity: O(1)
 */
function factorial(n: number): number {
    if (n < 0) throw new Error("Factorial is not defined for negative numbers");
    if (n <= 1) return 1;

    let result = 1;
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/**
 * Check if a number is prime.
 * Time complexity: O(sqrt(n))
 * Space complexity: O(1)
 */
function isPrime(n: number): boolean {
    if (n < 2) return false;
    if (n === 2) return true;
    if (n % 2 === 0) return false;

    for (let i = 3; i * i <= n; i += 2) {
        if (n % i === 0) return false;
    }
    return true;
}

/**
 * Sieve of Eratosthenes to find all primes up to n.
 * Time complexity: O(n log log n)
 * Space complexity: O(n)
 */
function sieveOfEratosthenes(n: number): number[] {
    if (n < 2) return [];

    const isPrime = new Array(n + 1).fill(true);
    isPrime[0] = isPrime[1] = false;

    for (let i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (let j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }

    return isPrime.map((prime, index) => prime ? index : -1).filter(x => x !== -1);
}

/**
 * Two Sum problem - find two numbers that add up to target.
 * Time complexity: O(n)
 * Space complexity: O(n)
 */
function twoSum(nums: number[], target: number): number[] | null {
    const numMap = new Map<number, number>();

    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (numMap.has(complement)) {
            return [numMap.get(complement)!, i];
        }
        numMap.set(nums[i], i);
    }

    return null;
}

/**
 * Maximum subarray sum (Kadane's algorithm).
 * Time complexity: O(n)
 * Space complexity: O(1)
 */
function maxSubarraySum(nums: number[]): number {
    if (nums.length === 0) return 0;

    let maxSum = nums[0];
    let currentSum = nums[0];

    for (let i = 1; i < nums.length; i++) {
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }

    return maxSum;
}

/**
 * Valid parentheses checker.
 * Time complexity: O(n)
 * Space complexity: O(n)
 */
function isValidParentheses(s: string): boolean {
    const stack: string[] = [];
    const mapping: Record<string, string> = {
        ')': '(',
        '}': '{',
        ']': '['
    };

    for (const char of s) {
        if (char === '(' || char === '{' || char === '[') {
            stack.push(char);
        } else if (char === ')' || char === '}' || char === ']') {
            if (stack.length === 0 || stack.pop() !== mapping[char]) {
                return false;
            }
        }
    }

    return stack.length === 0;
}

/**
 * Reverse a string.
 * Time complexity: O(n)
 * Space complexity: O(n)
 */
function reverseString(s: string): string {
    return s.split('').reverse().join('');
}

/**
 * Check if a string is a palindrome.
 * Time complexity: O(n)
 * Space complexity: O(1)
 */
function isPalindrome(s: string): boolean {
    let left = 0;
    let right = s.length - 1;

    while (left < right) {
        if (s[left] !== s[right]) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}

/**
 * Longest common subsequence.
 * Time complexity: O(m * n)
 * Space complexity: O(m * n)
 */
function longestCommonSubsequence(text1: string, text2: string): number {
    const m = text1.length;
    const n = text2.length;
    const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (text1[i - 1] === text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[m][n];
}

/**
 * Dijkstra's shortest path algorithm.
 * Time complexity: O((V + E) log V)
 * Space complexity: O(V)
 */
interface Graph {
    [key: string]: { [neighbor: string]: number };
}

function dijkstra(graph: Graph, start: string): Record<string, number> {
    const distances: Record<string, number> = {};
    const visited = new Set<string>();
    const priorityQueue: [string, number][] = [];

    // Initialize distances
    for (const vertex in graph) {
        distances[vertex] = vertex === start ? 0 : Infinity;
    }

    priorityQueue.push([start, 0]);

    while (priorityQueue.length > 0) {
        // Sort by distance (in a real implementation, use a proper priority queue)
        priorityQueue.sort((a, b) => a[1] - b[1]);
        const [currentVertex, currentDistance] = priorityQueue.shift()!;

        if (visited.has(currentVertex)) continue;
        visited.add(currentVertex);

        for (const neighbor in graph[currentVertex]) {
            const distance = currentDistance + graph[currentVertex][neighbor];
            if (distance < distances[neighbor]) {
                distances[neighbor] = distance;
                priorityQueue.push([neighbor, distance]);
            }
        }
    }

    return distances;
}

/**
 * Depth-First Search for tree/graph traversal.
 */
interface TreeNode<T> {
    value: T;
    children?: TreeNode<T>[];
}

function dfsTraversal<T>(node: TreeNode<T> | null, visited: Set<T> = new Set()): T[] {
    if (!node || visited.has(node.value)) {
        return [];
    }

    visited.add(node.value);
    const result: T[] = [node.value];

    if (node.children) {
        for (const child of node.children) {
            result.push(...dfsTraversal(child, visited));
        }
    }

    return result;
}

/**
 * Breadth-First Search for tree/graph traversal.
 */
function bfsTraversal<T>(root: TreeNode<T> | null): T[] {
    if (!root) return [];

    const result: T[] = [];
    const queue: TreeNode<T>[] = [root];

    while (queue.length > 0) {
        const node = queue.shift()!;
        result.push(node.value);

        if (node.children) {
            queue.push(...node.children);
        }
    }

    return result;
}

// Example usage and testing
function runTests(): void {
    console.log("=== Algorithm Tests ===");

    // Test bubble sort
    const arr = [64, 34, 25, 12, 22, 11, 90];
    console.log("Original array:", arr);
    console.log("Bubble sorted:", bubbleSort(arr));

    // Test binary search
    const sortedArr = [1, 3, 5, 7, 9, 11, 13];
    console.log("Binary search for 7:", binarySearch(sortedArr, 7));

    // Test Fibonacci
    console.log("Fibonacci(10):", fibonacci(10));
    console.log("Fibonacci with memo(10):", fibonacciMemo(10));

    // Test prime checking
    console.log("Is 17 prime?", isPrime(17));
    console.log("Primes up to 30:", sieveOfEratosthenes(30));

    // Test two sum
    const nums = [2, 7, 11, 15];
    console.log("Two sum for target 9:", twoSum(nums, 9));

    // Test max subarray
    const subarrayNums = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
    console.log("Max subarray sum:", maxSubarraySum(subarrayNums));

    // Test valid parentheses
    console.log("Valid parentheses '()[]{}':", isValidParentheses("()[]{}")    );

    // Test palindrome
    console.log("Is 'racecar' palindrome?", isPalindrome("racecar"));

    // Test LCS
    console.log("LCS of 'abcde' and 'ace':", longestCommonSubsequence("abcde", "ace"));

    // Test Dijkstra
    const graph: Graph = {
        'A': { 'B': 4, 'C': 2 },
        'B': { 'D': 3 },
        'C': { 'D': 1, 'E': 5 },
        'D': { 'E': 1 },
        'E': {}
    };
    console.log("Dijkstra from A:", dijkstra(graph, 'A'));
}

// Run tests if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    runTests();
}

// Export functions for use in other modules
export {
    bubbleSort,
    bubbleSortGeneric,
    binarySearch,
    binarySearchGeneric,
    quickSort,
    mergeSort,
    fibonacci,
    fibonacciMemo,
    gcd,
    lcm,
    factorial,
    isPrime,
    sieveOfEratosthenes,
    twoSum,
    maxSubarraySum,
    isValidParentheses,
    reverseString,
    isPalindrome,
    longestCommonSubsequence,
    dijkstra,
    dfsTraversal,
    bfsTraversal,
    runTests
};