// Sample C# code for AI training dataset.
// Demonstrates basic algorithms and patterns with .NET features.

using System;
using System.Collections.Generic;
using System.Linq;

namespace DataStructuresAndAlgorithms
{
    /// <summary>
    /// Collection of basic algorithms for AI training.
    /// </summary>
    public static class BasicAlgorithms
    {
        /// <summary>
        /// Implementation of bubble sort algorithm.
        /// Time complexity: O(n^2)
        /// Space complexity: O(1)
        /// </summary>
        public static void BubbleSort(int[] arr)
        {
            int n = arr.Length;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n - i - 1; j++)
                {
                    if (arr[j] > arr[j + 1])
                    {
                        (arr[j], arr[j + 1]) = (arr[j + 1], arr[j]); // Tuple swap
                    }
                }
            }
        }

        /// <summary>
        /// Generic bubble sort for any comparable type.
        /// </summary>
        public static void BubbleSort<T>(T[] arr) where T : IComparable<T>
        {
            int n = arr.Length;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n - i - 1; j++)
                {
                    if (arr[j].CompareTo(arr[j + 1]) > 0)
                    {
                        (arr[j], arr[j + 1]) = (arr[j + 1], arr[j]);
                    }
                }
            }
        }

        /// <summary>
        /// Binary search implementation for sorted arrays.
        /// Time complexity: O(log n)
        /// Space complexity: O(1)
        /// </summary>
        public static int BinarySearch(int[] arr, int target)
        {
            int left = 0;
            int right = arr.Length - 1;

            while (left <= right)
            {
                int mid = left + (right - left) / 2;
                if (arr[mid] == target)
                    return mid;
                else if (arr[mid] < target)
                    left = mid + 1;
                else
                    right = mid - 1;
            }
            return -1;
        }

        /// <summary>
        /// Quick sort implementation using recursion.
        /// Time complexity: O(n log n) average, O(n^2) worst case
        /// Space complexity: O(log n)
        /// </summary>
        public static void QuickSort(int[] arr, int low = 0, int high = -1)
        {
            if (high == -1) high = arr.Length - 1;
            
            if (low < high)
            {
                int pi = Partition(arr, low, high);
                QuickSort(arr, low, pi - 1);
                QuickSort(arr, pi + 1, high);
            }
        }

        private static int Partition(int[] arr, int low, int high)
        {
            int pivot = arr[high];
            int i = low - 1;

            for (int j = low; j < high; j++)
            {
                if (arr[j] < pivot)
                {
                    i++;
                    (arr[i], arr[j]) = (arr[j], arr[i]);
                }
            }
            (arr[i + 1], arr[high]) = (arr[high], arr[i + 1]);
            return i + 1;
        }

        /// <summary>
        /// Merge sort implementation.
        /// Time complexity: O(n log n)
        /// Space complexity: O(n)
        /// </summary>
        public static int[] MergeSort(int[] arr)
        {
            if (arr.Length <= 1)
                return arr;

            int mid = arr.Length / 2;
            int[] left = MergeSort(arr.Take(mid).ToArray());
            int[] right = MergeSort(arr.Skip(mid).ToArray());

            return Merge(left, right);
        }

        private static int[] Merge(int[] left, int[] right)
        {
            var result = new List<int>();
            int i = 0, j = 0;

            while (i < left.Length && j < right.Length)
            {
                if (left[i] <= right[j])
                    result.Add(left[i++]);
                else
                    result.Add(right[j++]);
            }

            result.AddRange(left.Skip(i));
            result.AddRange(right.Skip(j));
            return result.ToArray();
        }

        /// <summary>
        /// Fibonacci sequence implementation using iteration.
        /// Time complexity: O(n)
        /// Space complexity: O(1)
        /// </summary>
        public static long Fibonacci(int n)
        {
            if (n <= 1) return n;

            long prev = 0, curr = 1;
            for (int i = 2; i <= n; i++)
            {
                (prev, curr) = (curr, prev + curr);
            }
            return curr;
        }

        /// <summary>
        /// Fibonacci with memoization using dictionary.
        /// Time complexity: O(n)
        /// Space complexity: O(n)
        /// </summary>
        public static long FibonacciMemo(int n, Dictionary<int, long> memo = null)
        {
            memo ??= new Dictionary<int, long>();
            
            if (n <= 1) return n;
            if (memo.ContainsKey(n)) return memo[n];

            memo[n] = FibonacciMemo(n - 1, memo) + FibonacciMemo(n - 2, memo);
            return memo[n];
        }

        /// <summary>
        /// Greatest Common Divisor using Euclidean algorithm.
        /// Time complexity: O(log min(a, b))
        /// Space complexity: O(1)
        /// </summary>
        public static int GCD(int a, int b)
        {
            while (b != 0)
            {
                (a, b) = (b, a % b);
            }
            return a;
        }

        /// <summary>
        /// Least Common Multiple.
        /// </summary>
        public static int LCM(int a, int b) => (a * b) / GCD(a, b);

        /// <summary>
        /// Factorial implementation using iteration.
        /// Time complexity: O(n)
        /// Space complexity: O(1)
        /// </summary>
        public static long Factorial(int n)
        {
            if (n < 0) throw new ArgumentException("Factorial is not defined for negative numbers");
            if (n <= 1) return 1;

            long result = 1;
            for (int i = 2; i <= n; i++)
            {
                result *= i;
            }
            return result;
        }

        /// <summary>
        /// Check if a number is prime.
        /// Time complexity: O(sqrt(n))
        /// Space complexity: O(1)
        /// </summary>
        public static bool IsPrime(int n)
        {
            if (n < 2) return false;
            if (n == 2) return true;
            if (n % 2 == 0) return false;

            for (int i = 3; i * i <= n; i += 2)
            {
                if (n % i == 0) return false;
            }
            return true;
        }

        /// <summary>
        /// Sieve of Eratosthenes to find all primes up to n.
        /// Time complexity: O(n log log n)
        /// Space complexity: O(n)
        /// </summary>
        public static List<int> SieveOfEratosthenes(int n)
        {
            if (n < 2) return new List<int>();

            bool[] isPrime = new bool[n + 1];
            Array.Fill(isPrime, true);
            isPrime[0] = isPrime[1] = false;

            for (int i = 2; i * i <= n; i++)
            {
                if (isPrime[i])
                {
                    for (int j = i * i; j <= n; j += i)
                    {
                        isPrime[j] = false;
                    }
                }
            }

            return isPrime.Select((prime, index) => new { prime, index })
                         .Where(x => x.prime)
                         .Select(x => x.index)
                         .ToList();
        }

        /// <summary>
        /// Two Sum problem - find two numbers that add up to target.
        /// Time complexity: O(n)
        /// Space complexity: O(n)
        /// </summary>
        public static int[] TwoSum(int[] nums, int target)
        {
            var numMap = new Dictionary<int, int>();

            for (int i = 0; i < nums.Length; i++)
            {
                int complement = target - nums[i];
                if (numMap.ContainsKey(complement))
                {
                    return new int[] { numMap[complement], i };
                }
                numMap[nums[i]] = i;
            }

            return null;
        }

        /// <summary>
        /// Maximum subarray sum (Kadane's algorithm).
        /// Time complexity: O(n)
        /// Space complexity: O(1)
        /// </summary>
        public static int MaxSubarraySum(int[] nums)
        {
            if (nums.Length == 0) return 0;

            int maxSum = nums[0];
            int currentSum = nums[0];

            for (int i = 1; i < nums.Length; i++)
            {
                currentSum = Math.Max(nums[i], currentSum + nums[i]);
                maxSum = Math.Max(maxSum, currentSum);
            }

            return maxSum;
        }

        /// <summary>
        /// Valid parentheses checker.
        /// Time complexity: O(n)
        /// Space complexity: O(n)
        /// </summary>
        public static bool IsValidParentheses(string s)
        {
            var stack = new Stack<char>();
            var mapping = new Dictionary<char, char>
            {
                { ')', '(' },
                { '}', '{' },
                { ']', '[' }
            };

            foreach (char c in s)
            {
                if (c == '(' || c == '{' || c == '[')
                {
                    stack.Push(c);
                }
                else if (mapping.ContainsKey(c))
                {
                    if (stack.Count == 0 || stack.Pop() != mapping[c])
                        return false;
                }
            }

            return stack.Count == 0;
        }

        /// <summary>
        /// Reverse a string.
        /// Time complexity: O(n)
        /// Space complexity: O(n)
        /// </summary>
        public static string ReverseString(string s)
        {
            return new string(s.Reverse().ToArray());
        }

        /// <summary>
        /// Check if a string is a palindrome.
        /// Time complexity: O(n)
        /// Space complexity: O(1)
        /// </summary>
        public static bool IsPalindrome(string s)
        {
            int left = 0, right = s.Length - 1;
            while (left < right)
            {
                if (s[left] != s[right])
                    return false;
                left++;
                right--;
            }
            return true;
        }

        /// <summary>
        /// Longest common subsequence.
        /// Time complexity: O(m * n)
        /// Space complexity: O(m * n)
        /// </summary>
        public static int LongestCommonSubsequence(string text1, string text2)
        {
            int m = text1.Length, n = text2.Length;
            int[,] dp = new int[m + 1, n + 1];

            for (int i = 1; i <= m; i++)
            {
                for (int j = 1; j <= n; j++)
                {
                    if (text1[i - 1] == text2[j - 1])
                        dp[i, j] = dp[i - 1, j - 1] + 1;
                    else
                        dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
                }
            }

            return dp[m, n];
        }
    }

    /// <summary>
    /// Extension methods for LINQ-style operations.
    /// </summary>
    public static class AlgorithmExtensions
    {
        /// <summary>
        /// Extension method for bubble sort.
        /// </summary>
        public static T[] BubbleSorted<T>(this T[] source) where T : IComparable<T>
        {
            var result = (T[])source.Clone();
            BasicAlgorithms.BubbleSort(result);
            return result;
        }

        /// <summary>
        /// Extension method for binary search.
        /// </summary>
        public static int BinarySearchIndex(this int[] source, int target)
        {
            return BasicAlgorithms.BinarySearch(source, target);
        }

        /// <summary>
        /// Extension method to check if array is sorted.
        /// </summary>
        public static bool IsSorted<T>(this T[] source) where T : IComparable<T>
        {
            for (int i = 1; i < source.Length; i++)
            {
                if (source[i - 1].CompareTo(source[i]) > 0)
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Example program demonstrating the algorithms.
    /// </summary>
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== C# Algorithm Examples ===");

            // Test bubble sort
            int[] arr = { 64, 34, 25, 12, 22, 11, 90 };
            Console.WriteLine($"Original array: [{string.Join(", ", arr)}]");
            
            var sortedArr = arr.BubbleSorted();
            Console.WriteLine($"Bubble sorted: [{string.Join(", ", sortedArr)}]");

            // Test binary search
            int[] sortedArray = { 1, 3, 5, 7, 9, 11, 13 };
            int target = 7;
            int index = sortedArray.BinarySearchIndex(target);
            Console.WriteLine($"Binary search for {target}: index {index}");

            // Test Fibonacci
            int n = 10;
            Console.WriteLine($"Fibonacci({n}) = {BasicAlgorithms.Fibonacci(n)}");
            Console.WriteLine($"Fibonacci with memo({n}) = {BasicAlgorithms.FibonacciMemo(n)}");

            // Test prime checking
            int num = 17;
            Console.WriteLine($"Is {num} prime? {BasicAlgorithms.IsPrime(num)}");

            // Test Sieve of Eratosthenes
            var primes = BasicAlgorithms.SieveOfEratosthenes(30);
            Console.WriteLine($"Primes up to 30: [{string.Join(", ", primes)}]");

            // Test Two Sum
            int[] nums = { 2, 7, 11, 15 };
            int targetSum = 9;
            var result = BasicAlgorithms.TwoSum(nums, targetSum);
            Console.WriteLine($"Two sum indices for target {targetSum}: [{string.Join(", ", result)}]");

            // Test Maximum Subarray Sum
            int[] subarrayNums = { -2, 1, -3, 4, -1, 2, 1, -5, 4 };
            int maxSum = BasicAlgorithms.MaxSubarraySum(subarrayNums);
            Console.WriteLine($"Maximum subarray sum: {maxSum}");

            // Test Valid Parentheses
            string parentheses = "()[]{}", invalidParentheses = "([)]";
            Console.WriteLine($"Valid parentheses '{parentheses}': {BasicAlgorithms.IsValidParentheses(parentheses)}");
            Console.WriteLine($"Valid parentheses '{invalidParentheses}': {BasicAlgorithms.IsValidParentheses(invalidParentheses)}");

            // Test palindrome
            string testStr = "racecar";
            Console.WriteLine($"Is '{testStr}' a palindrome? {BasicAlgorithms.IsPalindrome(testStr)}");

            // Test LCS
            string text1 = "abcde", text2 = "ace";
            Console.WriteLine($"LCS of '{text1}' and '{text2}': {BasicAlgorithms.LongestCommonSubsequence(text1, text2)}");

            // Test extension methods
            int[] testArray = { 3, 1, 4, 1, 5 };
            Console.WriteLine($"Is array [{string.Join(", ", testArray)}] sorted? {testArray.IsSorted()}");
            var sortedTest = testArray.BubbleSorted();
            Console.WriteLine($"Sorted array: [{string.Join(", ", sortedTest)}]");
            Console.WriteLine($"Is sorted array sorted? {sortedTest.IsSorted()}");
        }
    }
}