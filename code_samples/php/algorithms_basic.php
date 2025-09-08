<?php
/**
 * Sample PHP code for AI training dataset.
 * Demonstrates basic algorithms and patterns.
 */

class BasicAlgorithms {
    
    /**
     * Implementation of bubble sort algorithm.
     * Time complexity: O(n^2)
     * Space complexity: O(1)
     */
    public static function bubbleSort(array &$arr): void {
        $n = count($arr);
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n - $i - 1; $j++) {
                if ($arr[$j] > $arr[$j + 1]) {
                    // Swap elements
                    $temp = $arr[$j];
                    $arr[$j] = $arr[$j + 1];
                    $arr[$j + 1] = $temp;
                }
            }
        }
    }
    
    /**
     * Binary search implementation for sorted arrays.
     * Time complexity: O(log n)
     * Space complexity: O(1)
     */
    public static function binarySearch(array $arr, $target): int {
        $left = 0;
        $right = count($arr) - 1;
        
        while ($left <= $right) {
            $mid = intval($left + ($right - $left) / 2);
            if ($arr[$mid] == $target) {
                return $mid;
            } elseif ($arr[$mid] < $target) {
                $left = $mid + 1;
            } else {
                $right = $mid - 1;
            }
        }
        return -1;
    }
    
    /**
     * Quick sort implementation using recursion.
     * Time complexity: O(n log n) average, O(n^2) worst case
     * Space complexity: O(log n)
     */
    public static function quickSort(array &$arr, int $low = 0, int $high = -1): void {
        if ($high == -1) {
            $high = count($arr) - 1;
        }
        
        if ($low < $high) {
            $pi = self::partition($arr, $low, $high);
            self::quickSort($arr, $low, $pi - 1);
            self::quickSort($arr, $pi + 1, $high);
        }
    }
    
    private static function partition(array &$arr, int $low, int $high): int {
        $pivot = $arr[$high];
        $i = $low - 1;
        
        for ($j = $low; $j < $high; $j++) {
            if ($arr[$j] < $pivot) {
                $i++;
                $temp = $arr[$i];
                $arr[$i] = $arr[$j];
                $arr[$j] = $temp;
            }
        }
        $temp = $arr[$i + 1];
        $arr[$i + 1] = $arr[$high];
        $arr[$high] = $temp;
        return $i + 1;
    }
    
    /**
     * Merge sort implementation.
     * Time complexity: O(n log n)
     * Space complexity: O(n)
     */
    public static function mergeSort(array $arr): array {
        if (count($arr) <= 1) {
            return $arr;
        }
        
        $mid = intval(count($arr) / 2);
        $left = self::mergeSort(array_slice($arr, 0, $mid));
        $right = self::mergeSort(array_slice($arr, $mid));
        
        return self::merge($left, $right);
    }
    
    private static function merge(array $left, array $right): array {
        $result = [];
        $i = $j = 0;
        
        while ($i < count($left) && $j < count($right)) {
            if ($left[$i] <= $right[$j]) {
                $result[] = $left[$i];
                $i++;
            } else {
                $result[] = $right[$j];
                $j++;
            }
        }
        
        while ($i < count($left)) {
            $result[] = $left[$i];
            $i++;
        }
        
        while ($j < count($right)) {
            $result[] = $right[$j];
            $j++;
        }
        
        return $result;
    }
    
    /**
     * Fibonacci sequence implementation using iteration.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function fibonacci(int $n): int {
        if ($n <= 1) {
            return $n;
        }
        
        $prev = 0;
        $curr = 1;
        
        for ($i = 2; $i <= $n; $i++) {
            $temp = $curr;
            $curr = $prev + $curr;
            $prev = $temp;
        }
        
        return $curr;
    }
    
    /**
     * Fibonacci with memoization.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    public static function fibonacciMemo(int $n, array &$memo = []): int {
        if ($n <= 1) {
            return $n;
        }
        
        if (isset($memo[$n])) {
            return $memo[$n];
        }
        
        $memo[$n] = self::fibonacciMemo($n - 1, $memo) + self::fibonacciMemo($n - 2, $memo);
        return $memo[$n];
    }
    
    /**
     * Greatest Common Divisor using Euclidean algorithm.
     * Time complexity: O(log min(a, b))
     * Space complexity: O(1)
     */
    public static function gcd(int $a, int $b): int {
        while ($b != 0) {
            $temp = $b;
            $b = $a % $b;
            $a = $temp;
        }
        return $a;
    }
    
    /**
     * Least Common Multiple.
     */
    public static function lcm(int $a, int $b): int {
        return ($a * $b) / self::gcd($a, $b);
    }
    
    /**
     * Factorial implementation using iteration.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function factorial(int $n): int {
        if ($n < 0) {
            throw new InvalidArgumentException("Factorial is not defined for negative numbers");
        }
        if ($n <= 1) {
            return 1;
        }
        
        $result = 1;
        for ($i = 2; $i <= $n; $i++) {
            $result *= $i;
        }
        return $result;
    }
    
    /**
     * Check if a number is prime.
     * Time complexity: O(sqrt(n))
     * Space complexity: O(1)
     */
    public static function isPrime(int $n): bool {
        if ($n < 2) {
            return false;
        }
        if ($n == 2) {
            return true;
        }
        if ($n % 2 == 0) {
            return false;
        }
        
        for ($i = 3; $i * $i <= $n; $i += 2) {
            if ($n % $i == 0) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Sieve of Eratosthenes to find all primes up to n.
     * Time complexity: O(n log log n)
     * Space complexity: O(n)
     */
    public static function sieveOfEratosthenes(int $n): array {
        if ($n < 2) {
            return [];
        }
        
        $isPrime = array_fill(0, $n + 1, true);
        $isPrime[0] = $isPrime[1] = false;
        
        for ($i = 2; $i * $i <= $n; $i++) {
            if ($isPrime[$i]) {
                for ($j = $i * $i; $j <= $n; $j += $i) {
                    $isPrime[$j] = false;
                }
            }
        }
        
        $primes = [];
        for ($i = 2; $i <= $n; $i++) {
            if ($isPrime[$i]) {
                $primes[] = $i;
            }
        }
        
        return $primes;
    }
    
    /**
     * Two Sum problem - find two numbers that add up to target.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    public static function twoSum(array $nums, int $target): ?array {
        $numMap = [];
        
        for ($i = 0; $i < count($nums); $i++) {
            $complement = $target - $nums[$i];
            if (array_key_exists($complement, $numMap)) {
                return [$numMap[$complement], $i];
            }
            $numMap[$nums[$i]] = $i;
        }
        
        return null;
    }
    
    /**
     * Maximum subarray sum (Kadane's algorithm).
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function maxSubarraySum(array $nums): int {
        if (empty($nums)) {
            return 0;
        }
        
        $maxSum = $nums[0];
        $currentSum = $nums[0];
        
        for ($i = 1; $i < count($nums); $i++) {
            $currentSum = max($nums[$i], $currentSum + $nums[$i]);
            $maxSum = max($maxSum, $currentSum);
        }
        
        return $maxSum;
    }
    
    /**
     * Valid parentheses checker.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    public static function isValidParentheses(string $s): bool {
        $stack = [];
        $mapping = [
            ')' => '(',
            '}' => '{',
            ']' => '['
        ];
        
        for ($i = 0; $i < strlen($s); $i++) {
            $char = $s[$i];
            
            if (in_array($char, ['(', '{', '['])) {
                array_push($stack, $char);
            } elseif (array_key_exists($char, $mapping)) {
                if (empty($stack) || array_pop($stack) !== $mapping[$char]) {
                    return false;
                }
            }
        }
        
        return empty($stack);
    }
    
    /**
     * Reverse a string.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    public static function reverseString(string $s): string {
        return strrev($s);
    }
    
    /**
     * Check if a string is a palindrome.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function isPalindrome(string $s): bool {
        $left = 0;
        $right = strlen($s) - 1;
        
        while ($left < $right) {
            if ($s[$left] !== $s[$right]) {
                return false;
            }
            $left++;
            $right--;
        }
        return true;
    }
    
    /**
     * Linear search implementation.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function linearSearch(array $arr, $target): int {
        for ($i = 0; $i < count($arr); $i++) {
            if ($arr[$i] == $target) {
                return $i;
            }
        }
        return -1;
    }
    
    /**
     * Find the maximum element in an array.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function findMax(array $arr): ?int {
        if (empty($arr)) {
            return null;
        }
        
        $max = $arr[0];
        for ($i = 1; $i < count($arr); $i++) {
            if ($arr[$i] > $max) {
                $max = $arr[$i];
            }
        }
        return $max;
    }
    
    /**
     * Count occurrences of each element in array.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    public static function countOccurrences(array $arr): array {
        $counts = [];
        foreach ($arr as $element) {
            if (array_key_exists($element, $counts)) {
                $counts[$element]++;
            } else {
                $counts[$element] = 1;
            }
        }
        return $counts;
    }
    
    /**
     * Remove duplicates from array.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    public static function removeDuplicates(array $arr): array {
        return array_values(array_unique($arr));
    }
    
    /**
     * Check if array is sorted.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    public static function isSorted(array $arr): bool {
        for ($i = 1; $i < count($arr); $i++) {
            if ($arr[$i - 1] > $arr[$i]) {
                return false;
            }
        }
        return true;
    }
}

/**
 * String utility functions.
 */
class StringAlgorithms {
    
    /**
     * Count character frequency in string.
     * Time complexity: O(n)
     * Space complexity: O(1) for ASCII
     */
    public static function charFrequency(string $s): array {
        $frequency = [];
        for ($i = 0; $i < strlen($s); $i++) {
            $char = $s[$i];
            if (array_key_exists($char, $frequency)) {
                $frequency[$char]++;
            } else {
                $frequency[$char] = 1;
            }
        }
        return $frequency;
    }
    
    /**
     * Check if two strings are anagrams.
     * Time complexity: O(n)
     * Space complexity: O(1) for ASCII
     */
    public static function areAnagrams(string $s1, string $s2): bool {
        if (strlen($s1) !== strlen($s2)) {
            return false;
        }
        
        return self::charFrequency($s1) === self::charFrequency($s2);
    }
    
    /**
     * Find longest common prefix.
     * Time complexity: O(S) where S is sum of all characters
     * Space complexity: O(1)
     */
    public static function longestCommonPrefix(array $strs): string {
        if (empty($strs)) {
            return "";
        }
        
        $prefix = $strs[0];
        for ($i = 1; $i < count($strs); $i++) {
            while (strpos($strs[$i], $prefix) !== 0) {
                $prefix = substr($prefix, 0, -1);
                if (empty($prefix)) {
                    return "";
                }
            }
        }
        
        return $prefix;
    }
    
    /**
     * Implement basic string matching (naive approach).
     * Time complexity: O(n*m)
     * Space complexity: O(1)
     */
    public static function strstr(string $haystack, string $needle): int {
        $haystackLen = strlen($haystack);
        $needleLen = strlen($needle);
        
        if ($needleLen === 0) {
            return 0;
        }
        
        for ($i = 0; $i <= $haystackLen - $needleLen; $i++) {
            $j = 0;
            while ($j < $needleLen && $haystack[$i + $j] === $needle[$j]) {
                $j++;
            }
            if ($j === $needleLen) {
                return $i;
            }
        }
        
        return -1;
    }
}

// Example usage and testing
function runTests(): void {
    echo "=== PHP Algorithm Tests ===\n";
    
    // Test bubble sort
    $arr = [64, 34, 25, 12, 22, 11, 90];
    echo "Original array: " . implode(", ", $arr) . "\n";
    BasicAlgorithms::bubbleSort($arr);
    echo "Bubble sorted: " . implode(", ", $arr) . "\n";
    
    // Test binary search
    $sortedArr = [1, 3, 5, 7, 9, 11, 13];
    $target = 7;
    $index = BasicAlgorithms::binarySearch($sortedArr, $target);
    echo "Binary search for $target: index $index\n";
    
    // Test Fibonacci
    $n = 10;
    echo "Fibonacci($n) = " . BasicAlgorithms::fibonacci($n) . "\n";
    
    $memo = [];
    echo "Fibonacci with memo($n) = " . BasicAlgorithms::fibonacciMemo($n, $memo) . "\n";
    
    // Test prime checking
    $num = 17;
    $isPrime = BasicAlgorithms::isPrime($num) ? "true" : "false";
    echo "Is $num prime? $isPrime\n";
    
    // Test Sieve of Eratosthenes
    $primes = BasicAlgorithms::sieveOfEratosthenes(30);
    echo "Primes up to 30: " . implode(", ", $primes) . "\n";
    
    // Test Two Sum
    $nums = [2, 7, 11, 15];
    $targetSum = 9;
    $result = BasicAlgorithms::twoSum($nums, $targetSum);
    echo "Two sum indices for target $targetSum: " . implode(", ", $result) . "\n";
    
    // Test Maximum Subarray Sum
    $subarrayNums = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
    $maxSum = BasicAlgorithms::maxSubarraySum($subarrayNums);
    echo "Maximum subarray sum: $maxSum\n";
    
    // Test Valid Parentheses
    $parentheses = "()[]{}, ([)]";
    $valid1 = BasicAlgorithms::isValidParentheses("()[]{}") ? "true" : "false";
    $valid2 = BasicAlgorithms::isValidParentheses("([)]") ? "true" : "false";
    echo "Valid parentheses '()[]{}': $valid1\n";
    echo "Valid parentheses '([)]': $valid2\n";
    
    // Test palindrome
    $testStr = "racecar";
    $isPalindrome = BasicAlgorithms::isPalindrome($testStr) ? "true" : "false";
    echo "Is '$testStr' a palindrome? $isPalindrome\n";
    
    // Test string algorithms
    echo "\n=== String Algorithm Tests ===\n";
    
    $s1 = "listen";
    $s2 = "silent";
    $areAnagrams = StringAlgorithms::areAnagrams($s1, $s2) ? "true" : "false";
    echo "Are '$s1' and '$s2' anagrams? $areAnagrams\n";
    
    $strs = ["flower", "flow", "flight"];
    $commonPrefix = StringAlgorithms::longestCommonPrefix($strs);
    echo "Longest common prefix of [" . implode(", ", $strs) . "]: '$commonPrefix'\n";
    
    $haystack = "hello world";
    $needle = "world";
    $strPos = StringAlgorithms::strstr($haystack, $needle);
    echo "Position of '$needle' in '$haystack': $strPos\n";
}

// Run tests if this file is executed directly
if (basename(__FILE__) === basename($_SERVER['SCRIPT_NAME'])) {
    runTests();
}

?>