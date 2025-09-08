// Sample Kotlin code for AI training dataset.
// Demonstrates basic algorithms and patterns with Kotlin features.

/**
 * Collection of basic algorithms for AI training.
 */
object BasicAlgorithms {
    
    /**
     * Implementation of bubble sort algorithm.
     * Time complexity: O(n^2)
     * Space complexity: O(1)
     */
    fun <T : Comparable<T>> bubbleSort(arr: MutableList<T>) {
        val n = arr.size
        for (i in 0 until n) {
            for (j in 0 until n - i - 1) {
                if (arr[j] > arr[j + 1]) {
                    val temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
                }
            }
        }
    }
    
    /**
     * Binary search implementation for sorted arrays.
     * Time complexity: O(log n)
     * Space complexity: O(1)
     */
    fun <T : Comparable<T>> binarySearch(arr: List<T>, target: T): Int {
        var left = 0
        var right = arr.size - 1
        
        while (left <= right) {
            val mid = left + (right - left) / 2
            when {
                arr[mid] == target -> return mid
                arr[mid] < target -> left = mid + 1
                else -> right = mid - 1
            }
        }
        return -1
    }
    
    /**
     * Quick sort implementation using recursion.
     * Time complexity: O(n log n) average, O(n^2) worst case
     * Space complexity: O(log n)
     */
    fun <T : Comparable<T>> quickSort(arr: List<T>): List<T> {
        if (arr.size <= 1) return arr
        
        val pivot = arr[arr.size / 2]
        val left = arr.filter { it < pivot }
        val middle = arr.filter { it == pivot }
        val right = arr.filter { it > pivot }
        
        return quickSort(left) + middle + quickSort(right)
    }
    
    /**
     * Fibonacci sequence implementation using iteration.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    fun fibonacci(n: Int): Long {
        if (n <= 1) return n.toLong()
        
        var prev = 0L
        var curr = 1L
        
        for (i in 2..n) {
            val temp = curr
            curr = prev + curr
            prev = temp
        }
        return curr
    }
    
    /**
     * Fibonacci with memoization using map.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    fun fibonacciMemo(n: Int, memo: MutableMap<Int, Long> = mutableMapOf()): Long {
        if (n <= 1) return n.toLong()
        if (memo.containsKey(n)) return memo[n]!!
        
        memo[n] = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo)
        return memo[n]!!
    }
    
    /**
     * Greatest Common Divisor using Euclidean algorithm.
     * Time complexity: O(log min(a, b))
     * Space complexity: O(1)
     */
    fun gcd(a: Int, b: Int): Int {
        var x = a
        var y = b
        while (y != 0) {
            val temp = y
            y = x % y
            x = temp
        }
        return x
    }
    
    /**
     * Least Common Multiple.
     */
    fun lcm(a: Int, b: Int): Int = (a * b) / gcd(a, b)
    
    /**
     * Factorial implementation using iteration.
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    fun factorial(n: Int): Long {
        require(n >= 0) { "Factorial is not defined for negative numbers" }
        return (1..n).fold(1L) { acc, i -> acc * i }
    }
    
    /**
     * Check if a number is prime.
     * Time complexity: O(sqrt(n))
     * Space complexity: O(1)
     */
    fun isPrime(n: Int): Boolean {
        if (n < 2) return false
        if (n == 2) return true
        if (n % 2 == 0) return false
        
        for (i in 3..kotlin.math.sqrt(n.toDouble()).toInt() step 2) {
            if (n % i == 0) return false
        }
        return true
    }
    
    /**
     * Two Sum problem - find two numbers that add up to target.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    fun twoSum(nums: IntArray, target: Int): IntArray? {
        val numMap = mutableMapOf<Int, Int>()
        
        for ((i, num) in nums.withIndex()) {
            val complement = target - num
            if (numMap.containsKey(complement)) {
                return intArrayOf(numMap[complement]!!, i)
            }
            numMap[num] = i
        }
        return null
    }
    
    /**
     * Maximum subarray sum (Kadane's algorithm).
     * Time complexity: O(n)
     * Space complexity: O(1)
     */
    fun maxSubarraySum(nums: IntArray): Int {
        if (nums.isEmpty()) return 0
        
        var maxSum = nums[0]
        var currentSum = nums[0]
        
        for (i in 1 until nums.size) {
            currentSum = maxOf(nums[i], currentSum + nums[i])
            maxSum = maxOf(maxSum, currentSum)
        }
        return maxSum
    }
    
    /**
     * Valid parentheses checker.
     * Time complexity: O(n)
     * Space complexity: O(n)
     */
    fun isValidParentheses(s: String): Boolean {
        val stack = mutableListOf<Char>()
        val mapping = mapOf(')' to '(', '}' to '{', ']' to '[')
        
        for (char in s) {
            when (char) {
                '(', '{', '[' -> stack.add(char)
                ')', '}', ']' -> {
                    if (stack.isEmpty() || stack.removeAt(stack.size - 1) != mapping[char]) {
                        return false
                    }
                }
            }
        }
        return stack.isEmpty()
    }
}

// Data Structures

/**
 * Generic Stack implementation.
 */
class Stack<T> {
    private val items = mutableListOf<T>()
    
    fun push(item: T) {
        items.add(item)
    }
    
    fun pop(): T? {
        return if (items.isNotEmpty()) items.removeAt(items.size - 1) else null
    }
    
    fun peek(): T? {
        return items.lastOrNull()
    }
    
    fun isEmpty(): Boolean = items.isEmpty()
    
    fun size(): Int = items.size
}

/**
 * Generic Queue implementation.
 */
class Queue<T> {
    private val items = mutableListOf<T>()
    
    fun enqueue(item: T) {
        items.add(item)
    }
    
    fun dequeue(): T? {
        return if (items.isNotEmpty()) items.removeAt(0) else null
    }
    
    fun front(): T? = items.firstOrNull()
    
    fun isEmpty(): Boolean = items.isEmpty()
    
    fun size(): Int = items.size
}

/**
 * Binary Tree Node class.
 */
data class TreeNode<T>(
    var value: T,
    var left: TreeNode<T>? = null,
    var right: TreeNode<T>? = null
)

/**
 * Binary Search Tree implementation.
 */
class BinarySearchTree<T : Comparable<T>> {
    private var root: TreeNode<T>? = null
    
    fun insert(value: T) {
        root = insertRecursive(root, value)
    }
    
    private fun insertRecursive(node: TreeNode<T>?, value: T): TreeNode<T> {
        if (node == null) {
            return TreeNode(value)
        }
        
        when {
            value < node.value -> node.left = insertRecursive(node.left, value)
            value > node.value -> node.right = insertRecursive(node.right, value)
        }
        
        return node
    }
    
    fun search(value: T): Boolean {
        return searchRecursive(root, value)
    }
    
    private fun searchRecursive(node: TreeNode<T>?, value: T): Boolean {
        if (node == null) return false
        
        return when {
            value == node.value -> true
            value < node.value -> searchRecursive(node.left, value)
            else -> searchRecursive(node.right, value)
        }
    }
    
    fun inorderTraversal(): List<T> {
        val result = mutableListOf<T>()
        inorderRecursive(root, result)
        return result
    }
    
    private fun inorderRecursive(node: TreeNode<T>?, result: MutableList<T>) {
        if (node != null) {
            inorderRecursive(node.left, result)
            result.add(node.value)
            inorderRecursive(node.right, result)
        }
    }
}

// Design Patterns

/**
 * Singleton pattern using object declaration.
 */
object DatabaseConnection {
    private const val connectionString = "database://localhost:5432"
    
    fun executeQuery(query: String): String {
        return "Executing '$query' on $connectionString"
    }
}

/**
 * Builder pattern using Kotlin DSL.
 */
data class Computer(
    val cpu: String,
    val ram: Int,
    val storage: Int,
    val gpu: String? = null,
    val bluetooth: Boolean = false,
    val wifi: Boolean = false
)

class ComputerBuilder {
    private var cpu: String? = null
    private var ram: Int? = null
    private var storage: Int? = null
    private var gpu: String? = null
    private var bluetooth: Boolean = false
    private var wifi: Boolean = false
    
    fun cpu(cpu: String) = apply { this.cpu = cpu }
    fun ram(ram: Int) = apply { this.ram = ram }
    fun storage(storage: Int) = apply { this.storage = storage }
    fun gpu(gpu: String) = apply { this.gpu = gpu }
    fun bluetooth(enabled: Boolean = true) = apply { this.bluetooth = enabled }
    fun wifi(enabled: Boolean = true) = apply { this.wifi = enabled }
    
    fun build(): Computer {
        requireNotNull(cpu) { "CPU is required" }
        requireNotNull(ram) { "RAM is required" }
        requireNotNull(storage) { "Storage is required" }
        
        return Computer(cpu!!, ram!!, storage!!, gpu, bluetooth, wifi)
    }
}

// DSL function for computer builder
fun computer(init: ComputerBuilder.() -> Unit): Computer {
    val builder = ComputerBuilder()
    builder.init()
    return builder.build()
}

/**
 * Observer pattern using delegates.
 */
interface Observer<T> {
    fun update(data: T)
}

class Subject<T> {
    private val observers = mutableListOf<Observer<T>>()
    
    fun attach(observer: Observer<T>) {
        observers.add(observer)
    }
    
    fun detach(observer: Observer<T>) {
        observers.remove(observer)
    }
    
    fun notify(data: T) {
        observers.forEach { it.update(data) }
    }
}

class EmailNotifier(private val email: String) : Observer<String> {
    override fun update(data: String) {
        println("Email notification to $email: $data")
    }
}

// Kotlin Extensions

/**
 * Extension functions for collections.
 */
fun <T : Comparable<T>> List<T>.isSorted(): Boolean {
    return this.zipWithNext().all { (a, b) -> a <= b }
}

fun <T : Comparable<T>> List<T>.binarySearchIndex(target: T): Int {
    return BasicAlgorithms.binarySearch(this, target)
}

/**
 * Extension functions for strings.
 */
fun String.isPalindrome(): Boolean {
    val cleaned = this.toLowerCase().filter { it.isLetterOrDigit() }
    return cleaned == cleaned.reversed()
}

fun String.toCamelCase(): String {
    return this.split("_", "-", " ").mapIndexed { index, word ->
        if (index == 0) word.toLowerCase() else word.capitalize()
    }.joinToString("")
}

// Coroutines example
suspend fun fetchData(id: Int): String {
    kotlinx.coroutines.delay(1000) // Simulate network call
    return "Data for ID: $id"
}

// Higher-order functions and lambdas
fun <T, R> List<T>.mapNotNull(transform: (T) -> R?): List<R> {
    return this.mapNotNull(transform)
}

inline fun <T> List<T>.forEachIndexed(action: (index: Int, T) -> Unit) {
    for ((index, item) in this.withIndex()) {
        action(index, item)
    }
}

// Example usage and testing
fun main() {
    println("=== Kotlin Algorithm Tests ===")
    
    // Test bubble sort
    val arr = mutableListOf(64, 34, 25, 12, 22, 11, 90)
    println("Original array: $arr")
    BasicAlgorithms.bubbleSort(arr)
    println("Bubble sorted: $arr")
    
    // Test binary search
    val sortedArr = listOf(1, 3, 5, 7, 9, 11, 13)
    val target = 7
    val index = BasicAlgorithms.binarySearch(sortedArr, target)
    println("Binary search for $target: index $index")
    
    // Test Fibonacci
    val n = 10
    println("Fibonacci($n) = ${BasicAlgorithms.fibonacci(n)}")
    println("Fibonacci with memo($n) = ${BasicAlgorithms.fibonacciMemo(n)}")
    
    // Test prime checking
    val num = 17
    println("Is $num prime? ${BasicAlgorithms.isPrime(num)}")
    
    // Test Two Sum
    val nums = intArrayOf(2, 7, 11, 15)
    val targetSum = 9
    val result = BasicAlgorithms.twoSum(nums, targetSum)
    println("Two sum indices for target $targetSum: ${result?.contentToString()}")
    
    // Test data structures
    println("\n=== Data Structure Tests ===")
    
    // Test Stack
    val stack = Stack<Int>()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    println("Stack peek: ${stack.peek()}")
    println("Stack pop: ${stack.pop()}")
    
    // Test BST
    val bst = BinarySearchTree<Int>()
    listOf(5, 3, 7, 2, 4, 6, 8).forEach { bst.insert(it) }
    println("BST inorder traversal: ${bst.inorderTraversal()}")
    println("BST search 4: ${bst.search(4)}")
    
    // Test design patterns
    println("\n=== Design Pattern Tests ===")
    
    // Test Singleton
    val db = DatabaseConnection
    println(db.executeQuery("SELECT * FROM users"))
    
    // Test Builder with DSL
    val computerWithDSL = computer {
        cpu("Intel i7")
        ram(16)
        storage(512)
        gpu("NVIDIA RTX 4080")
        wifi()
        bluetooth()
    }
    println("Built computer: $computerWithDSL")
    
    // Test Observer
    val subject = Subject<String>()
    val emailObserver = EmailNotifier("user@example.com")
    subject.attach(emailObserver)
    subject.notify("System update available")
    
    // Test extensions
    println("\n=== Extension Tests ===")
    
    val testList = listOf(1, 2, 3, 4, 5)
    println("Is list $testList sorted? ${testList.isSorted()}")
    
    val searchIndex = testList.binarySearchIndex(3)
    println("Binary search for 3 in list: index $searchIndex")
    
    val palindromeTest = "A man a plan a canal Panama"
    println("Is '$palindromeTest' a palindrome? ${palindromeTest.isPalindrome()}")
    
    val camelTest = "hello_world_example"
    println("'$camelTest' in camelCase: '${camelTest.toCamelCase()}'")
}