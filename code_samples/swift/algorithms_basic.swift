// Sample Swift code for AI training dataset.
// Demonstrates basic algorithms and patterns with iOS/macOS features.

import Foundation

/// Collection of basic algorithms for AI training.
struct BasicAlgorithms {
    
    /// Implementation of bubble sort algorithm.
    /// Time complexity: O(n^2)
    /// Space complexity: O(1)
    static func bubbleSort<T: Comparable>(_ array: inout [T]) {
        let n = array.count
        for i in 0..<n {
            for j in 0..<(n - i - 1) {
                if array[j] > array[j + 1] {
                    array.swapAt(j, j + 1)
                }
            }
        }
    }
    
    /// Generic bubble sort that returns a new sorted array.
    static func bubbleSorted<T: Comparable>(_ array: [T]) -> [T] {
        var result = array
        bubbleSort(&result)
        return result
    }
    
    /// Binary search implementation for sorted arrays.
    /// Time complexity: O(log n)
    /// Space complexity: O(1)
    static func binarySearch<T: Comparable>(_ array: [T], target: T) -> Int? {
        var left = 0
        var right = array.count - 1
        
        while left <= right {
            let mid = left + (right - left) / 2
            if array[mid] == target {
                return mid
            } else if array[mid] < target {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return nil
    }
    
    /// Quick sort implementation using recursion.
    /// Time complexity: O(n log n) average, O(n^2) worst case
    /// Space complexity: O(log n)
    static func quickSort<T: Comparable>(_ array: [T]) -> [T] {
        guard array.count > 1 else { return array }
        
        let pivot = array[array.count / 2]
        let left = array.filter { $0 < pivot }
        let middle = array.filter { $0 == pivot }
        let right = array.filter { $0 > pivot }
        
        return quickSort(left) + middle + quickSort(right)
    }
    
    /// Merge sort implementation.
    /// Time complexity: O(n log n)
    /// Space complexity: O(n)
    static func mergeSort<T: Comparable>(_ array: [T]) -> [T] {
        guard array.count > 1 else { return array }
        
        let mid = array.count / 2
        let left = mergeSort(Array(array[0..<mid]))
        let right = mergeSort(Array(array[mid..<array.count]))
        
        return merge(left, right)
    }
    
    private static func merge<T: Comparable>(_ left: [T], _ right: [T]) -> [T] {
        var result: [T] = []
        var i = 0, j = 0
        
        while i < left.count && j < right.count {
            if left[i] <= right[j] {
                result.append(left[i])
                i += 1
            } else {
                result.append(right[j])
                j += 1
            }
        }
        
        result.append(contentsOf: left[i...])
        result.append(contentsOf: right[j...])
        return result
    }
    
    /// Fibonacci sequence implementation using iteration.
    /// Time complexity: O(n)
    /// Space complexity: O(1)
    static func fibonacci(_ n: Int) -> Int {
        guard n > 1 else { return n }
        
        var prev = 0, curr = 1
        for _ in 2...n {
            (prev, curr) = (curr, prev + curr)
        }
        return curr
    }
    
    /// Fibonacci with memoization.
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    static func fibonacciMemo(_ n: Int, memo: inout [Int: Int]) -> Int {
        guard n > 1 else { return n }
        if let cached = memo[n] { return cached }
        
        memo[n] = fibonacciMemo(n - 1, memo: &memo) + fibonacciMemo(n - 2, memo: &memo)
        return memo[n]!
    }
    
    /// Greatest Common Divisor using Euclidean algorithm.
    /// Time complexity: O(log min(a, b))
    /// Space complexity: O(1)
    static func gcd(_ a: Int, _ b: Int) -> Int {
        var a = a, b = b
        while b != 0 {
            (a, b) = (b, a % b)
        }
        return a
    }
    
    /// Least Common Multiple.
    static func lcm(_ a: Int, _ b: Int) -> Int {
        return (a * b) / gcd(a, b)
    }
    
    /// Factorial implementation using iteration.
    /// Time complexity: O(n)
    /// Space complexity: O(1)
    static func factorial(_ n: Int) -> Int {
        precondition(n >= 0, "Factorial is not defined for negative numbers")
        return n <= 1 ? 1 : (2...n).reduce(1, *)
    }
    
    /// Check if a number is prime.
    /// Time complexity: O(sqrt(n))
    /// Space complexity: O(1)
    static func isPrime(_ n: Int) -> Bool {
        guard n >= 2 else { return false }
        guard n != 2 else { return true }
        guard n % 2 != 0 else { return false }
        
        for i in stride(from: 3, through: Int(sqrt(Double(n))), by: 2) {
            if n % i == 0 { return false }
        }
        return true
    }
    
    /// Sieve of Eratosthenes to find all primes up to n.
    /// Time complexity: O(n log log n)
    /// Space complexity: O(n)
    static func sieveOfEratosthenes(_ n: Int) -> [Int] {
        guard n >= 2 else { return [] }
        
        var isPrime = Array(repeating: true, count: n + 1)
        isPrime[0] = false
        isPrime[1] = false
        
        for i in 2...Int(sqrt(Double(n))) {
            if isPrime[i] {
                for j in stride(from: i * i, through: n, by: i) {
                    isPrime[j] = false
                }
            }
        }
        
        return isPrime.enumerated().compactMap { $1 ? $0 : nil }
    }
    
    /// Two Sum problem - find two numbers that add up to target.
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    static func twoSum(_ nums: [Int], _ target: Int) -> [Int]? {
        var numMap: [Int: Int] = [:]
        
        for (i, num) in nums.enumerated() {
            let complement = target - num
            if let j = numMap[complement] {
                return [j, i]
            }
            numMap[num] = i
        }
        return nil
    }
    
    /// Maximum subarray sum (Kadane's algorithm).
    /// Time complexity: O(n)
    /// Space complexity: O(1)
    static func maxSubarraySum(_ nums: [Int]) -> Int {
        guard !nums.isEmpty else { return 0 }
        
        var maxSum = nums[0]
        var currentSum = nums[0]
        
        for i in 1..<nums.count {
            currentSum = max(nums[i], currentSum + nums[i])
            maxSum = max(maxSum, currentSum)
        }
        return maxSum
    }
    
    /// Valid parentheses checker.
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    static func isValidParentheses(_ s: String) -> Bool {
        var stack: [Character] = []
        let mapping: [Character: Character] = [")": "(", "}": "{", "]": "["]
        
        for char in s {
            if "({[".contains(char) {
                stack.append(char)
            } else if let expected = mapping[char] {
                guard !stack.isEmpty, stack.removeLast() == expected else {
                    return false
                }
            }
        }
        return stack.isEmpty
    }
    
    /// Check if a string is a palindrome.
    /// Time complexity: O(n)
    /// Space complexity: O(1)
    static func isPalindrome(_ s: String) -> Bool {
        let chars = Array(s)
        var left = 0, right = chars.count - 1
        
        while left < right {
            if chars[left] != chars[right] {
                return false
            }
            left += 1
            right -= 1
        }
        return true
    }
    
    /// Linear search implementation.
    /// Time complexity: O(n)
    /// Space complexity: O(1)
    static func linearSearch<T: Equatable>(_ array: [T], target: T) -> Int? {
        for (index, element) in array.enumerated() {
            if element == target {
                return index
            }
        }
        return nil
    }
}

// MARK: - Data Structures

/// Generic Stack implementation.
struct Stack<T> {
    private var items: [T] = []
    
    mutating func push(_ item: T) {
        items.append(item)
    }
    
    @discardableResult
    mutating func pop() -> T? {
        return items.popLast()
    }
    
    func peek() -> T? {
        return items.last
    }
    
    var isEmpty: Bool {
        return items.isEmpty
    }
    
    var count: Int {
        return items.count
    }
}

/// Generic Queue implementation.
struct Queue<T> {
    private var items: [T] = []
    
    mutating func enqueue(_ item: T) {
        items.append(item)
    }
    
    @discardableResult
    mutating func dequeue() -> T? {
        return items.isEmpty ? nil : items.removeFirst()
    }
    
    func front() -> T? {
        return items.first
    }
    
    var isEmpty: Bool {
        return items.isEmpty
    }
    
    var count: Int {
        return items.count
    }
}

/// Binary Tree Node class.
class TreeNode<T> {
    var value: T
    var left: TreeNode<T>?
    var right: TreeNode<T>?
    
    init(_ value: T) {
        self.value = value
    }
}

/// Binary Search Tree implementation.
class BinarySearchTree<T: Comparable> {
    private var root: TreeNode<T>?
    
    func insert(_ value: T) {
        root = insertRecursive(root, value)
    }
    
    private func insertRecursive(_ node: TreeNode<T>?, _ value: T) -> TreeNode<T> {
        guard let node = node else {
            return TreeNode(value)
        }
        
        if value < node.value {
            node.left = insertRecursive(node.left, value)
        } else if value > node.value {
            node.right = insertRecursive(node.right, value)
        }
        
        return node
    }
    
    func search(_ value: T) -> Bool {
        return searchRecursive(root, value)
    }
    
    private func searchRecursive(_ node: TreeNode<T>?, _ value: T) -> Bool {
        guard let node = node else { return false }
        
        if value == node.value {
            return true
        } else if value < node.value {
            return searchRecursive(node.left, value)
        } else {
            return searchRecursive(node.right, value)
        }
    }
    
    func inorderTraversal() -> [T] {
        var result: [T] = []
        inorderRecursive(root, &result)
        return result
    }
    
    private func inorderRecursive(_ node: TreeNode<T>?, _ result: inout [T]) {
        guard let node = node else { return }
        
        inorderRecursive(node.left, &result)
        result.append(node.value)
        inorderRecursive(node.right, &result)
    }
}

/// Linked List implementation.
class LinkedList<T> {
    private class ListNode {
        var value: T
        var next: ListNode?
        
        init(_ value: T) {
            self.value = value
        }
    }
    
    private var head: ListNode?
    private var count = 0
    
    func append(_ value: T) {
        let newNode = ListNode(value)
        
        guard let head = head else {
            self.head = newNode
            count += 1
            return
        }
        
        var current = head
        while current.next != nil {
            current = current.next!
        }
        current.next = newNode
        count += 1
    }
    
    func prepend(_ value: T) {
        let newNode = ListNode(value)
        newNode.next = head
        head = newNode
        count += 1
    }
    
    @discardableResult
    func remove(at index: Int) -> T? {
        guard index >= 0, index < count else { return nil }
        
        if index == 0 {
            let value = head?.value
            head = head?.next
            count -= 1
            return value
        }
        
        var current = head
        for _ in 0..<(index - 1) {
            current = current?.next
        }
        
        let value = current?.next?.value
        current?.next = current?.next?.next
        count -= 1
        return value
    }
    
    func get(at index: Int) -> T? {
        guard index >= 0, index < count else { return nil }
        
        var current = head
        for _ in 0..<index {
            current = current?.next
        }
        return current?.value
    }
    
    var size: Int {
        return count
    }
    
    var isEmpty: Bool {
        return count == 0
    }
    
    func toArray() -> [T] {
        var result: [T] = []
        var current = head
        
        while current != nil {
            result.append(current!.value)
            current = current?.next
        }
        return result
    }
}

// MARK: - Design Patterns

/// Singleton pattern.
class DatabaseConnection {
    static let shared = DatabaseConnection()
    
    private let connectionString: String
    
    private init() {
        connectionString = "database://localhost:5432"
    }
    
    func executeQuery(_ query: String) -> String {
        return "Executing '\(query)' on \(connectionString)"
    }
}

/// Builder pattern.
struct Computer {
    let cpu: String
    let ram: Int
    let storage: Int
    let gpu: String?
    let bluetooth: Bool
    let wifi: Bool
    
    class Builder {
        private var cpu: String?
        private var ram: Int?
        private var storage: Int?
        private var gpu: String?
        private var bluetooth = false
        private var wifi = false
        
        func cpu(_ cpu: String) -> Builder {
            self.cpu = cpu
            return self
        }
        
        func ram(_ ram: Int) -> Builder {
            self.ram = ram
            return self
        }
        
        func storage(_ storage: Int) -> Builder {
            self.storage = storage
            return self
        }
        
        func gpu(_ gpu: String) -> Builder {
            self.gpu = gpu
            return self
        }
        
        func bluetooth(_ enabled: Bool = true) -> Builder {
            self.bluetooth = enabled
            return self
        }
        
        func wifi(_ enabled: Bool = true) -> Builder {
            self.wifi = enabled
            return self
        }
        
        func build() throws -> Computer {
            guard let cpu = cpu else {
                throw BuilderError.missingRequiredField("CPU")
            }
            guard let ram = ram else {
                throw BuilderError.missingRequiredField("RAM")
            }
            guard let storage = storage else {
                throw BuilderError.missingRequiredField("Storage")
            }
            
            return Computer(
                cpu: cpu,
                ram: ram,
                storage: storage,
                gpu: gpu,
                bluetooth: bluetooth,
                wifi: wifi
            )
        }
    }
    
    enum BuilderError: Error, LocalizedError {
        case missingRequiredField(String)
        
        var errorDescription: String? {
            switch self {
            case .missingRequiredField(let field):
                return "\(field) is required"
            }
        }
    }
}

/// Observer pattern.
protocol Observer {
    func update(with message: String)
}

class EmailNotifier: Observer {
    private let email: String
    
    init(email: String) {
        self.email = email
    }
    
    func update(with message: String) {
        print("Email notification to \(email): \(message)")
    }
}

class SMSNotifier: Observer {
    private let phone: String
    
    init(phone: String) {
        self.phone = phone
    }
    
    func update(with message: String) {
        print("SMS notification to \(phone): \(message)")
    }
}

class Subject {
    private var observers: [Observer] = []
    private var state = ""
    
    func attach(_ observer: Observer) {
        observers.append(observer)
    }
    
    func notify() {
        observers.forEach { $0.update(with: state) }
    }
    
    func setState(_ state: String) {
        self.state = state
        notify()
    }
    
    func getState() -> String {
        return state
    }
}

// MARK: - String Extensions

extension String {
    /// Check if string is palindrome (extension example).
    var isPalindrome: Bool {
        return BasicAlgorithms.isPalindrome(self)
    }
    
    /// Convert to camelCase.
    var camelCased: String {
        let components = self.components(separatedBy: CharacterSet.alphanumerics.inverted)
        return components.enumerated().map { index, component in
            index == 0 ? component.lowercased() : component.capitalized
        }.joined()
    }
    
    /// Convert to snake_case.
    var snakeCased: String {
        return self.replacingOccurrences(of: "([A-Z])", with: "_$1", options: .regularExpression)
            .lowercased()
            .replacingOccurrences(of: "^_", with: "", options: .regularExpression)
    }
}

// MARK: - Array Extensions

extension Array where Element: Comparable {
    /// Check if array is sorted (extension example).
    var isSorted: Bool {
        for i in 1..<count {
            if self[i-1] > self[i] {
                return false
            }
        }
        return true
    }
    
    /// Binary search extension method.
    func binarySearchIndex(of target: Element) -> Int? {
        return BasicAlgorithms.binarySearch(self, target: target)
    }
}

// MARK: - Example Usage

/// Example function demonstrating the algorithms and data structures.
func runTests() {
    print("=== Swift Algorithm Tests ===")
    
    // Test bubble sort
    let arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array: \(arr)")
    let sorted = BasicAlgorithms.bubbleSorted(arr)
    print("Bubble sorted: \(sorted)")
    
    // Test binary search
    let sortedArr = [1, 3, 5, 7, 9, 11, 13]
    let target = 7
    if let index = BasicAlgorithms.binarySearch(sortedArr, target: target) {
        print("Binary search for \(target): index \(index)")
    }
    
    // Test Fibonacci
    let n = 10
    print("Fibonacci(\(n)) = \(BasicAlgorithms.fibonacci(n))")
    
    var memo: [Int: Int] = [:]
    print("Fibonacci with memo(\(n)) = \(BasicAlgorithms.fibonacciMemo(n, memo: &memo))")
    
    // Test prime checking
    let num = 17
    print("Is \(num) prime? \(BasicAlgorithms.isPrime(num))")
    
    // Test Sieve of Eratosthenes
    let primes = BasicAlgorithms.sieveOfEratosthenes(30)
    print("Primes up to 30: \(primes)")
    
    // Test Two Sum
    let nums = [2, 7, 11, 15]
    let targetSum = 9
    if let result = BasicAlgorithms.twoSum(nums, targetSum) {
        print("Two sum indices for target \(targetSum): \(result)")
    }
    
    // Test Maximum Subarray Sum
    let subarrayNums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    let maxSum = BasicAlgorithms.maxSubarraySum(subarrayNums)
    print("Maximum subarray sum: \(maxSum)")
    
    // Test Valid Parentheses
    print("Valid parentheses '()[]{}': \(BasicAlgorithms.isValidParentheses("()[]{}"))")
    print("Valid parentheses '([)]': \(BasicAlgorithms.isValidParentheses("([)]"))")
    
    // Test palindrome
    let testStr = "racecar"
    print("Is '\(testStr)' a palindrome? \(testStr.isPalindrome)")
    
    // Test data structures
    print("\n=== Data Structure Tests ===")
    
    // Test Stack
    var stack = Stack<Int>()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print("Stack peek: \(stack.peek() ?? 0)")
    print("Stack pop: \(stack.pop() ?? 0)")
    
    // Test BST
    let bst = BinarySearchTree<Int>()
    [5, 3, 7, 2, 4, 6, 8].forEach { bst.insert($0) }
    print("BST inorder traversal: \(bst.inorderTraversal())")
    print("BST search 4: \(bst.search(4))")
    
    // Test design patterns
    print("\n=== Design Pattern Tests ===")
    
    // Test Singleton
    let db = DatabaseConnection.shared
    print(db.executeQuery("SELECT * FROM users"))
    
    // Test Builder
    do {
        let computer = try Computer.Builder()
            .cpu("Intel i7")
            .ram(16)
            .storage(512)
            .gpu("NVIDIA RTX 4080")
            .wifi()
            .bluetooth()
            .build()
        
        print("Built computer: CPU=\(computer.cpu), RAM=\(computer.ram)GB, WiFi=\(computer.wifi)")
    } catch {
        print("Builder error: \(error.localizedDescription)")
    }
    
    // Test Observer
    let subject = Subject()
    let emailObserver = EmailNotifier(email: "user@example.com")
    let smsObserver = SMSNotifier(phone: "+1234567890")
    
    subject.attach(emailObserver)
    subject.attach(smsObserver)
    subject.setState("System update available")
    
    // Test extensions
    print("\n=== Extension Tests ===")
    
    let testArray = [1, 2, 3, 4, 5]
    print("Is array \(testArray) sorted? \(testArray.isSorted)")
    
    if let index = testArray.binarySearchIndex(of: 3) {
        print("Binary search for 3 in array: index \(index)")
    }
    
    let camelTest = "hello_world_example"
    print("'\(camelTest)' in camelCase: '\(camelTest.camelCased)'")
    
    let snakeTest = "HelloWorldExample"
    print("'\(snakeTest)' in snake_case: '\(snakeTest.snakeCased)'")
}

// Uncomment to run tests
// runTests()