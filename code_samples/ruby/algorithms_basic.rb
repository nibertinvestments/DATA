# Sample Ruby code for AI training dataset.
# Demonstrates basic algorithms and patterns.

class BasicAlgorithms
  
  # Implementation of bubble sort algorithm.
  # Time complexity: O(n^2)
  # Space complexity: O(1)
  def self.bubble_sort(arr)
    n = arr.length
    n.times do |i|
      (n - i - 1).times do |j|
        if arr[j] > arr[j + 1]
          arr[j], arr[j + 1] = arr[j + 1], arr[j]
        end
      end
    end
    arr
  end
  
  # Binary search implementation for sorted arrays.
  # Time complexity: O(log n)
  # Space complexity: O(1)
  def self.binary_search(arr, target)
    left = 0
    right = arr.length - 1
    
    while left <= right
      mid = left + (right - left) / 2
      return mid if arr[mid] == target
      
      if arr[mid] < target
        left = mid + 1
      else
        right = mid - 1
      end
    end
    -1
  end
  
  # Quick sort implementation using recursion.
  # Time complexity: O(n log n) average, O(n^2) worst case
  # Space complexity: O(log n)
  def self.quick_sort(arr)
    return arr if arr.length <= 1
    
    pivot = arr[arr.length / 2]
    left = arr.select { |x| x < pivot }
    middle = arr.select { |x| x == pivot }
    right = arr.select { |x| x > pivot }
    
    quick_sort(left) + middle + quick_sort(right)
  end
  
  # Merge sort implementation.
  # Time complexity: O(n log n)
  # Space complexity: O(n)
  def self.merge_sort(arr)
    return arr if arr.length <= 1
    
    mid = arr.length / 2
    left = merge_sort(arr[0...mid])
    right = merge_sort(arr[mid..-1])
    
    merge(left, right)
  end
  
  def self.merge(left, right)
    result = []
    i = j = 0
    
    while i < left.length && j < right.length
      if left[i] <= right[j]
        result << left[i]
        i += 1
      else
        result << right[j]
        j += 1
      end
    end
    
    result + left[i..-1] + right[j..-1]
  end
  
  # Fibonacci sequence implementation using iteration.
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.fibonacci(n)
    return n if n <= 1
    
    prev, curr = 0, 1
    (2..n).each do
      prev, curr = curr, prev + curr
    end
    curr
  end
  
  # Fibonacci with memoization.
  # Time complexity: O(n)
  # Space complexity: O(n)
  def self.fibonacci_memo(n, memo = {})
    return n if n <= 1
    return memo[n] if memo.key?(n)
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
  end
  
  # Greatest Common Divisor using Euclidean algorithm.
  # Time complexity: O(log min(a, b))
  # Space complexity: O(1)
  def self.gcd(a, b)
    while b != 0
      a, b = b, a % b
    end
    a
  end
  
  # Least Common Multiple.
  def self.lcm(a, b)
    (a * b) / gcd(a, b)
  end
  
  # Factorial implementation using iteration.
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.factorial(n)
    raise ArgumentError, "Factorial is not defined for negative numbers" if n < 0
    return 1 if n <= 1
    
    (2..n).inject(1, :*)
  end
  
  # Check if a number is prime.
  # Time complexity: O(sqrt(n))
  # Space complexity: O(1)
  def self.prime?(n)
    return false if n < 2
    return true if n == 2
    return false if n.even?
    
    3.step(Math.sqrt(n).to_i, 2) do |i|
      return false if n % i == 0
    end
    true
  end
  
  # Sieve of Eratosthenes to find all primes up to n.
  # Time complexity: O(n log log n)
  # Space complexity: O(n)
  def self.sieve_of_eratosthenes(n)
    return [] if n < 2
    
    is_prime = Array.new(n + 1, true)
    is_prime[0] = is_prime[1] = false
    
    2.upto(Math.sqrt(n).to_i) do |i|
      if is_prime[i]
        (i * i).step(n, i) do |j|
          is_prime[j] = false
        end
      end
    end
    
    (2..n).select { |i| is_prime[i] }
  end
  
  # Two Sum problem - find two numbers that add up to target.
  # Time complexity: O(n)
  # Space complexity: O(n)
  def self.two_sum(nums, target)
    num_map = {}
    
    nums.each_with_index do |num, i|
      complement = target - num
      return [num_map[complement], i] if num_map.key?(complement)
      
      num_map[num] = i
    end
    
    nil
  end
  
  # Maximum subarray sum (Kadane's algorithm).
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.max_subarray_sum(nums)
    return 0 if nums.empty?
    
    max_sum = current_sum = nums[0]
    
    nums[1..-1].each do |num|
      current_sum = [num, current_sum + num].max
      max_sum = [max_sum, current_sum].max
    end
    
    max_sum
  end
  
  # Valid parentheses checker.
  # Time complexity: O(n)
  # Space complexity: O(n)
  def self.valid_parentheses?(s)
    stack = []
    mapping = { ')' => '(', '}' => '{', ']' => '[' }
    
    s.each_char do |char|
      if %w[( { [].include?(char)
        stack.push(char)
      elsif mapping.key?(char)
        return false if stack.empty? || stack.pop != mapping[char]
      end
    end
    
    stack.empty?
  end
  
  # Check if a string is a palindrome.
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.palindrome?(s)
    left = 0
    right = s.length - 1
    
    while left < right
      return false if s[left] != s[right]
      left += 1
      right -= 1
    end
    true
  end
  
  # Linear search implementation.
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.linear_search(arr, target)
    arr.each_with_index do |element, index|
      return index if element == target
    end
    -1
  end
  
  # Find the maximum element in an array.
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.find_max(arr)
    return nil if arr.empty?
    arr.max
  end
  
  # Count occurrences of each element in array.
  # Time complexity: O(n)
  # Space complexity: O(n)
  def self.count_occurrences(arr)
    arr.each_with_object(Hash.new(0)) { |element, counts| counts[element] += 1 }
  end
  
  # Remove duplicates from array.
  # Time complexity: O(n)
  # Space complexity: O(n)
  def self.remove_duplicates(arr)
    arr.uniq
  end
  
  # Check if array is sorted.
  # Time complexity: O(n)
  # Space complexity: O(1)
  def self.sorted?(arr)
    arr.each_cons(2).all? { |a, b| a <= b }
  end
end

# String utility functions.
module StringAlgorithms
  
  # Count character frequency in string.
  # Time complexity: O(n)
  # Space complexity: O(1) for ASCII
  def self.char_frequency(s)
    s.each_char.with_object(Hash.new(0)) { |char, freq| freq[char] += 1 }
  end
  
  # Check if two strings are anagrams.
  # Time complexity: O(n)
  # Space complexity: O(1) for ASCII
  def self.anagrams?(s1, s2)
    return false if s1.length != s2.length
    char_frequency(s1) == char_frequency(s2)
  end
  
  # Find longest common prefix.
  # Time complexity: O(S) where S is sum of all characters
  # Space complexity: O(1)
  def self.longest_common_prefix(strs)
    return "" if strs.empty?
    
    prefix = strs.first
    strs[1..-1].each do |str|
      while !str.start_with?(prefix)
        prefix = prefix[0...-1]
        return "" if prefix.empty?
      end
    end
    
    prefix
  end
  
  # Implement basic string matching (naive approach).
  # Time complexity: O(n*m)
  # Space complexity: O(1)
  def self.string_match(haystack, needle)
    return 0 if needle.empty?
    
    (0..haystack.length - needle.length).each do |i|
      return i if haystack[i, needle.length] == needle
    end
    
    -1
  end
  
  # Reverse words in a string.
  # Time complexity: O(n)
  # Space complexity: O(n)
  def self.reverse_words(s)
    s.split.reverse.join(' ')
  end
  
  # Convert string to camelCase.
  def self.to_camel_case(s)
    s.split(/[^a-zA-Z0-9]/).map.with_index do |word, index|
      index == 0 ? word.downcase : word.capitalize
    end.join
  end
  
  # Convert string to snake_case.
  def self.to_snake_case(s)
    s.gsub(/([A-Z])/, '_\1').downcase.gsub(/^_/, '')
  end
end

# Ruby-specific design patterns and idioms.
module RubyPatterns
  
  # Singleton pattern using module.
  module DatabaseConnection
    extend self
    
    attr_accessor :connection_string
    
    def initialize
      @connection_string = "database://localhost:5432"
    end
    
    def execute_query(query)
      "Executing '#{query}' on #{@connection_string}"
    end
  end
  
  # Builder pattern using method chaining.
  class Computer
    attr_reader :cpu, :ram, :storage, :gpu, :bluetooth, :wifi
    
    def initialize
      @bluetooth = false
      @wifi = false
    end
    
    def cpu(cpu)
      @cpu = cpu
      self
    end
    
    def ram(ram)
      @ram = ram
      self
    end
    
    def storage(storage)
      @storage = storage
      self
    end
    
    def gpu(gpu)
      @gpu = gpu
      self
    end
    
    def bluetooth(enabled = true)
      @bluetooth = enabled
      self
    end
    
    def wifi(enabled = true)
      @wifi = enabled
      self
    end
    
    def build
      raise "CPU is required" unless @cpu
      raise "RAM is required" unless @ram
      raise "Storage is required" unless @storage
      
      self.dup.freeze
    end
  end
  
  # Observer pattern using Ruby's observer library.
  require 'observer'
  
  class Subject
    include Observable
    
    attr_reader :state
    
    def state=(new_state)
      @state = new_state
      changed
      notify_observers(@state)
    end
  end
  
  class EmailObserver
    def initialize(email)
      @email = email
    end
    
    def update(state)
      puts "Email notification to #{@email}: #{state}"
    end
  end
  
  class SMSObserver
    def initialize(phone)
      @phone = phone
    end
    
    def update(state)
      puts "SMS notification to #{@phone}: #{state}"
    end
  end
  
  # Strategy pattern using blocks/procs.
  class SortContext
    def initialize(&strategy)
      @strategy = strategy
    end
    
    def strategy=(&strategy)
      @strategy = strategy
    end
    
    def execute_sort(data)
      @strategy.call(data)
    end
  end
  
  # Factory pattern using case statements.
  class VehicleFactory
    def self.create_vehicle(type, brand)
      case type.downcase
      when 'car'
        Car.new(brand)
      when 'motorcycle'
        Motorcycle.new(brand)
      else
        raise ArgumentError, "Unknown vehicle type: #{type}"
      end
    end
  end
  
  class Car
    def initialize(brand)
      @brand = brand
    end
    
    def start
      "Starting #{@brand} car engine"
    end
    
    def stop
      "Stopping #{@brand} car engine"
    end
    
    def type
      "Car"
    end
  end
  
  class Motorcycle
    def initialize(brand)
      @brand = brand
    end
    
    def start
      "Starting #{@brand} motorcycle engine"
    end
    
    def stop
      "Stopping #{@brand} motorcycle engine"
    end
    
    def type
      "Motorcycle"
    end
  end
end

# Functional programming patterns in Ruby.
module FunctionalPatterns
  
  # Higher-order functions examples.
  def self.map_with_index(array, &block)
    array.each_with_index.map(&block)
  end
  
  def self.filter_map(array, &block)
    array.filter_map(&block)
  end
  
  def self.reduce_with_memo(array, initial = 0, &block)
    array.reduce(initial, &block)
  end
  
  # Currying examples.
  def self.add(x)
    ->(y) { x + y }
  end
  
  def self.multiply(x)
    ->(y) { x * y }
  end
  
  # Partial application.
  def self.partial(method, *args1)
    ->(*args2) { method.call(*args1, *args2) }
  end
  
  # Function composition.
  def self.compose(f, g)
    ->(x) { f.call(g.call(x)) }
  end
end

# Example usage and testing
def run_tests
  puts "=== Ruby Algorithm Tests ==="
  
  # Test bubble sort
  arr = [64, 34, 25, 12, 22, 11, 90]
  puts "Original array: #{arr}"
  sorted = BasicAlgorithms.bubble_sort(arr.dup)
  puts "Bubble sorted: #{sorted}"
  
  # Test binary search
  sorted_arr = [1, 3, 5, 7, 9, 11, 13]
  target = 7
  index = BasicAlgorithms.binary_search(sorted_arr, target)
  puts "Binary search for #{target}: index #{index}"
  
  # Test Fibonacci
  n = 10
  puts "Fibonacci(#{n}) = #{BasicAlgorithms.fibonacci(n)}"
  puts "Fibonacci with memo(#{n}) = #{BasicAlgorithms.fibonacci_memo(n)}"
  
  # Test prime checking
  num = 17
  puts "Is #{num} prime? #{BasicAlgorithms.prime?(num)}"
  
  # Test Sieve of Eratosthenes
  primes = BasicAlgorithms.sieve_of_eratosthenes(30)
  puts "Primes up to 30: #{primes}"
  
  # Test Two Sum
  nums = [2, 7, 11, 15]
  target_sum = 9
  result = BasicAlgorithms.two_sum(nums, target_sum)
  puts "Two sum indices for target #{target_sum}: #{result}"
  
  # Test Maximum Subarray Sum
  subarray_nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
  max_sum = BasicAlgorithms.max_subarray_sum(subarray_nums)
  puts "Maximum subarray sum: #{max_sum}"
  
  # Test Valid Parentheses
  puts "Valid parentheses '()[]{}': #{BasicAlgorithms.valid_parentheses?('()[]{})}")
  puts "Valid parentheses '([)]': #{BasicAlgorithms.valid_parentheses?('([)]')}"
  
  # Test palindrome
  test_str = "racecar"
  puts "Is '#{test_str}' a palindrome? #{BasicAlgorithms.palindrome?(test_str)}"
  
  # Test string algorithms
  puts "\n=== String Algorithm Tests ==="
  
  s1, s2 = "listen", "silent"
  puts "Are '#{s1}' and '#{s2}' anagrams? #{StringAlgorithms.anagrams?(s1, s2)}"
  
  strs = %w[flower flow flight]
  common_prefix = StringAlgorithms.longest_common_prefix(strs)
  puts "Longest common prefix of #{strs}: '#{common_prefix}'"
  
  # Test Ruby patterns
  puts "\n=== Ruby Pattern Tests ==="
  
  # Test builder pattern
  computer = RubyPatterns::Computer.new
    .cpu("Intel i7")
    .ram(16)
    .storage(512)
    .gpu("NVIDIA RTX 4080")
    .wifi
    .bluetooth
    .build
  
  puts "Built computer: CPU=#{computer.cpu}, RAM=#{computer.ram}GB, WiFi=#{computer.wifi}"
  
  # Test observer pattern
  subject = RubyPatterns::Subject.new
  email_observer = RubyPatterns::EmailObserver.new("user@example.com")
  sms_observer = RubyPatterns::SMSObserver.new("+1234567890")
  
  subject.add_observer(email_observer)
  subject.add_observer(sms_observer)
  subject.state = "System update available"
  
  # Test functional patterns
  puts "\n=== Functional Pattern Tests ==="
  
  add_five = FunctionalPatterns.add(5)
  puts "add_five(10) = #{add_five.call(10)}"
  
  multiply_three = FunctionalPatterns.multiply(3)
  puts "multiply_three(4) = #{multiply_three.call(4)}"
  
  # Function composition
  add_one = ->(x) { x + 1 }
  square = ->(x) { x * x }
  add_then_square = FunctionalPatterns.compose(square, add_one)
  puts "add_then_square(5) = #{add_then_square.call(5)}"
end

# Run tests if this file is executed directly
if __FILE__ == $0
  run_tests
end