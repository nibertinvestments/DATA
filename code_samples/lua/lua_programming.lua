-- Lua programming examples for ML/AI training

-- Basic data types and variables
local name = "Lua Programming"
local version = 5.4
local isAwesome = true
local nothing = nil

print("Welcome to " .. name .. " version " .. tostring(version))

-- Tables (Lua's main data structure)
local person = {
    name = "Alice",
    age = 30,
    email = "alice@example.com",
    hobbies = {"reading", "coding", "music"}
}

-- Adding methods to tables
function person:greet()
    return "Hello, I'm " .. self.name .. " and I'm " .. self.age .. " years old."
end

function person:addHobby(hobby)
    table.insert(self.hobbies, hobby)
end

-- Array-like tables
local numbers = {10, 20, 30, 40, 50}
local fruits = {"apple", "banana", "orange", "grape"}

-- Functions
function factorial(n)
    if n <= 1 then
        return 1
    else
        return n * factorial(n - 1)
    end
end

-- Function with multiple return values
function divmod(a, b)
    return math.floor(a / b), a % b
end

-- Higher-order functions
function map(func, array)
    local result = {}
    for i, v in ipairs(array) do
        result[i] = func(v)
    end
    return result
end

function filter(predicate, array)
    local result = {}
    for _, v in ipairs(array) do
        if predicate(v) then
            table.insert(result, v)
        end
    end
    return result
end

function reduce(func, array, initial)
    local result = initial
    for _, v in ipairs(array) do
        result = func(result, v)
    end
    return result
end

-- Closure example
function createCounter(start)
    local count = start or 0
    return function()
        count = count + 1
        return count
    end
end

-- Coroutines
function fibonacci()
    local a, b = 0, 1
    while true do
        coroutine.yield(a)
        a, b = b, a + b
    end
end

-- Object-oriented programming with metatables
local Animal = {}
Animal.__index = Animal

function Animal:new(name, species)
    local obj = {
        name = name,
        species = species
    }
    setmetatable(obj, self)
    return obj
end

function Animal:speak()
    return self.name .. " the " .. self.species .. " makes a sound"
end

function Animal:info()
    return "Name: " .. self.name .. ", Species: " .. self.species
end

-- Inheritance
local Dog = {}
Dog.__index = Dog
setmetatable(Dog, Animal)

function Dog:new(name, breed)
    local obj = Animal.new(self, name, "Dog")
    obj.breed = breed
    return obj
end

function Dog:speak()
    return self.name .. " says Woof!"
end

function Dog:fetch()
    return self.name .. " is fetching the ball!"
end

-- Module pattern
local MathUtils = {}

function MathUtils.isPrime(n)
    if n < 2 then return false end
    if n == 2 then return true end
    if n % 2 == 0 then return false end
    
    for i = 3, math.sqrt(n), 2 do
        if n % i == 0 then return false end
    end
    return true
end

function MathUtils.gcd(a, b)
    while b ~= 0 do
        a, b = b, a % b
    end
    return a
end

function MathUtils.lcm(a, b)
    return (a * b) / MathUtils.gcd(a, b)
end

function MathUtils.isPerfectSquare(n)
    local sqrt_n = math.sqrt(n)
    return sqrt_n == math.floor(sqrt_n)
end

-- String manipulation
local StringUtils = {}

function StringUtils.split(str, delimiter)
    local result = {}
    local pattern = "([^" .. delimiter .. "]+)"
    for word in string.gmatch(str, pattern) do
        table.insert(result, word)
    end
    return result
end

function StringUtils.trim(str)
    return str:match("^%s*(.-)%s*$")
end

function StringUtils.capitalize(str)
    return str:gsub("(%a)([%w_']*)", function(first, rest)
        return first:upper() .. rest:lower()
    end)
end

function StringUtils.reverse(str)
    return str:reverse()
end

function StringUtils.isPalindrome(str)
    local cleaned = str:lower():gsub("%W", "")
    return cleaned == cleaned:reverse()
end

-- Data structures
local Stack = {}
Stack.__index = Stack

function Stack:new()
    return setmetatable({items = {}}, self)
end

function Stack:push(item)
    table.insert(self.items, item)
end

function Stack:pop()
    return table.remove(self.items)
end

function Stack:peek()
    return self.items[#self.items]
end

function Stack:isEmpty()
    return #self.items == 0
end

function Stack:size()
    return #self.items
end

-- Queue implementation
local Queue = {}
Queue.__index = Queue

function Queue:new()
    return setmetatable({items = {}, head = 1, tail = 0}, self)
end

function Queue:enqueue(item)
    self.tail = self.tail + 1
    self.items[self.tail] = item
end

function Queue:dequeue()
    if self.head > self.tail then
        return nil
    end
    
    local item = self.items[self.head]
    self.items[self.head] = nil
    self.head = self.head + 1
    return item
end

function Queue:isEmpty()
    return self.head > self.tail
end

function Queue:size()
    return self.tail - self.head + 1
end

-- Sorting algorithms
local function bubbleSort(arr)
    local n = #arr
    for i = 1, n do
        for j = 1, n - i do
            if arr[j] > arr[j + 1] then
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            end
        end
    end
    return arr
end

local function quickSort(arr, low, high)
    if low < high then
        local pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)
    end
end

function partition(arr, low, high)
    local pivot = arr[high]
    local i = low - 1
    
    for j = low, high - 1 do
        if arr[j] <= pivot then
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
        end
    end
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
end

-- Error handling
function safeCall(func, ...)
    local success, result = pcall(func, ...)
    if success then
        return result
    else
        print("Error: " .. tostring(result))
        return nil
    end
end

-- File operations example
function writeToFile(filename, content)
    local file = io.open(filename, "w")
    if file then
        file:write(content)
        file:close()
        return true
    else
        return false
    end
end

function readFromFile(filename)
    local file = io.open(filename, "r")
    if file then
        local content = file:read("*all")
        file:close()
        return content
    else
        return nil
    end
end

-- Main execution
print("=== Basic Data Types ===")
print("Name: " .. name)
print("Version: " .. version)
print("Is awesome: " .. tostring(isAwesome))

print("\n=== Tables and Objects ===")
print(person:greet())
person:addHobby("gaming")
print("Hobbies: " .. table.concat(person.hobbies, ", "))

print("\n=== Functions ===")
print("Factorial of 5: " .. factorial(5))
local quotient, remainder = divmod(17, 5)
print("17 divided by 5: quotient = " .. quotient .. ", remainder = " .. remainder)

print("\n=== Higher-Order Functions ===")
local doubled = map(function(x) return x * 2 end, numbers)
print("Doubled numbers: " .. table.concat(doubled, ", "))

local evenNumbers = filter(function(x) return x % 2 == 0 end, numbers)
print("Even numbers: " .. table.concat(evenNumbers, ", "))

local sum = reduce(function(acc, x) return acc + x end, numbers, 0)
print("Sum: " .. sum)

print("\n=== Closures ===")
local counter1 = createCounter(0)
local counter2 = createCounter(10)
print("Counter1: " .. counter1() .. ", " .. counter1() .. ", " .. counter1())
print("Counter2: " .. counter2() .. ", " .. counter2())

print("\n=== Coroutines ===")
local fib = coroutine.create(fibonacci)
print("First 10 Fibonacci numbers:")
for i = 1, 10 do
    local success, value = coroutine.resume(fib)
    io.write(value .. " ")
end
print()

print("\n=== Object-Oriented Programming ===")
local animal = Animal:new("Generic", "Unknown")
print(animal:info())
print(animal:speak())

local dog = Dog:new("Buddy", "Golden Retriever")
print(dog:info())
print(dog:speak())
print(dog:fetch())

print("\n=== Math Utils ===")
print("Is 17 prime? " .. tostring(MathUtils.isPrime(17)))
print("Is 15 prime? " .. tostring(MathUtils.isPrime(15)))
print("GCD of 48 and 18: " .. MathUtils.gcd(48, 18))
print("LCM of 12 and 8: " .. MathUtils.lcm(12, 8))
print("Is 16 a perfect square? " .. tostring(MathUtils.isPerfectSquare(16)))

print("\n=== String Utils ===")
local testString = "Hello, World! How are you?"
local words = StringUtils.split(testString, " ")
print("Split string: " .. table.concat(words, " | "))
print("Capitalized: " .. StringUtils.capitalize("hello world"))
print("Reversed: " .. StringUtils.reverse("Lua"))
print("Is 'racecar' a palindrome? " .. tostring(StringUtils.isPalindrome("racecar")))

print("\n=== Data Structures ===")
local stack = Stack:new()
stack:push(1)
stack:push(2)
stack:push(3)
print("Stack size: " .. stack:size())
print("Popped: " .. stack:pop())
print("Peek: " .. stack:peek())

local queue = Queue:new()
queue:enqueue("first")
queue:enqueue("second")
queue:enqueue("third")
print("Queue size: " .. queue:size())
print("Dequeued: " .. queue:dequeue())

print("\n=== Sorting ===")
local testArray = {64, 34, 25, 12, 22, 11, 90}
print("Original array: " .. table.concat(testArray, ", "))

local sortedArray = {64, 34, 25, 12, 22, 11, 90}
bubbleSort(sortedArray)
print("Bubble sorted: " .. table.concat(sortedArray, ", "))

local quickSortArray = {64, 34, 25, 12, 22, 11, 90}
quickSort(quickSortArray, 1, #quickSortArray)
print("Quick sorted: " .. table.concat(quickSortArray, ", "))

print("\n=== Error Handling ===")
local result = safeCall(function() return 10 / 2 end)
print("Safe division result: " .. tostring(result))

safeCall(function() error("This is a test error") end)

print("\nAll Lua examples completed!")