--[[
Advanced Lua Programming Examples
==================================

This module demonstrates intermediate to advanced Lua concepts including:
- Advanced table manipulation and metatables
- Coroutines and cooperative multitasking
- Object-oriented programming patterns
- Functional programming techniques
- C API integration patterns
- Game development and scripting patterns
- Performance optimization techniques
]]

-- Advanced Table and Metatable Operations
-- =======================================

-- Class implementation using metatables
local function createClass(className)
    local class = {}
    class.__index = class
    class.__name = className
    
    -- Constructor
    function class:new(...)
        local instance = setmetatable({}, self)
        if instance.init then
            instance:init(...)
        end
        return instance
    end
    
    -- Inheritance support
    function class:extend(subclassName)
        local subclass = createClass(subclassName)
        setmetatable(subclass, {__index = self})
        return subclass
    end
    
    return class
end

-- Vector class example
local Vector = createClass("Vector")

function Vector:init(x, y, z)
    self.x = x or 0
    self.y = y or 0
    self.z = z or 0
end

function Vector:__tostring()
    return string.format("Vector(%.2f, %.2f, %.2f)", self.x, self.y, self.z)
end

function Vector:__add(other)
    return Vector:new(
        self.x + other.x,
        self.y + other.y,
        self.z + other.z
    )
end

function Vector:__mul(scalar)
    if type(scalar) == "number" then
        return Vector:new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )
    end
end

function Vector:magnitude()
    return math.sqrt(self.x^2 + self.y^2 + self.z^2)
end

function Vector:normalize()
    local mag = self:magnitude()
    if mag > 0 then
        return self * (1 / mag)
    end
    return Vector:new(0, 0, 0)
end

function Vector:dot(other)
    return self.x * other.x + self.y * other.y + self.z * other.z
end

function Vector:cross(other)
    return Vector:new(
        self.y * other.z - self.z * other.y,
        self.z * other.x - self.x * other.z,
        self.x * other.y - self.y * other.x
    )
end

-- Advanced Game Entity System
-- ===========================

local Entity = createClass("Entity")

function Entity:init(id)
    self.id = id or math.random(1000000)
    self.components = {}
    self.active = true
end

function Entity:addComponent(componentType, ...)
    local component = componentType:new(self, ...)
    self.components[componentType.__name] = component
    return component
end

function Entity:getComponent(componentType)
    return self.components[componentType.__name]
end

function Entity:hasComponent(componentType)
    return self.components[componentType.__name] ~= nil
end

function Entity:removeComponent(componentType)
    self.components[componentType.__name] = nil
end

-- Component classes
local PositionComponent = createClass("PositionComponent")
function PositionComponent:init(entity, x, y, z)
    self.entity = entity
    self.position = Vector:new(x, y, z)
end

local VelocityComponent = createClass("VelocityComponent")
function VelocityComponent:init(entity, vx, vy, vz)
    self.entity = entity
    self.velocity = Vector:new(vx, vy, vz)
end

local HealthComponent = createClass("HealthComponent")
function HealthComponent:init(entity, maxHealth)
    self.entity = entity
    self.maxHealth = maxHealth or 100
    self.currentHealth = self.maxHealth
end

function HealthComponent:takeDamage(amount)
    self.currentHealth = math.max(0, self.currentHealth - amount)
    return self.currentHealth <= 0
end

function HealthComponent:heal(amount)
    self.currentHealth = math.min(self.maxHealth, self.currentHealth + amount)
end

-- System class for processing entities
local System = createClass("System")

function System:init()
    self.entities = {}
end

function System:addEntity(entity)
    if self:canProcess(entity) then
        table.insert(self.entities, entity)
    end
end

function System:removeEntity(entity)
    for i, e in ipairs(self.entities) do
        if e.id == entity.id then
            table.remove(self.entities, i)
            break
        end
    end
end

function System:canProcess(entity)
    -- Override in subclasses
    return true
end

function System:update(deltaTime)
    for _, entity in ipairs(self.entities) do
        if entity.active then
            self:processEntity(entity, deltaTime)
        end
    end
end

function System:processEntity(entity, deltaTime)
    -- Override in subclasses
end

-- Movement system example
local MovementSystem = System:extend("MovementSystem")

function MovementSystem:canProcess(entity)
    return entity:hasComponent(PositionComponent) and entity:hasComponent(VelocityComponent)
end

function MovementSystem:processEntity(entity, deltaTime)
    local position = entity:getComponent(PositionComponent)
    local velocity = entity:getComponent(VelocityComponent)
    
    position.position = position.position + (velocity.velocity * deltaTime)
end

-- Coroutines for Game Logic
-- =========================

-- Coroutine-based tween system
local function createTween(startValue, endValue, duration, easingFunction)
    return coroutine.create(function()
        local startTime = os.clock()
        local elapsed = 0
        
        while elapsed < duration do
            elapsed = os.clock() - startTime
            local progress = math.min(elapsed / duration, 1)
            local easedProgress = easingFunction and easingFunction(progress) or progress
            local currentValue = startValue + (endValue - startValue) * easedProgress
            
            coroutine.yield(currentValue)
        end
        
        return endValue
    end)
end

-- Easing functions
local Easing = {
    linear = function(t) return t end,
    
    easeInQuad = function(t) return t * t end,
    
    easeOutQuad = function(t) return t * (2 - t) end,
    
    easeInOutQuad = function(t)
        if t < 0.5 then
            return 2 * t * t
        else
            return -1 + (4 - 2 * t) * t
        end
    end,
    
    easeInCubic = function(t) return t * t * t end,
    
    easeOutCubic = function(t)
        local t1 = t - 1
        return t1 * t1 * t1 + 1
    end,
    
    easeInSine = function(t) return 1 - math.cos(t * math.pi / 2) end,
    
    easeOutSine = function(t) return math.sin(t * math.pi / 2) end
}

-- State machine using coroutines
local function createStateMachine(states, initialState)
    local machine = {
        currentState = initialState,
        states = states,
        context = {}
    }
    
    function machine:setState(stateName, ...)
        if self.states[stateName] then
            self.currentState = stateName
            self.stateCoroutine = coroutine.create(self.states[stateName])
            local success, result = coroutine.resume(self.stateCoroutine, self, ...)
            if not success then
                error("State machine error: " .. tostring(result))
            end
        end
    end
    
    function machine:update(...)
        if self.stateCoroutine and coroutine.status(self.stateCoroutine) ~= "dead" then
            local success, result = coroutine.resume(self.stateCoroutine, ...)
            if not success then
                error("State machine error: " .. tostring(result))
            end
            return result
        end
    end
    
    -- Initialize with the initial state
    machine:setState(initialState)
    
    return machine
end

-- Data Structures and Algorithms
-- ==============================

-- Binary Search Tree implementation
local BinarySearchTree = createClass("BinarySearchTree")

function BinarySearchTree:init(compare)
    self.root = nil
    self.compare = compare or function(a, b)
        if a < b then return -1
        elseif a > b then return 1
        else return 0 end
    end
end

local TreeNode = createClass("TreeNode")
function TreeNode:init(value)
    self.value = value
    self.left = nil
    self.right = nil
end

function BinarySearchTree:insert(value)
    self.root = self:_insertNode(self.root, value)
end

function BinarySearchTree:_insertNode(node, value)
    if not node then
        return TreeNode:new(value)
    end
    
    local cmp = self.compare(value, node.value)
    if cmp < 0 then
        node.left = self:_insertNode(node.left, value)
    elseif cmp > 0 then
        node.right = self:_insertNode(node.right, value)
    end
    
    return node
end

function BinarySearchTree:search(value)
    return self:_searchNode(self.root, value)
end

function BinarySearchTree:_searchNode(node, value)
    if not node then
        return false
    end
    
    local cmp = self.compare(value, node.value)
    if cmp == 0 then
        return true
    elseif cmp < 0 then
        return self:_searchNode(node.left, value)
    else
        return self:_searchNode(node.right, value)
    end
end

function BinarySearchTree:inorderTraversal()
    local result = {}
    self:_inorderTraversal(self.root, result)
    return result
end

function BinarySearchTree:_inorderTraversal(node, result)
    if node then
        self:_inorderTraversal(node.left, result)
        table.insert(result, node.value)
        self:_inorderTraversal(node.right, result)
    end
end

-- Priority Queue using binary heap
local PriorityQueue = createClass("PriorityQueue")

function PriorityQueue:init(compare)
    self.heap = {}
    self.compare = compare or function(a, b) return a < b end
end

function PriorityQueue:push(item)
    table.insert(self.heap, item)
    self:_bubbleUp(#self.heap)
end

function PriorityQueue:pop()
    if #self.heap == 0 then return nil end
    
    local top = self.heap[1]
    self.heap[1] = self.heap[#self.heap]
    table.remove(self.heap)
    
    if #self.heap > 0 then
        self:_bubbleDown(1)
    end
    
    return top
end

function PriorityQueue:peek()
    return self.heap[1]
end

function PriorityQueue:size()
    return #self.heap
end

function PriorityQueue:_bubbleUp(index)
    while index > 1 do
        local parentIndex = math.floor(index / 2)
        if not self.compare(self.heap[index], self.heap[parentIndex]) then
            break
        end
        
        self.heap[index], self.heap[parentIndex] = self.heap[parentIndex], self.heap[index]
        index = parentIndex
    end
end

function PriorityQueue:_bubbleDown(index)
    local size = #self.heap
    
    while true do
        local minIndex = index
        local leftChild = 2 * index
        local rightChild = 2 * index + 1
        
        if leftChild <= size and self.compare(self.heap[leftChild], self.heap[minIndex]) then
            minIndex = leftChild
        end
        
        if rightChild <= size and self.compare(self.heap[rightChild], self.heap[minIndex]) then
            minIndex = rightChild
        end
        
        if minIndex == index then break end
        
        self.heap[index], self.heap[minIndex] = self.heap[minIndex], self.heap[index]
        index = minIndex
    end
end

-- Functional Programming Utilities
-- ================================

-- Higher-order functions
local function map(func, table)
    local result = {}
    for i, v in ipairs(table) do
        result[i] = func(v)
    end
    return result
end

local function filter(predicate, table)
    local result = {}
    for _, v in ipairs(table) do
        if predicate(v) then
            table.insert(result, v)
        end
    end
    return result
end

local function reduce(func, table, initialValue)
    local accumulator = initialValue
    for _, v in ipairs(table) do
        accumulator = func(accumulator, v)
    end
    return accumulator
end

local function curry(func, arity)
    arity = arity or 2
    
    return function(...)
        local args = {...}
        if #args >= arity then
            return func(table.unpack(args))
        else
            return function(...)
                local newArgs = {}
                for _, arg in ipairs(args) do
                    table.insert(newArgs, arg)
                end
                for _, arg in ipairs({...}) do
                    table.insert(newArgs, arg)
                end
                return curry(func, arity)(table.unpack(newArgs))
            end
        end
    end
end

local function compose(...)
    local functions = {...}
    return function(value)
        for i = #functions, 1, -1 do
            value = functions[i](value)
        end
        return value
    end
end

-- Memoization
local function memoize(func)
    local cache = {}
    return function(...)
        local key = table.concat({...}, ",")
        if cache[key] == nil then
            cache[key] = func(...)
        end
        return cache[key]
    end
end

-- Advanced String and Pattern Matching
-- ====================================

-- String interpolation utility
local function interpolate(template, values)
    return template:gsub("%${([^}]+)}", function(key)
        return tostring(values[key] or "")
    end)
end

-- Advanced pattern matching for parsing
local function parseCSV(csvString)
    local result = {}
    local currentField = ""
    local inQuotes = false
    local currentRow = {}
    
    for i = 1, #csvString do
        local char = csvString:sub(i, i)
        
        if char == '"' then
            inQuotes = not inQuotes
        elseif char == ',' and not inQuotes then
            table.insert(currentRow, currentField)
            currentField = ""
        elseif char == '\n' and not inQuotes then
            table.insert(currentRow, currentField)
            table.insert(result, currentRow)
            currentRow = {}
            currentField = ""
        else
            currentField = currentField .. char
        end
    end
    
    -- Add the last field and row
    if currentField ~= "" or #currentRow > 0 then
        table.insert(currentRow, currentField)
        table.insert(result, currentRow)
    end
    
    return result
end

-- JSON-like serialization
local function serialize(value, indent)
    indent = indent or 0
    local indentStr = string.rep("  ", indent)
    
    if type(value) == "table" then
        local isArray = true
        local maxIndex = 0
        
        -- Check if table is an array
        for k, v in pairs(value) do
            if type(k) ~= "number" or k ~= math.floor(k) or k < 1 then
                isArray = false
                break
            end
            maxIndex = math.max(maxIndex, k)
        end
        
        if isArray and maxIndex == #value then
            -- Array format
            local result = "[\n"
            for i, v in ipairs(value) do
                result = result .. indentStr .. "  " .. serialize(v, indent + 1)
                if i < #value then result = result .. "," end
                result = result .. "\n"
            end
            result = result .. indentStr .. "]"
            return result
        else
            -- Object format
            local result = "{\n"
            local first = true
            for k, v in pairs(value) do
                if not first then result = result .. ",\n" end
                result = result .. indentStr .. "  " .. tostring(k) .. ": " .. serialize(v, indent + 1)
                first = false
            end
            result = result .. "\n" .. indentStr .. "}"
            return result
        end
    elseif type(value) == "string" then
        return '"' .. value:gsub('"', '\\"') .. '"'
    else
        return tostring(value)
    end
end

-- Performance Optimization Utilities
-- ==================================

-- Object pooling for memory management
local function createObjectPool(createFunc, resetFunc, initialSize)
    local pool = {
        objects = {},
        createFunc = createFunc,
        resetFunc = resetFunc or function() end
    }
    
    -- Pre-populate the pool
    for i = 1, (initialSize or 10) do
        table.insert(pool.objects, createFunc())
    end
    
    function pool:acquire()
        if #self.objects > 0 then
            return table.remove(self.objects)
        else
            return self.createFunc()
        end
    end
    
    function pool:release(obj)
        self.resetFunc(obj)
        table.insert(self.objects, obj)
    end
    
    return pool
end

-- Benchmarking utility
local function benchmark(name, func, iterations)
    iterations = iterations or 1000
    
    collectgarbage("collect")
    local startTime = os.clock()
    local startMemory = collectgarbage("count")
    
    for i = 1, iterations do
        func()
    end
    
    local endTime = os.clock()
    collectgarbage("collect")
    local endMemory = collectgarbage("count")
    
    local totalTime = endTime - startTime
    local avgTime = totalTime / iterations
    local memoryUsed = endMemory - startMemory
    
    print(string.format("Benchmark '%s':", name))
    print(string.format("  Total time: %.4f seconds", totalTime))
    print(string.format("  Average time: %.6f seconds", avgTime))
    print(string.format("  Iterations: %d", iterations))
    print(string.format("  Memory used: %.2f KB", memoryUsed))
end

-- Example Usage and Testing
-- =========================

local function runExamples()
    print("=== Advanced Lua Programming Examples ===\n")
    
    -- 1. Vector mathematics
    print("1. Vector Mathematics:")
    local v1 = Vector:new(1, 2, 3)
    local v2 = Vector:new(4, 5, 6)
    local v3 = v1 + v2
    local v4 = v1 * 2
    
    print("   v1:", v1)
    print("   v2:", v2)
    print("   v1 + v2:", v3)
    print("   v1 * 2:", v4)
    print("   v1 · v2:", v1:dot(v2))
    print("   |v1|:", v1:magnitude())
    
    -- 2. Entity Component System
    print("\n2. Entity Component System:")
    local entity = Entity:new()
    entity:addComponent(PositionComponent, 10, 20, 0)
    entity:addComponent(VelocityComponent, 1, -0.5, 0)
    entity:addComponent(HealthComponent, 100)
    
    local movementSystem = MovementSystem:new()
    movementSystem:addEntity(entity)
    
    print("   Entity created with components")
    local pos = entity:getComponent(PositionComponent)
    print("   Initial position:", pos.position)
    
    movementSystem:update(1.0) -- 1 second update
    print("   Position after 1s:", pos.position)
    
    -- 3. Coroutine-based tween
    print("\n3. Coroutine Tween:")
    local tween = createTween(0, 100, 1, Easing.easeInOutQuad)
    
    print("   Tween values:")
    for i = 1, 5 do
        local success, value = coroutine.resume(tween)
        if success and value then
            print(string.format("     Step %d: %.2f", i, value))
        end
    end
    
    -- 4. Data structures
    print("\n4. Data Structures:")
    
    -- Binary Search Tree
    local bst = BinarySearchTree:new()
    local values = {50, 30, 70, 20, 40, 60, 80}
    
    for _, value in ipairs(values) do
        bst:insert(value)
    end
    
    print("   BST inorder traversal:", table.concat(bst:inorderTraversal(), ", "))
    print("   BST contains 40:", bst:search(40))
    print("   BST contains 25:", bst:search(25))
    
    -- Priority Queue
    local pq = PriorityQueue:new()
    for _, value in ipairs({3, 1, 4, 1, 5, 9, 2, 6}) do
        pq:push(value)
    end
    
    print("   Priority Queue (ascending):")
    local pqResults = {}
    while pq:size() > 0 do
        table.insert(pqResults, pq:pop())
    end
    print("     " .. table.concat(pqResults, ", "))
    
    -- 5. Functional programming
    print("\n5. Functional Programming:")
    local numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    
    local squares = map(function(x) return x * x end, numbers)
    print("   Squares:", table.concat(squares, ", "))
    
    local evens = filter(function(x) return x % 2 == 0 end, numbers)
    print("   Evens:", table.concat(evens, ", "))
    
    local sum = reduce(function(acc, x) return acc + x end, numbers, 0)
    print("   Sum:", sum)
    
    -- Memoized fibonacci
    local fibMemo = memoize(function(n)
        if n <= 1 then return n end
        return fibMemo(n - 1) + fibMemo(n - 2)
    end)
    
    print("   Fibonacci(20):", fibMemo(20))
    
    -- 6. String operations
    print("\n6. String Operations:")
    local template = "Hello ${name}, you have ${count} messages!"
    local result = interpolate(template, {name = "Alice", count = 5})
    print("   Interpolated:", result)
    
    local csvData = 'name,age,city\n"John Doe",30,"New York"\n"Jane Smith",25,"Los Angeles"'
    local parsed = parseCSV(csvData)
    print("   Parsed CSV:")
    for i, row in ipairs(parsed) do
        print("     Row " .. i .. ":", table.concat(row, ", "))
    end
    
    -- 7. Serialization
    print("\n7. Serialization:")
    local data = {
        name = "Test Object",
        values = {1, 2, 3, 4, 5},
        nested = {
            x = 10,
            y = 20
        }
    }
    
    print("   Serialized object:")
    print(serialize(data))
    
    -- 8. Performance testing
    print("\n8. Performance Testing:")
    benchmark("Table creation", function()
        local t = {}
        for i = 1, 100 do
            t[i] = i * i
        end
    end, 1000)
    
    print("\n=== Lua Examples Complete ===")
end

-- State machine example for game AI
local function createEnemyAI()
    local states = {
        idle = function(machine, dt)
            machine.context.idleTime = (machine.context.idleTime or 0) + dt
            if machine.context.idleTime > 2 then
                machine:setState("patrol")
            end
            coroutine.yield("idling")
        end,
        
        patrol = function(machine, dt)
            machine.context.patrolTime = (machine.context.patrolTime or 0) + dt
            if machine.context.patrolTime > 5 then
                machine:setState("idle")
            elseif machine.context.playerDetected then
                machine:setState("chase")
            end
            coroutine.yield("patrolling")
        end,
        
        chase = function(machine, dt)
            if not machine.context.playerDetected then
                machine:setState("idle")
            end
            coroutine.yield("chasing")
        end
    }
    
    return createStateMachine(states, "idle")
end

-- Run examples if this file is executed directly
if arg and arg[0] and arg[0]:match("lua_advanced_programming%.lua$") then
    runExamples()
end

-- Export functions for module usage
return {
    createClass = createClass,
    Vector = Vector,
    Entity = Entity,
    System = System,
    MovementSystem = MovementSystem,
    BinarySearchTree = BinarySearchTree,
    PriorityQueue = PriorityQueue,
    createTween = createTween,
    Easing = Easing,
    createStateMachine = createStateMachine,
    createEnemyAI = createEnemyAI,
    map = map,
    filter = filter,
    reduce = reduce,
    curry = curry,
    compose = compose,
    memoize = memoize,
    interpolate = interpolate,
    parseCSV = parseCSV,
    serialize = serialize,
    createObjectPool = createObjectPool,
    benchmark = benchmark,
    runExamples = runExamples
}