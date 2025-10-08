-- Data Structures: Heap
-- AI/ML Training Sample

Heap = {}
Heap.__ index = Heap

function Heap:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Heap:process(input)
    self.data = input
end

function Heap:getData()
    return self.data
end

function Heap:validate()
    return #self.data > 0
end

-- Example usage
local instance = Heap:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
