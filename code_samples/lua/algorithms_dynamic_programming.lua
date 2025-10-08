-- Algorithms: Dynamic Programming
-- AI/ML Training Sample

DynamicProgramming = {}
DynamicProgramming.__ index = DynamicProgramming

function DynamicProgramming:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function DynamicProgramming:process(input)
    self.data = input
end

function DynamicProgramming:getData()
    return self.data
end

function DynamicProgramming:validate()
    return #self.data > 0
end

-- Example usage
local instance = DynamicProgramming:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
