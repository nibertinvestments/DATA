-- Data Structures: Stack
-- AI/ML Training Sample

Stack = {}
Stack.__ index = Stack

function Stack:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Stack:process(input)
    self.data = input
end

function Stack:getData()
    return self.data
end

function Stack:validate()
    return #self.data > 0
end

-- Example usage
local instance = Stack:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
