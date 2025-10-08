-- Oop: Inheritance
-- AI/ML Training Sample

Inheritance = {}
Inheritance.__ index = Inheritance

function Inheritance:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Inheritance:process(input)
    self.data = input
end

function Inheritance:getData()
    return self.data
end

function Inheritance:validate()
    return #self.data > 0
end

-- Example usage
local instance = Inheritance:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
