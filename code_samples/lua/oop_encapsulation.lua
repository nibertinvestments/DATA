-- Oop: Encapsulation
-- AI/ML Training Sample

Encapsulation = {}
Encapsulation.__ index = Encapsulation

function Encapsulation:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Encapsulation:process(input)
    self.data = input
end

function Encapsulation:getData()
    return self.data
end

function Encapsulation:validate()
    return #self.data > 0
end

-- Example usage
local instance = Encapsulation:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
