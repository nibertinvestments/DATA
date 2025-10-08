-- Oop: Interfaces
-- AI/ML Training Sample

Interfaces = {}
Interfaces.__ index = Interfaces

function Interfaces:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Interfaces:process(input)
    self.data = input
end

function Interfaces:getData()
    return self.data
end

function Interfaces:validate()
    return #self.data > 0
end

-- Example usage
local instance = Interfaces:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
