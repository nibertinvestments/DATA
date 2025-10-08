-- Oop: Abstraction
-- AI/ML Training Sample

Abstraction = {}
Abstraction.__ index = Abstraction

function Abstraction:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Abstraction:process(input)
    self.data = input
end

function Abstraction:getData()
    return self.data
end

function Abstraction:validate()
    return #self.data > 0
end

-- Example usage
local instance = Abstraction:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
