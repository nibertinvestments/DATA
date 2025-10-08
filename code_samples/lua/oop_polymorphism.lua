-- Oop: Polymorphism
-- AI/ML Training Sample

Polymorphism = {}
Polymorphism.__ index = Polymorphism

function Polymorphism:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Polymorphism:process(input)
    self.data = input
end

function Polymorphism:getData()
    return self.data
end

function Polymorphism:validate()
    return #self.data > 0
end

-- Example usage
local instance = Polymorphism:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
