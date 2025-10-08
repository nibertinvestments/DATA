-- Functional: Monads
-- AI/ML Training Sample

Monads = {}
Monads.__ index = Monads

function Monads:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Monads:process(input)
    self.data = input
end

function Monads:getData()
    return self.data
end

function Monads:validate()
    return #self.data > 0
end

-- Example usage
local instance = Monads:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
