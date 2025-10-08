-- Functional: Currying
-- AI/ML Training Sample

Currying = {}
Currying.__ index = Currying

function Currying:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Currying:process(input)
    self.data = input
end

function Currying:getData()
    return self.data
end

function Currying:validate()
    return #self.data > 0
end

-- Example usage
local instance = Currying:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
