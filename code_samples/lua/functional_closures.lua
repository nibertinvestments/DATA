-- Functional: Closures
-- AI/ML Training Sample

Closures = {}
Closures.__ index = Closures

function Closures:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Closures:process(input)
    self.data = input
end

function Closures:getData()
    return self.data
end

function Closures:validate()
    return #self.data > 0
end

-- Example usage
local instance = Closures:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
