-- Design Patterns: Decorator
-- AI/ML Training Sample

Decorator = {}
Decorator.__ index = Decorator

function Decorator:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Decorator:process(input)
    self.data = input
end

function Decorator:getData()
    return self.data
end

function Decorator:validate()
    return #self.data > 0
end

-- Example usage
local instance = Decorator:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
