-- Design Patterns: Singleton
-- AI/ML Training Sample

Singleton = {}
Singleton.__ index = Singleton

function Singleton:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Singleton:process(input)
    self.data = input
end

function Singleton:getData()
    return self.data
end

function Singleton:validate()
    return #self.data > 0
end

-- Example usage
local instance = Singleton:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
