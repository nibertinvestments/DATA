-- Performance: Caching
-- AI/ML Training Sample

Caching = {}
Caching.__ index = Caching

function Caching:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Caching:process(input)
    self.data = input
end

function Caching:getData()
    return self.data
end

function Caching:validate()
    return #self.data > 0
end

-- Example usage
local instance = Caching:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
