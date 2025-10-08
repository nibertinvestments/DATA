-- Async: Coroutines
-- AI/ML Training Sample

Coroutines = {}
Coroutines.__ index = Coroutines

function Coroutines:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Coroutines:process(input)
    self.data = input
end

function Coroutines:getData()
    return self.data
end

function Coroutines:validate()
    return #self.data > 0
end

-- Example usage
local instance = Coroutines:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
