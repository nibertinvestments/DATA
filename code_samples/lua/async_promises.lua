-- Async: Promises
-- AI/ML Training Sample

Promises = {}
Promises.__ index = Promises

function Promises:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Promises:process(input)
    self.data = input
end

function Promises:getData()
    return self.data
end

function Promises:validate()
    return #self.data > 0
end

-- Example usage
local instance = Promises:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
