-- Performance: Batching
-- AI/ML Training Sample

Batching = {}
Batching.__ index = Batching

function Batching:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Batching:process(input)
    self.data = input
end

function Batching:getData()
    return self.data
end

function Batching:validate()
    return #self.data > 0
end

-- Example usage
local instance = Batching:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
