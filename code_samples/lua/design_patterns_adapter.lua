-- Design Patterns: Adapter
-- AI/ML Training Sample

Adapter = {}
Adapter.__ index = Adapter

function Adapter:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Adapter:process(input)
    self.data = input
end

function Adapter:getData()
    return self.data
end

function Adapter:validate()
    return #self.data > 0
end

-- Example usage
local instance = Adapter:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
