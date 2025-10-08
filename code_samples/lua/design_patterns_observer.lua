-- Design Patterns: Observer
-- AI/ML Training Sample

Observer = {}
Observer.__ index = Observer

function Observer:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Observer:process(input)
    self.data = input
end

function Observer:getData()
    return self.data
end

function Observer:validate()
    return #self.data > 0
end

-- Example usage
local instance = Observer:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
