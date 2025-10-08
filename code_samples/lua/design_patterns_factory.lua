-- Design Patterns: Factory
-- AI/ML Training Sample

Factory = {}
Factory.__ index = Factory

function Factory:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Factory:process(input)
    self.data = input
end

function Factory:getData()
    return self.data
end

function Factory:validate()
    return #self.data > 0
end

-- Example usage
local instance = Factory:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
