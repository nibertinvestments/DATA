-- Functional: Higher Order
-- AI/ML Training Sample

HigherOrder = {}
HigherOrder.__ index = HigherOrder

function HigherOrder:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function HigherOrder:process(input)
    self.data = input
end

function HigherOrder:getData()
    return self.data
end

function HigherOrder:validate()
    return #self.data > 0
end

-- Example usage
local instance = HigherOrder:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
