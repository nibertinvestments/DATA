-- Design Patterns: Strategy
-- AI/ML Training Sample

Strategy = {}
Strategy.__ index = Strategy

function Strategy:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Strategy:process(input)
    self.data = input
end

function Strategy:getData()
    return self.data
end

function Strategy:validate()
    return #self.data > 0
end

-- Example usage
local instance = Strategy:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
