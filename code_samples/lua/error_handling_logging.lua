-- Error Handling: Logging
-- AI/ML Training Sample

Logging = {}
Logging.__ index = Logging

function Logging:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Logging:process(input)
    self.data = input
end

function Logging:getData()
    return self.data
end

function Logging:validate()
    return #self.data > 0
end

-- Example usage
local instance = Logging:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
