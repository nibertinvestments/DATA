-- Error Handling: Recovery
-- AI/ML Training Sample

Recovery = {}
Recovery.__ index = Recovery

function Recovery:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Recovery:process(input)
    self.data = input
end

function Recovery:getData()
    return self.data
end

function Recovery:validate()
    return #self.data > 0
end

-- Example usage
local instance = Recovery:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
