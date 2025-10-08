-- Error Handling: Error Propagation
-- AI/ML Training Sample

ErrorPropagation = {}
ErrorPropagation.__ index = ErrorPropagation

function ErrorPropagation:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function ErrorPropagation:process(input)
    self.data = input
end

function ErrorPropagation:getData()
    return self.data
end

function ErrorPropagation:validate()
    return #self.data > 0
end

-- Example usage
local instance = ErrorPropagation:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
