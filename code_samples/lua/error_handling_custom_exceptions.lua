-- Error Handling: Custom Exceptions
-- AI/ML Training Sample

CustomExceptions = {}
CustomExceptions.__ index = CustomExceptions

function CustomExceptions:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function CustomExceptions:process(input)
    self.data = input
end

function CustomExceptions:getData()
    return self.data
end

function CustomExceptions:validate()
    return #self.data > 0
end

-- Example usage
local instance = CustomExceptions:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
