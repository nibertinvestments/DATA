-- Security: Input Validation
-- AI/ML Training Sample

InputValidation = {}
InputValidation.__ index = InputValidation

function InputValidation:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function InputValidation:process(input)
    self.data = input
end

function InputValidation:getData()
    return self.data
end

function InputValidation:validate()
    return #self.data > 0
end

-- Example usage
local instance = InputValidation:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
