-- Web Development: Validation
-- AI/ML Training Sample

Validation = {}
Validation.__ index = Validation

function Validation:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Validation:process(input)
    self.data = input
end

function Validation:getData()
    return self.data
end

function Validation:validate()
    return #self.data > 0
end

-- Example usage
local instance = Validation:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
