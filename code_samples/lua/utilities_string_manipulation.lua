-- Utilities: String Manipulation
-- AI/ML Training Sample

StringManipulation = {}
StringManipulation.__ index = StringManipulation

function StringManipulation:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function StringManipulation:process(input)
    self.data = input
end

function StringManipulation:getData()
    return self.data
end

function StringManipulation:validate()
    return #self.data > 0
end

-- Example usage
local instance = StringManipulation:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
