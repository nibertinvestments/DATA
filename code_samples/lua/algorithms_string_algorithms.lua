-- Algorithms: String Algorithms
-- AI/ML Training Sample

StringAlgorithms = {}
StringAlgorithms.__ index = StringAlgorithms

function StringAlgorithms:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function StringAlgorithms:process(input)
    self.data = input
end

function StringAlgorithms:getData()
    return self.data
end

function StringAlgorithms:validate()
    return #self.data > 0
end

-- Example usage
local instance = StringAlgorithms:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
