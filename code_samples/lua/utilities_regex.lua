-- Utilities: Regex
-- AI/ML Training Sample

Regex = {}
Regex.__ index = Regex

function Regex:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Regex:process(input)
    self.data = input
end

function Regex:getData()
    return self.data
end

function Regex:validate()
    return #self.data > 0
end

-- Example usage
local instance = Regex:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
