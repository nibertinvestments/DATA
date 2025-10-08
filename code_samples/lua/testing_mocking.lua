-- Testing: Mocking
-- AI/ML Training Sample

Mocking = {}
Mocking.__ index = Mocking

function Mocking:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Mocking:process(input)
    self.data = input
end

function Mocking:getData()
    return self.data
end

function Mocking:validate()
    return #self.data > 0
end

-- Example usage
local instance = Mocking:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
