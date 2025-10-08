-- Testing: Fixtures
-- AI/ML Training Sample

Fixtures = {}
Fixtures.__ index = Fixtures

function Fixtures:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Fixtures:process(input)
    self.data = input
end

function Fixtures:getData()
    return self.data
end

function Fixtures:validate()
    return #self.data > 0
end

-- Example usage
local instance = Fixtures:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
