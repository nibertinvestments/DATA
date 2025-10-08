-- Testing: Unit Tests
-- AI/ML Training Sample

UnitTests = {}
UnitTests.__ index = UnitTests

function UnitTests:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function UnitTests:process(input)
    self.data = input
end

function UnitTests:getData()
    return self.data
end

function UnitTests:validate()
    return #self.data > 0
end

-- Example usage
local instance = UnitTests:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
