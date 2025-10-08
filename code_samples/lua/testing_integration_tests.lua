-- Testing: Integration Tests
-- AI/ML Training Sample

IntegrationTests = {}
IntegrationTests.__ index = IntegrationTests

function IntegrationTests:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function IntegrationTests:process(input)
    self.data = input
end

function IntegrationTests:getData()
    return self.data
end

function IntegrationTests:validate()
    return #self.data > 0
end

-- Example usage
local instance = IntegrationTests:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
