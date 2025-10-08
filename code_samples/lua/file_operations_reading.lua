-- File Operations: Reading
-- AI/ML Training Sample

Reading = {}
Reading.__ index = Reading

function Reading:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Reading:process(input)
    self.data = input
end

function Reading:getData()
    return self.data
end

function Reading:validate()
    return #self.data > 0
end

-- Example usage
local instance = Reading:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
