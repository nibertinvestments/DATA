-- Algorithms: Searching
-- AI/ML Training Sample

Searching = {}
Searching.__ index = Searching

function Searching:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Searching:process(input)
    self.data = input
end

function Searching:getData()
    return self.data
end

function Searching:validate()
    return #self.data > 0
end

-- Example usage
local instance = Searching:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
