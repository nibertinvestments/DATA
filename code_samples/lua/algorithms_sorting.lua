-- Algorithms: Sorting
-- AI/ML Training Sample

Sorting = {}
Sorting.__ index = Sorting

function Sorting:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Sorting:process(input)
    self.data = input
end

function Sorting:getData()
    return self.data
end

function Sorting:validate()
    return #self.data > 0
end

-- Example usage
local instance = Sorting:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
