-- Functional: Map Reduce
-- AI/ML Training Sample

MapReduce = {}
MapReduce.__ index = MapReduce

function MapReduce:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function MapReduce:process(input)
    self.data = input
end

function MapReduce:getData()
    return self.data
end

function MapReduce:validate()
    return #self.data > 0
end

-- Example usage
local instance = MapReduce:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
