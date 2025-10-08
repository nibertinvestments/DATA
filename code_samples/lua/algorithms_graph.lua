-- Algorithms: Graph
-- AI/ML Training Sample

Graph = {}
Graph.__ index = Graph

function Graph:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Graph:process(input)
    self.data = input
end

function Graph:getData()
    return self.data
end

function Graph:validate()
    return #self.data > 0
end

-- Example usage
local instance = Graph:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
