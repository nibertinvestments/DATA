-- Database: Query Builder
-- AI/ML Training Sample

QueryBuilder = {}
QueryBuilder.__ index = QueryBuilder

function QueryBuilder:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function QueryBuilder:process(input)
    self.data = input
end

function QueryBuilder:getData()
    return self.data
end

function QueryBuilder:validate()
    return #self.data > 0
end

-- Example usage
local instance = QueryBuilder:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
