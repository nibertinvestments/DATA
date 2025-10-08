-- Data Structures: Hash Table
-- AI/ML Training Sample

HashTable = {}
HashTable.__ index = HashTable

function HashTable:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function HashTable:process(input)
    self.data = input
end

function HashTable:getData()
    return self.data
end

function HashTable:validate()
    return #self.data > 0
end

-- Example usage
local instance = HashTable:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
