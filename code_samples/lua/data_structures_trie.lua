-- Data Structures: Trie
-- AI/ML Training Sample

Trie = {}
Trie.__ index = Trie

function Trie:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Trie:process(input)
    self.data = input
end

function Trie:getData()
    return self.data
end

function Trie:validate()
    return #self.data > 0
end

-- Example usage
local instance = Trie:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
