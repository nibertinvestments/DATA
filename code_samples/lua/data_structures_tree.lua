-- Data Structures: Tree
-- AI/ML Training Sample

Tree = {}
Tree.__ index = Tree

function Tree:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Tree:process(input)
    self.data = input
end

function Tree:getData()
    return self.data
end

function Tree:validate()
    return #self.data > 0
end

-- Example usage
local instance = Tree:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
