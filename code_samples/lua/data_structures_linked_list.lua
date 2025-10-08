-- Data Structures: Linked List
-- AI/ML Training Sample

LinkedList = {}
LinkedList.__ index = LinkedList

function LinkedList:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function LinkedList:process(input)
    self.data = input
end

function LinkedList:getData()
    return self.data
end

function LinkedList:validate()
    return #self.data > 0
end

-- Example usage
local instance = LinkedList:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
