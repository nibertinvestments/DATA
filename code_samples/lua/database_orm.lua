-- Database: Orm
-- AI/ML Training Sample

Orm = {}
Orm.__ index = Orm

function Orm:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Orm:process(input)
    self.data = input
end

function Orm:getData()
    return self.data
end

function Orm:validate()
    return #self.data > 0
end

-- Example usage
local instance = Orm:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
