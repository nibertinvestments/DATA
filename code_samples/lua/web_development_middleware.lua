-- Web Development: Middleware
-- AI/ML Training Sample

Middleware = {}
Middleware.__ index = Middleware

function Middleware:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Middleware:process(input)
    self.data = input
end

function Middleware:getData()
    return self.data
end

function Middleware:validate()
    return #self.data > 0
end

-- Example usage
local instance = Middleware:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
