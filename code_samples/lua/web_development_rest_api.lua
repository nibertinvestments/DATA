-- Web Development: Rest Api
-- AI/ML Training Sample

RestApi = {}
RestApi.__ index = RestApi

function RestApi:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function RestApi:process(input)
    self.data = input
end

function RestApi:getData()
    return self.data
end

function RestApi:validate()
    return #self.data > 0
end

-- Example usage
local instance = RestApi:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
