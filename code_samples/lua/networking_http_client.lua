-- Networking: Http Client
-- AI/ML Training Sample

HttpClient = {}
HttpClient.__ index = HttpClient

function HttpClient:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function HttpClient:process(input)
    self.data = input
end

function HttpClient:getData()
    return self.data
end

function HttpClient:validate()
    return #self.data > 0
end

-- Example usage
local instance = HttpClient:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
