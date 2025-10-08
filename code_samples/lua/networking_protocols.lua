-- Networking: Protocols
-- AI/ML Training Sample

Protocols = {}
Protocols.__ index = Protocols

function Protocols:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Protocols:process(input)
    self.data = input
end

function Protocols:getData()
    return self.data
end

function Protocols:validate()
    return #self.data > 0
end

-- Example usage
local instance = Protocols:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
