-- Networking: Tcp Udp
-- AI/ML Training Sample

TcpUdp = {}
TcpUdp.__ index = TcpUdp

function TcpUdp:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function TcpUdp:process(input)
    self.data = input
end

function TcpUdp:getData()
    return self.data
end

function TcpUdp:validate()
    return #self.data > 0
end

-- Example usage
local instance = TcpUdp:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
