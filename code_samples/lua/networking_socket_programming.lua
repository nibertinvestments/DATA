-- Networking: Socket Programming
-- AI/ML Training Sample

SocketProgramming = {}
SocketProgramming.__ index = SocketProgramming

function SocketProgramming:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function SocketProgramming:process(input)
    self.data = input
end

function SocketProgramming:getData()
    return self.data
end

function SocketProgramming:validate()
    return #self.data > 0
end

-- Example usage
local instance = SocketProgramming:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
