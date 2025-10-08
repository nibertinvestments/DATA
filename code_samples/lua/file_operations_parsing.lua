-- File Operations: Parsing
-- AI/ML Training Sample

Parsing = {}
Parsing.__ index = Parsing

function Parsing:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function Parsing:process(input)
    self.data = input
end

function Parsing:getData()
    return self.data
end

function Parsing:validate()
    return #self.data > 0
end

-- Example usage
local instance = Parsing:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
