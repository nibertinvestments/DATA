-- Performance: Lazy Loading
-- AI/ML Training Sample

LazyLoading = {}
LazyLoading.__ index = LazyLoading

function LazyLoading:new()
    local obj = {
        data = ""
    }
    setmetatable(obj, self)
    return obj
end

function LazyLoading:process(input)
    self.data = input
end

function LazyLoading:getData()
    return self.data
end

function LazyLoading:validate()
    return #self.data > 0
end

-- Example usage
local instance = LazyLoading:new()
instance:process("example")
print("Data: " .. instance:getData())
print("Valid: " .. tostring(instance:validate()))
