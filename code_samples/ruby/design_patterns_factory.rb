# Design Patterns: Factory
# AI/ML Training Sample

class Factory
  attr_accessor :data
  
  def initialize
    @data = ""
  end
  
  def process(input)
    @data = input
  end
  
  def validate
    !@data.empty?
  end
  
  def to_s
    "Data: #{@data}"
  end
end

# Example usage
instance = Factory.new
instance.process("example")
puts instance
puts "Valid: #{instance.validate}"
