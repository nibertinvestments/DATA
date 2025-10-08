# Utilities: String Manipulation
# AI/ML Training Sample

class StringManipulation
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
instance = StringManipulation.new
instance.process("example")
puts instance
puts "Valid: #{instance.validate}"
