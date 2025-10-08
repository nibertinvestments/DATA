# Testing: Unit Tests
# AI/ML Training Sample

defmodule UnitTests do
  defstruct data: ""
  
  def new(), do: %UnitTests{}
  
  def process(%UnitTests{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%UnitTests{data: data}), do: data
  
  def validate(%UnitTests{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = UnitTests.new()
updated = UnitTests.process(instance, "example")
IO.puts("Data: " <> UnitTests.get_data(updated))
IO.puts("Valid: " <> to_string(UnitTests.validate(updated)))
