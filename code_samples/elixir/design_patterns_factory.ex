# Design Patterns: Factory
# AI/ML Training Sample

defmodule Factory do
  defstruct data: ""
  
  def new(), do: %Factory{}
  
  def process(%Factory{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Factory{data: data}), do: data
  
  def validate(%Factory{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Factory.new()
updated = Factory.process(instance, "example")
IO.puts("Data: " <> Factory.get_data(updated))
IO.puts("Valid: " <> to_string(Factory.validate(updated)))
