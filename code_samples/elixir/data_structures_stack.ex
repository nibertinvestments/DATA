# Data Structures: Stack
# AI/ML Training Sample

defmodule Stack do
  defstruct data: ""
  
  def new(), do: %Stack{}
  
  def process(%Stack{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Stack{data: data}), do: data
  
  def validate(%Stack{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Stack.new()
updated = Stack.process(instance, "example")
IO.puts("Data: " <> Stack.get_data(updated))
IO.puts("Valid: " <> to_string(Stack.validate(updated)))
