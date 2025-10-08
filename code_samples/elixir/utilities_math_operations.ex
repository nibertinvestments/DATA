# Utilities: Math Operations
# AI/ML Training Sample

defmodule MathOperations do
  defstruct data: ""
  
  def new(), do: %MathOperations{}
  
  def process(%MathOperations{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%MathOperations{data: data}), do: data
  
  def validate(%MathOperations{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = MathOperations.new()
updated = MathOperations.process(instance, "example")
IO.puts("Data: " <> MathOperations.get_data(updated))
IO.puts("Valid: " <> to_string(MathOperations.validate(updated)))
