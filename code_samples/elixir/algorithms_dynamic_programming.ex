# Algorithms: Dynamic Programming
# AI/ML Training Sample

defmodule DynamicProgramming do
  defstruct data: ""
  
  def new(), do: %DynamicProgramming{}
  
  def process(%DynamicProgramming{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%DynamicProgramming{data: data}), do: data
  
  def validate(%DynamicProgramming{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = DynamicProgramming.new()
updated = DynamicProgramming.process(instance, "example")
IO.puts("Data: " <> DynamicProgramming.get_data(updated))
IO.puts("Valid: " <> to_string(DynamicProgramming.validate(updated)))
