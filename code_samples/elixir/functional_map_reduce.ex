# Functional: Map Reduce
# AI/ML Training Sample

defmodule MapReduce do
  defstruct data: ""
  
  def new(), do: %MapReduce{}
  
  def process(%MapReduce{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%MapReduce{data: data}), do: data
  
  def validate(%MapReduce{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = MapReduce.new()
updated = MapReduce.process(instance, "example")
IO.puts("Data: " <> MapReduce.get_data(updated))
IO.puts("Valid: " <> to_string(MapReduce.validate(updated)))
