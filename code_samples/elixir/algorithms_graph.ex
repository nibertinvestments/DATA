# Algorithms: Graph
# AI/ML Training Sample

defmodule Graph do
  defstruct data: ""
  
  def new(), do: %Graph{}
  
  def process(%Graph{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Graph{data: data}), do: data
  
  def validate(%Graph{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Graph.new()
updated = Graph.process(instance, "example")
IO.puts("Data: " <> Graph.get_data(updated))
IO.puts("Valid: " <> to_string(Graph.validate(updated)))
