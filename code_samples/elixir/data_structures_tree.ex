# Data Structures: Tree
# AI/ML Training Sample

defmodule Tree do
  defstruct data: ""
  
  def new(), do: %Tree{}
  
  def process(%Tree{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Tree{data: data}), do: data
  
  def validate(%Tree{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Tree.new()
updated = Tree.process(instance, "example")
IO.puts("Data: " <> Tree.get_data(updated))
IO.puts("Valid: " <> to_string(Tree.validate(updated)))
