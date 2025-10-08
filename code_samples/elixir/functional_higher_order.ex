# Functional: Higher Order
# AI/ML Training Sample

defmodule HigherOrder do
  defstruct data: ""
  
  def new(), do: %HigherOrder{}
  
  def process(%HigherOrder{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%HigherOrder{data: data}), do: data
  
  def validate(%HigherOrder{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = HigherOrder.new()
updated = HigherOrder.process(instance, "example")
IO.puts("Data: " <> HigherOrder.get_data(updated))
IO.puts("Valid: " <> to_string(HigherOrder.validate(updated)))
