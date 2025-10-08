# Data Structures: Heap
# AI/ML Training Sample

defmodule Heap do
  defstruct data: ""
  
  def new(), do: %Heap{}
  
  def process(%Heap{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Heap{data: data}), do: data
  
  def validate(%Heap{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Heap.new()
updated = Heap.process(instance, "example")
IO.puts("Data: " <> Heap.get_data(updated))
IO.puts("Valid: " <> to_string(Heap.validate(updated)))
