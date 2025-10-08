# Algorithms: Sorting
# AI/ML Training Sample

defmodule Sorting do
  defstruct data: ""
  
  def new(), do: %Sorting{}
  
  def process(%Sorting{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Sorting{data: data}), do: data
  
  def validate(%Sorting{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Sorting.new()
updated = Sorting.process(instance, "example")
IO.puts("Data: " <> Sorting.get_data(updated))
IO.puts("Valid: " <> to_string(Sorting.validate(updated)))
