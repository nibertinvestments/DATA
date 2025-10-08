# Performance: Batching
# AI/ML Training Sample

defmodule Batching do
  defstruct data: ""
  
  def new(), do: %Batching{}
  
  def process(%Batching{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Batching{data: data}), do: data
  
  def validate(%Batching{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Batching.new()
updated = Batching.process(instance, "example")
IO.puts("Data: " <> Batching.get_data(updated))
IO.puts("Valid: " <> to_string(Batching.validate(updated)))
