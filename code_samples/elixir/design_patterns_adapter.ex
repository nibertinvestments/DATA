# Design Patterns: Adapter
# AI/ML Training Sample

defmodule Adapter do
  defstruct data: ""
  
  def new(), do: %Adapter{}
  
  def process(%Adapter{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Adapter{data: data}), do: data
  
  def validate(%Adapter{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Adapter.new()
updated = Adapter.process(instance, "example")
IO.puts("Data: " <> Adapter.get_data(updated))
IO.puts("Valid: " <> to_string(Adapter.validate(updated)))
