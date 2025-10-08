# Algorithms: Searching
# AI/ML Training Sample

defmodule Searching do
  defstruct data: ""
  
  def new(), do: %Searching{}
  
  def process(%Searching{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Searching{data: data}), do: data
  
  def validate(%Searching{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Searching.new()
updated = Searching.process(instance, "example")
IO.puts("Data: " <> Searching.get_data(updated))
IO.puts("Valid: " <> to_string(Searching.validate(updated)))
