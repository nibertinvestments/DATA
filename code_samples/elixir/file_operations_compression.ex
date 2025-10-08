# File Operations: Compression
# AI/ML Training Sample

defmodule Compression do
  defstruct data: ""
  
  def new(), do: %Compression{}
  
  def process(%Compression{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Compression{data: data}), do: data
  
  def validate(%Compression{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Compression.new()
updated = Compression.process(instance, "example")
IO.puts("Data: " <> Compression.get_data(updated))
IO.puts("Valid: " <> to_string(Compression.validate(updated)))
