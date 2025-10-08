# File Operations: Reading
# AI/ML Training Sample

defmodule Reading do
  defstruct data: ""
  
  def new(), do: %Reading{}
  
  def process(%Reading{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Reading{data: data}), do: data
  
  def validate(%Reading{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Reading.new()
updated = Reading.process(instance, "example")
IO.puts("Data: " <> Reading.get_data(updated))
IO.puts("Valid: " <> to_string(Reading.validate(updated)))
