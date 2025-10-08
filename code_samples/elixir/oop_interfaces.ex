# Oop: Interfaces
# AI/ML Training Sample

defmodule Interfaces do
  defstruct data: ""
  
  def new(), do: %Interfaces{}
  
  def process(%Interfaces{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Interfaces{data: data}), do: data
  
  def validate(%Interfaces{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Interfaces.new()
updated = Interfaces.process(instance, "example")
IO.puts("Data: " <> Interfaces.get_data(updated))
IO.puts("Valid: " <> to_string(Interfaces.validate(updated)))
