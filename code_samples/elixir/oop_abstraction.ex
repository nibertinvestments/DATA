# Oop: Abstraction
# AI/ML Training Sample

defmodule Abstraction do
  defstruct data: ""
  
  def new(), do: %Abstraction{}
  
  def process(%Abstraction{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Abstraction{data: data}), do: data
  
  def validate(%Abstraction{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Abstraction.new()
updated = Abstraction.process(instance, "example")
IO.puts("Data: " <> Abstraction.get_data(updated))
IO.puts("Valid: " <> to_string(Abstraction.validate(updated)))
