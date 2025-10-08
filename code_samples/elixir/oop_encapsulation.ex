# Oop: Encapsulation
# AI/ML Training Sample

defmodule Encapsulation do
  defstruct data: ""
  
  def new(), do: %Encapsulation{}
  
  def process(%Encapsulation{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Encapsulation{data: data}), do: data
  
  def validate(%Encapsulation{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Encapsulation.new()
updated = Encapsulation.process(instance, "example")
IO.puts("Data: " <> Encapsulation.get_data(updated))
IO.puts("Valid: " <> to_string(Encapsulation.validate(updated)))
