# Oop: Polymorphism
# AI/ML Training Sample

defmodule Polymorphism do
  defstruct data: ""
  
  def new(), do: %Polymorphism{}
  
  def process(%Polymorphism{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Polymorphism{data: data}), do: data
  
  def validate(%Polymorphism{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Polymorphism.new()
updated = Polymorphism.process(instance, "example")
IO.puts("Data: " <> Polymorphism.get_data(updated))
IO.puts("Valid: " <> to_string(Polymorphism.validate(updated)))
