# Oop: Inheritance
# AI/ML Training Sample

defmodule Inheritance do
  defstruct data: ""
  
  def new(), do: %Inheritance{}
  
  def process(%Inheritance{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Inheritance{data: data}), do: data
  
  def validate(%Inheritance{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Inheritance.new()
updated = Inheritance.process(instance, "example")
IO.puts("Data: " <> Inheritance.get_data(updated))
IO.puts("Valid: " <> to_string(Inheritance.validate(updated)))
