# Functional: Monads
# AI/ML Training Sample

defmodule Monads do
  defstruct data: ""
  
  def new(), do: %Monads{}
  
  def process(%Monads{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Monads{data: data}), do: data
  
  def validate(%Monads{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Monads.new()
updated = Monads.process(instance, "example")
IO.puts("Data: " <> Monads.get_data(updated))
IO.puts("Valid: " <> to_string(Monads.validate(updated)))
