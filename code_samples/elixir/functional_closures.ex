# Functional: Closures
# AI/ML Training Sample

defmodule Closures do
  defstruct data: ""
  
  def new(), do: %Closures{}
  
  def process(%Closures{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Closures{data: data}), do: data
  
  def validate(%Closures{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Closures.new()
updated = Closures.process(instance, "example")
IO.puts("Data: " <> Closures.get_data(updated))
IO.puts("Valid: " <> to_string(Closures.validate(updated)))
