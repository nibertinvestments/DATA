# Design Patterns: Decorator
# AI/ML Training Sample

defmodule Decorator do
  defstruct data: ""
  
  def new(), do: %Decorator{}
  
  def process(%Decorator{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Decorator{data: data}), do: data
  
  def validate(%Decorator{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Decorator.new()
updated = Decorator.process(instance, "example")
IO.puts("Data: " <> Decorator.get_data(updated))
IO.puts("Valid: " <> to_string(Decorator.validate(updated)))
