# Design Patterns: Observer
# AI/ML Training Sample

defmodule Observer do
  defstruct data: ""
  
  def new(), do: %Observer{}
  
  def process(%Observer{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Observer{data: data}), do: data
  
  def validate(%Observer{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Observer.new()
updated = Observer.process(instance, "example")
IO.puts("Data: " <> Observer.get_data(updated))
IO.puts("Valid: " <> to_string(Observer.validate(updated)))
