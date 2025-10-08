# Testing: Mocking
# AI/ML Training Sample

defmodule Mocking do
  defstruct data: ""
  
  def new(), do: %Mocking{}
  
  def process(%Mocking{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Mocking{data: data}), do: data
  
  def validate(%Mocking{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Mocking.new()
updated = Mocking.process(instance, "example")
IO.puts("Data: " <> Mocking.get_data(updated))
IO.puts("Valid: " <> to_string(Mocking.validate(updated)))
