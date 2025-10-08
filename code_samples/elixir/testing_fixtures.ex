# Testing: Fixtures
# AI/ML Training Sample

defmodule Fixtures do
  defstruct data: ""
  
  def new(), do: %Fixtures{}
  
  def process(%Fixtures{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Fixtures{data: data}), do: data
  
  def validate(%Fixtures{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Fixtures.new()
updated = Fixtures.process(instance, "example")
IO.puts("Data: " <> Fixtures.get_data(updated))
IO.puts("Valid: " <> to_string(Fixtures.validate(updated)))
