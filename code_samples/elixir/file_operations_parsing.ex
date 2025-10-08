# File Operations: Parsing
# AI/ML Training Sample

defmodule Parsing do
  defstruct data: ""
  
  def new(), do: %Parsing{}
  
  def process(%Parsing{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Parsing{data: data}), do: data
  
  def validate(%Parsing{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Parsing.new()
updated = Parsing.process(instance, "example")
IO.puts("Data: " <> Parsing.get_data(updated))
IO.puts("Valid: " <> to_string(Parsing.validate(updated)))
