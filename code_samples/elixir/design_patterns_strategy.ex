# Design Patterns: Strategy
# AI/ML Training Sample

defmodule Strategy do
  defstruct data: ""
  
  def new(), do: %Strategy{}
  
  def process(%Strategy{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Strategy{data: data}), do: data
  
  def validate(%Strategy{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Strategy.new()
updated = Strategy.process(instance, "example")
IO.puts("Data: " <> Strategy.get_data(updated))
IO.puts("Valid: " <> to_string(Strategy.validate(updated)))
