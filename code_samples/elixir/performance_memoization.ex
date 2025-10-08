# Performance: Memoization
# AI/ML Training Sample

defmodule Memoization do
  defstruct data: ""
  
  def new(), do: %Memoization{}
  
  def process(%Memoization{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Memoization{data: data}), do: data
  
  def validate(%Memoization{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Memoization.new()
updated = Memoization.process(instance, "example")
IO.puts("Data: " <> Memoization.get_data(updated))
IO.puts("Valid: " <> to_string(Memoization.validate(updated)))
