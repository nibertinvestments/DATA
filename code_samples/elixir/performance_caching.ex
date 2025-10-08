# Performance: Caching
# AI/ML Training Sample

defmodule Caching do
  defstruct data: ""
  
  def new(), do: %Caching{}
  
  def process(%Caching{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Caching{data: data}), do: data
  
  def validate(%Caching{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Caching.new()
updated = Caching.process(instance, "example")
IO.puts("Data: " <> Caching.get_data(updated))
IO.puts("Valid: " <> to_string(Caching.validate(updated)))
