# Security: Hashing
# AI/ML Training Sample

defmodule Hashing do
  defstruct data: ""
  
  def new(), do: %Hashing{}
  
  def process(%Hashing{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Hashing{data: data}), do: data
  
  def validate(%Hashing{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Hashing.new()
updated = Hashing.process(instance, "example")
IO.puts("Data: " <> Hashing.get_data(updated))
IO.puts("Valid: " <> to_string(Hashing.validate(updated)))
