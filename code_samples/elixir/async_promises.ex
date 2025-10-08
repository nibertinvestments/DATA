# Async: Promises
# AI/ML Training Sample

defmodule Promises do
  defstruct data: ""
  
  def new(), do: %Promises{}
  
  def process(%Promises{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Promises{data: data}), do: data
  
  def validate(%Promises{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Promises.new()
updated = Promises.process(instance, "example")
IO.puts("Data: " <> Promises.get_data(updated))
IO.puts("Valid: " <> to_string(Promises.validate(updated)))
