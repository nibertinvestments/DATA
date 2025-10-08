# Async: Coroutines
# AI/ML Training Sample

defmodule Coroutines do
  defstruct data: ""
  
  def new(), do: %Coroutines{}
  
  def process(%Coroutines{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Coroutines{data: data}), do: data
  
  def validate(%Coroutines{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Coroutines.new()
updated = Coroutines.process(instance, "example")
IO.puts("Data: " <> Coroutines.get_data(updated))
IO.puts("Valid: " <> to_string(Coroutines.validate(updated)))
