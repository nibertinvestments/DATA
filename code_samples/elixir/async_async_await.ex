# Async: Async Await
# AI/ML Training Sample

defmodule AsyncAwait do
  defstruct data: ""
  
  def new(), do: %AsyncAwait{}
  
  def process(%AsyncAwait{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%AsyncAwait{data: data}), do: data
  
  def validate(%AsyncAwait{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = AsyncAwait.new()
updated = AsyncAwait.process(instance, "example")
IO.puts("Data: " <> AsyncAwait.get_data(updated))
IO.puts("Valid: " <> to_string(AsyncAwait.validate(updated)))
