# Async: Threads
# AI/ML Training Sample

defmodule Threads do
  defstruct data: ""
  
  def new(), do: %Threads{}
  
  def process(%Threads{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Threads{data: data}), do: data
  
  def validate(%Threads{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Threads.new()
updated = Threads.process(instance, "example")
IO.puts("Data: " <> Threads.get_data(updated))
IO.puts("Valid: " <> to_string(Threads.validate(updated)))
