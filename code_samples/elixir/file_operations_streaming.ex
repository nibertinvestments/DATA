# File Operations: Streaming
# AI/ML Training Sample

defmodule Streaming do
  defstruct data: ""
  
  def new(), do: %Streaming{}
  
  def process(%Streaming{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Streaming{data: data}), do: data
  
  def validate(%Streaming{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Streaming.new()
updated = Streaming.process(instance, "example")
IO.puts("Data: " <> Streaming.get_data(updated))
IO.puts("Valid: " <> to_string(Streaming.validate(updated)))
