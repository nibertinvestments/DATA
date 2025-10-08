# Async: Channels
# AI/ML Training Sample

defmodule Channels do
  defstruct data: ""
  
  def new(), do: %Channels{}
  
  def process(%Channels{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Channels{data: data}), do: data
  
  def validate(%Channels{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Channels.new()
updated = Channels.process(instance, "example")
IO.puts("Data: " <> Channels.get_data(updated))
IO.puts("Valid: " <> to_string(Channels.validate(updated)))
