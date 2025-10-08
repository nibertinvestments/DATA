# Networking: Protocols
# AI/ML Training Sample

defmodule Protocols do
  defstruct data: ""
  
  def new(), do: %Protocols{}
  
  def process(%Protocols{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Protocols{data: data}), do: data
  
  def validate(%Protocols{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Protocols.new()
updated = Protocols.process(instance, "example")
IO.puts("Data: " <> Protocols.get_data(updated))
IO.puts("Valid: " <> to_string(Protocols.validate(updated)))
