# Networking: Socket Programming
# AI/ML Training Sample

defmodule SocketProgramming do
  defstruct data: ""
  
  def new(), do: %SocketProgramming{}
  
  def process(%SocketProgramming{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%SocketProgramming{data: data}), do: data
  
  def validate(%SocketProgramming{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = SocketProgramming.new()
updated = SocketProgramming.process(instance, "example")
IO.puts("Data: " <> SocketProgramming.get_data(updated))
IO.puts("Valid: " <> to_string(SocketProgramming.validate(updated)))
