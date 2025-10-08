# Networking: Tcp Udp
# AI/ML Training Sample

defmodule TcpUdp do
  defstruct data: ""
  
  def new(), do: %TcpUdp{}
  
  def process(%TcpUdp{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%TcpUdp{data: data}), do: data
  
  def validate(%TcpUdp{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = TcpUdp.new()
updated = TcpUdp.process(instance, "example")
IO.puts("Data: " <> TcpUdp.get_data(updated))
IO.puts("Valid: " <> to_string(TcpUdp.validate(updated)))
