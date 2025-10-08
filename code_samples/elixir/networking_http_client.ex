# Networking: Http Client
# AI/ML Training Sample

defmodule HttpClient do
  defstruct data: ""
  
  def new(), do: %HttpClient{}
  
  def process(%HttpClient{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%HttpClient{data: data}), do: data
  
  def validate(%HttpClient{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = HttpClient.new()
updated = HttpClient.process(instance, "example")
IO.puts("Data: " <> HttpClient.get_data(updated))
IO.puts("Valid: " <> to_string(HttpClient.validate(updated)))
