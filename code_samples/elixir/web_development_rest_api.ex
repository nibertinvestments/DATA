# Web Development: Rest Api
# AI/ML Training Sample

defmodule RestApi do
  defstruct data: ""
  
  def new(), do: %RestApi{}
  
  def process(%RestApi{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%RestApi{data: data}), do: data
  
  def validate(%RestApi{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = RestApi.new()
updated = RestApi.process(instance, "example")
IO.puts("Data: " <> RestApi.get_data(updated))
IO.puts("Valid: " <> to_string(RestApi.validate(updated)))
