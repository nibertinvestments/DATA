# Web Development: Authentication
# AI/ML Training Sample

defmodule Authentication do
  defstruct data: ""
  
  def new(), do: %Authentication{}
  
  def process(%Authentication{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Authentication{data: data}), do: data
  
  def validate(%Authentication{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Authentication.new()
updated = Authentication.process(instance, "example")
IO.puts("Data: " <> Authentication.get_data(updated))
IO.puts("Valid: " <> to_string(Authentication.validate(updated)))
