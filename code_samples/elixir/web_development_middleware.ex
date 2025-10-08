# Web Development: Middleware
# AI/ML Training Sample

defmodule Middleware do
  defstruct data: ""
  
  def new(), do: %Middleware{}
  
  def process(%Middleware{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Middleware{data: data}), do: data
  
  def validate(%Middleware{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Middleware.new()
updated = Middleware.process(instance, "example")
IO.puts("Data: " <> Middleware.get_data(updated))
IO.puts("Valid: " <> to_string(Middleware.validate(updated)))
