# Web Development: Routing
# AI/ML Training Sample

defmodule Routing do
  defstruct data: ""
  
  def new(), do: %Routing{}
  
  def process(%Routing{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Routing{data: data}), do: data
  
  def validate(%Routing{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Routing.new()
updated = Routing.process(instance, "example")
IO.puts("Data: " <> Routing.get_data(updated))
IO.puts("Valid: " <> to_string(Routing.validate(updated)))
