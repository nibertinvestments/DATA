# Utilities: Regex
# AI/ML Training Sample

defmodule Regex do
  defstruct data: ""
  
  def new(), do: %Regex{}
  
  def process(%Regex{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Regex{data: data}), do: data
  
  def validate(%Regex{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Regex.new()
updated = Regex.process(instance, "example")
IO.puts("Data: " <> Regex.get_data(updated))
IO.puts("Valid: " <> to_string(Regex.validate(updated)))
