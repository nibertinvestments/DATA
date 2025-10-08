# File Operations: Writing
# AI/ML Training Sample

defmodule Writing do
  defstruct data: ""
  
  def new(), do: %Writing{}
  
  def process(%Writing{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Writing{data: data}), do: data
  
  def validate(%Writing{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Writing.new()
updated = Writing.process(instance, "example")
IO.puts("Data: " <> Writing.get_data(updated))
IO.puts("Valid: " <> to_string(Writing.validate(updated)))
