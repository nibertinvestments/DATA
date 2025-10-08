# Utilities: String Manipulation
# AI/ML Training Sample

defmodule StringManipulation do
  defstruct data: ""
  
  def new(), do: %StringManipulation{}
  
  def process(%StringManipulation{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%StringManipulation{data: data}), do: data
  
  def validate(%StringManipulation{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = StringManipulation.new()
updated = StringManipulation.process(instance, "example")
IO.puts("Data: " <> StringManipulation.get_data(updated))
IO.puts("Valid: " <> to_string(StringManipulation.validate(updated)))
