# Algorithms: String Algorithms
# AI/ML Training Sample

defmodule StringAlgorithms do
  defstruct data: ""
  
  def new(), do: %StringAlgorithms{}
  
  def process(%StringAlgorithms{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%StringAlgorithms{data: data}), do: data
  
  def validate(%StringAlgorithms{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = StringAlgorithms.new()
updated = StringAlgorithms.process(instance, "example")
IO.puts("Data: " <> StringAlgorithms.get_data(updated))
IO.puts("Valid: " <> to_string(StringAlgorithms.validate(updated)))
