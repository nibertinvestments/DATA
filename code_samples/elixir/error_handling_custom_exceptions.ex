# Error Handling: Custom Exceptions
# AI/ML Training Sample

defmodule CustomExceptions do
  defstruct data: ""
  
  def new(), do: %CustomExceptions{}
  
  def process(%CustomExceptions{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%CustomExceptions{data: data}), do: data
  
  def validate(%CustomExceptions{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = CustomExceptions.new()
updated = CustomExceptions.process(instance, "example")
IO.puts("Data: " <> CustomExceptions.get_data(updated))
IO.puts("Valid: " <> to_string(CustomExceptions.validate(updated)))
