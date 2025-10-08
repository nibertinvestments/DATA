# Error Handling: Error Propagation
# AI/ML Training Sample

defmodule ErrorPropagation do
  defstruct data: ""
  
  def new(), do: %ErrorPropagation{}
  
  def process(%ErrorPropagation{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%ErrorPropagation{data: data}), do: data
  
  def validate(%ErrorPropagation{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = ErrorPropagation.new()
updated = ErrorPropagation.process(instance, "example")
IO.puts("Data: " <> ErrorPropagation.get_data(updated))
IO.puts("Valid: " <> to_string(ErrorPropagation.validate(updated)))
