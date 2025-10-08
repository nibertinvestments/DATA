# Security: Input Validation
# AI/ML Training Sample

defmodule InputValidation do
  defstruct data: ""
  
  def new(), do: %InputValidation{}
  
  def process(%InputValidation{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%InputValidation{data: data}), do: data
  
  def validate(%InputValidation{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = InputValidation.new()
updated = InputValidation.process(instance, "example")
IO.puts("Data: " <> InputValidation.get_data(updated))
IO.puts("Valid: " <> to_string(InputValidation.validate(updated)))
