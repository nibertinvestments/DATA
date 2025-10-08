# Web Development: Validation
# AI/ML Training Sample

defmodule Validation do
  defstruct data: ""
  
  def new(), do: %Validation{}
  
  def process(%Validation{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Validation{data: data}), do: data
  
  def validate(%Validation{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Validation.new()
updated = Validation.process(instance, "example")
IO.puts("Data: " <> Validation.get_data(updated))
IO.puts("Valid: " <> to_string(Validation.validate(updated)))
