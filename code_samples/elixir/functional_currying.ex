# Functional: Currying
# AI/ML Training Sample

defmodule Currying do
  defstruct data: ""
  
  def new(), do: %Currying{}
  
  def process(%Currying{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Currying{data: data}), do: data
  
  def validate(%Currying{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Currying.new()
updated = Currying.process(instance, "example")
IO.puts("Data: " <> Currying.get_data(updated))
IO.puts("Valid: " <> to_string(Currying.validate(updated)))
