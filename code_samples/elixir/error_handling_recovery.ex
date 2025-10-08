# Error Handling: Recovery
# AI/ML Training Sample

defmodule Recovery do
  defstruct data: ""
  
  def new(), do: %Recovery{}
  
  def process(%Recovery{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Recovery{data: data}), do: data
  
  def validate(%Recovery{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Recovery.new()
updated = Recovery.process(instance, "example")
IO.puts("Data: " <> Recovery.get_data(updated))
IO.puts("Valid: " <> to_string(Recovery.validate(updated)))
