# Error Handling: Logging
# AI/ML Training Sample

defmodule Logging do
  defstruct data: ""
  
  def new(), do: %Logging{}
  
  def process(%Logging{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Logging{data: data}), do: data
  
  def validate(%Logging{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Logging.new()
updated = Logging.process(instance, "example")
IO.puts("Data: " <> Logging.get_data(updated))
IO.puts("Valid: " <> to_string(Logging.validate(updated)))
