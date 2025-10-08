# Error Handling: Try Catch
# AI/ML Training Sample

defmodule TryCatch do
  defstruct data: ""
  
  def new(), do: %TryCatch{}
  
  def process(%TryCatch{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%TryCatch{data: data}), do: data
  
  def validate(%TryCatch{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = TryCatch.new()
updated = TryCatch.process(instance, "example")
IO.puts("Data: " <> TryCatch.get_data(updated))
IO.puts("Valid: " <> to_string(TryCatch.validate(updated)))
