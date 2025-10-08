# Performance: Lazy Loading
# AI/ML Training Sample

defmodule LazyLoading do
  defstruct data: ""
  
  def new(), do: %LazyLoading{}
  
  def process(%LazyLoading{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%LazyLoading{data: data}), do: data
  
  def validate(%LazyLoading{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = LazyLoading.new()
updated = LazyLoading.process(instance, "example")
IO.puts("Data: " <> LazyLoading.get_data(updated))
IO.puts("Valid: " <> to_string(LazyLoading.validate(updated)))
