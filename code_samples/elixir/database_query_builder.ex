# Database: Query Builder
# AI/ML Training Sample

defmodule QueryBuilder do
  defstruct data: ""
  
  def new(), do: %QueryBuilder{}
  
  def process(%QueryBuilder{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%QueryBuilder{data: data}), do: data
  
  def validate(%QueryBuilder{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = QueryBuilder.new()
updated = QueryBuilder.process(instance, "example")
IO.puts("Data: " <> QueryBuilder.get_data(updated))
IO.puts("Valid: " <> to_string(QueryBuilder.validate(updated)))
