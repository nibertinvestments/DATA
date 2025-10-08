# Data Structures: Hash Table
# AI/ML Training Sample

defmodule HashTable do
  defstruct data: ""
  
  def new(), do: %HashTable{}
  
  def process(%HashTable{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%HashTable{data: data}), do: data
  
  def validate(%HashTable{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = HashTable.new()
updated = HashTable.process(instance, "example")
IO.puts("Data: " <> HashTable.get_data(updated))
IO.puts("Valid: " <> to_string(HashTable.validate(updated)))
