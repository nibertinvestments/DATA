# Data Structures: Trie
# AI/ML Training Sample

defmodule Trie do
  defstruct data: ""
  
  def new(), do: %Trie{}
  
  def process(%Trie{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Trie{data: data}), do: data
  
  def validate(%Trie{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Trie.new()
updated = Trie.process(instance, "example")
IO.puts("Data: " <> Trie.get_data(updated))
IO.puts("Valid: " <> to_string(Trie.validate(updated)))
