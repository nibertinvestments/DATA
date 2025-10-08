# Data Structures: Linked List
# AI/ML Training Sample

defmodule LinkedList do
  defstruct data: ""
  
  def new(), do: %LinkedList{}
  
  def process(%LinkedList{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%LinkedList{data: data}), do: data
  
  def validate(%LinkedList{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = LinkedList.new()
updated = LinkedList.process(instance, "example")
IO.puts("Data: " <> LinkedList.get_data(updated))
IO.puts("Valid: " <> to_string(LinkedList.validate(updated)))
