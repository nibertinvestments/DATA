# Design Patterns: Singleton
# AI/ML Training Sample

defmodule Singleton do
  defstruct data: ""
  
  def new(), do: %Singleton{}
  
  def process(%Singleton{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Singleton{data: data}), do: data
  
  def validate(%Singleton{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Singleton.new()
updated = Singleton.process(instance, "example")
IO.puts("Data: " <> Singleton.get_data(updated))
IO.puts("Valid: " <> to_string(Singleton.validate(updated)))
