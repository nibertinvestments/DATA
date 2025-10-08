# Database: Orm
# AI/ML Training Sample

defmodule Orm do
  defstruct data: ""
  
  def new(), do: %Orm{}
  
  def process(%Orm{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%Orm{data: data}), do: data
  
  def validate(%Orm{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = Orm.new()
updated = Orm.process(instance, "example")
IO.puts("Data: " <> Orm.get_data(updated))
IO.puts("Valid: " <> to_string(Orm.validate(updated)))
