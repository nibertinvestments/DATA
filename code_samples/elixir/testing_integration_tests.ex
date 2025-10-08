# Testing: Integration Tests
# AI/ML Training Sample

defmodule IntegrationTests do
  defstruct data: ""
  
  def new(), do: %IntegrationTests{}
  
  def process(%IntegrationTests{} = struct, input) do
    %{struct | data: input}
  end
  
  def get_data(%IntegrationTests{data: data}), do: data
  
  def validate(%IntegrationTests{data: data}) do
    String.length(data) > 0
  end
end

# Example usage
instance = IntegrationTests.new()
updated = IntegrationTests.process(instance, "example")
IO.puts("Data: " <> IntegrationTests.get_data(updated))
IO.puts("Valid: " <> to_string(IntegrationTests.validate(updated)))
