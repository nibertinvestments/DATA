using System;

/// <summary>
/// Error Handling: Error Propagation
/// AI/ML Training Sample
/// </summary>
public class ErrorPropagation
{
    private string data;
    
    public ErrorPropagation()
    {
        this.data = string.Empty;
    }
    
    public void Process(string input)
    {
        this.data = input;
    }
    
    public string GetData()
    {
        return this.data;
    }
    
    public bool Validate()
    {
        return !string.IsNullOrEmpty(this.data);
    }
    
    public static void Main(string[] args)
    {
        var instance = new ErrorPropagation();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
