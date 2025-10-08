using System;

/// <summary>
/// File Operations: Writing
/// AI/ML Training Sample
/// </summary>
public class Writing
{
    private string data;
    
    public Writing()
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
        var instance = new Writing();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
