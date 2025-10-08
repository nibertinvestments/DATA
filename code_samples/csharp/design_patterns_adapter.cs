using System;

/// <summary>
/// Design Patterns: Adapter
/// AI/ML Training Sample
/// </summary>
public class Adapter
{
    private string data;
    
    public Adapter()
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
        var instance = new Adapter();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
