using System;

/// <summary>
/// Algorithms: Dynamic Programming
/// AI/ML Training Sample
/// </summary>
public class DynamicProgramming
{
    private string data;
    
    public DynamicProgramming()
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
        var instance = new DynamicProgramming();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
