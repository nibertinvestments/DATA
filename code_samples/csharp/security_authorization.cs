using System;

/// <summary>
/// Security: Authorization
/// AI/ML Training Sample
/// </summary>
public class Authorization
{
    private string data;
    
    public Authorization()
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
        var instance = new Authorization();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
