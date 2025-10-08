using System;

/// <summary>
/// Data Structures: Tree
/// AI/ML Training Sample
/// </summary>
public class Tree
{
    private string data;
    
    public Tree()
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
        var instance = new Tree();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
