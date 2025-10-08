using System;

/// <summary>
/// Networking: Tcp Udp
/// AI/ML Training Sample
/// </summary>
public class TcpUdp
{
    private string data;
    
    public TcpUdp()
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
        var instance = new TcpUdp();
        instance.Process("example");
        Console.WriteLine($"Data: {instance.GetData()}");
        Console.WriteLine($"Valid: {instance.Validate()}");
    }
}
