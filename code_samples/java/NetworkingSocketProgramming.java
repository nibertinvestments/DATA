/**
 * Networking: Socket Programming
 * AI/ML Training Sample
 */
public class NetworkingSocketProgramming {
    
    private String data;
    
    public NetworkingSocketProgramming() {
        this.data = "";
    }
    
    public void process(String input) {
        this.data = input;
    }
    
    public String getData() {
        return this.data;
    }
    
    public boolean validate() {
        return this.data != null && !this.data.isEmpty();
    }
    
    public static void main(String[] args) {
        NetworkingSocketProgramming instance = new NetworkingSocketProgramming();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
