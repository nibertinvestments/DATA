/**
 * Networking: Tcp Udp
 * AI/ML Training Sample
 */
public class NetworkingTcpUdp {
    
    private String data;
    
    public NetworkingTcpUdp() {
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
        NetworkingTcpUdp instance = new NetworkingTcpUdp();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
