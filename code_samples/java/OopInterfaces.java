/**
 * Oop: Interfaces
 * AI/ML Training Sample
 */
public class OopInterfaces {
    
    private String data;
    
    public OopInterfaces() {
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
        OopInterfaces instance = new OopInterfaces();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
