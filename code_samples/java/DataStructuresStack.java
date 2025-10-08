/**
 * Data Structures: Stack
 * AI/ML Training Sample
 */
public class DataStructuresStack {
    
    private String data;
    
    public DataStructuresStack() {
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
        DataStructuresStack instance = new DataStructuresStack();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
