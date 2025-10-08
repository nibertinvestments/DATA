/**
 * Functional: Higher Order
 * AI/ML Training Sample
 */
public class FunctionalHigherOrder {
    
    private String data;
    
    public FunctionalHigherOrder() {
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
        FunctionalHigherOrder instance = new FunctionalHigherOrder();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
