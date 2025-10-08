/**
 * Functional: Closures
 * AI/ML Training Sample
 */
public class FunctionalClosures {
    
    private String data;
    
    public FunctionalClosures() {
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
        FunctionalClosures instance = new FunctionalClosures();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
