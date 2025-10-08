/**
 * Design Patterns: Strategy
 * AI/ML Training Sample
 */
public class DesignPatternsStrategy {
    
    private String data;
    
    public DesignPatternsStrategy() {
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
        DesignPatternsStrategy instance = new DesignPatternsStrategy();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
