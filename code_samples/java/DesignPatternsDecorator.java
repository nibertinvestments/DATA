/**
 * Design Patterns: Decorator
 * AI/ML Training Sample
 */
public class DesignPatternsDecorator {
    
    private String data;
    
    public DesignPatternsDecorator() {
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
        DesignPatternsDecorator instance = new DesignPatternsDecorator();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
