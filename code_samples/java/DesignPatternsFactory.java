/**
 * Design Patterns: Factory
 * AI/ML Training Sample
 */
public class DesignPatternsFactory {
    
    private String data;
    
    public DesignPatternsFactory() {
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
        DesignPatternsFactory instance = new DesignPatternsFactory();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
