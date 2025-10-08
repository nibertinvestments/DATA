/**
 * Design Patterns: Observer
 * AI/ML Training Sample
 */
public class DesignPatternsObserver {
    
    private String data;
    
    public DesignPatternsObserver() {
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
        DesignPatternsObserver instance = new DesignPatternsObserver();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
