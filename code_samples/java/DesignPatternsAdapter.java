/**
 * Design Patterns: Adapter
 * AI/ML Training Sample
 */
public class DesignPatternsAdapter {
    
    private String data;
    
    public DesignPatternsAdapter() {
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
        DesignPatternsAdapter instance = new DesignPatternsAdapter();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
