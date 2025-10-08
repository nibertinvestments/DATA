/**
 * Design Patterns: Singleton
 * AI/ML Training Sample
 */
public class DesignPatternsSingleton {
    
    private String data;
    
    public DesignPatternsSingleton() {
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
        DesignPatternsSingleton instance = new DesignPatternsSingleton();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
