/**
 * Performance: Lazy Loading
 * AI/ML Training Sample
 */
public class PerformanceLazyLoading {
    
    private String data;
    
    public PerformanceLazyLoading() {
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
        PerformanceLazyLoading instance = new PerformanceLazyLoading();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
