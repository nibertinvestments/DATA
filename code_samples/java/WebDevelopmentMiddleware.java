/**
 * Web Development: Middleware
 * AI/ML Training Sample
 */
public class WebDevelopmentMiddleware {
    
    private String data;
    
    public WebDevelopmentMiddleware() {
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
        WebDevelopmentMiddleware instance = new WebDevelopmentMiddleware();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
