/**
 * Web Development: Validation
 * AI/ML Training Sample
 */
public class WebDevelopmentValidation {
    
    private String data;
    
    public WebDevelopmentValidation() {
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
        WebDevelopmentValidation instance = new WebDevelopmentValidation();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
