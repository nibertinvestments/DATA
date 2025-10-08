/**
 * Web Development: Rest Api
 * AI/ML Training Sample
 */
public class WebDevelopmentRestApi {
    
    private String data;
    
    public WebDevelopmentRestApi() {
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
        WebDevelopmentRestApi instance = new WebDevelopmentRestApi();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
