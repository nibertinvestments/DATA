/**
 * Security: Input Validation
 * AI/ML Training Sample
 */
public class SecurityInputValidation {
    
    private String data;
    
    public SecurityInputValidation() {
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
        SecurityInputValidation instance = new SecurityInputValidation();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
