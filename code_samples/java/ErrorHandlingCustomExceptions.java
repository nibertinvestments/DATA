/**
 * Error Handling: Custom Exceptions
 * AI/ML Training Sample
 */
public class ErrorHandlingCustomExceptions {
    
    private String data;
    
    public ErrorHandlingCustomExceptions() {
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
        ErrorHandlingCustomExceptions instance = new ErrorHandlingCustomExceptions();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
