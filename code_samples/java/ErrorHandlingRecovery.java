/**
 * Error Handling: Recovery
 * AI/ML Training Sample
 */
public class ErrorHandlingRecovery {
    
    private String data;
    
    public ErrorHandlingRecovery() {
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
        ErrorHandlingRecovery instance = new ErrorHandlingRecovery();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
