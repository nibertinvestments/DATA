/**
 * Async: Threads
 * AI/ML Training Sample
 */
public class AsyncThreads {
    
    private String data;
    
    public AsyncThreads() {
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
        AsyncThreads instance = new AsyncThreads();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
