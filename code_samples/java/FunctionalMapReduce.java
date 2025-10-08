/**
 * Functional: Map Reduce
 * AI/ML Training Sample
 */
public class FunctionalMapReduce {
    
    private String data;
    
    public FunctionalMapReduce() {
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
        FunctionalMapReduce instance = new FunctionalMapReduce();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
