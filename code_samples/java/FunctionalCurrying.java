/**
 * Functional: Currying
 * AI/ML Training Sample
 */
public class FunctionalCurrying {
    
    private String data;
    
    public FunctionalCurrying() {
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
        FunctionalCurrying instance = new FunctionalCurrying();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
