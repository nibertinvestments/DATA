/**
 * Testing: Unit Tests
 * AI/ML Training Sample
 */
public class TestingUnitTests {
    
    private String data;
    
    public TestingUnitTests() {
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
        TestingUnitTests instance = new TestingUnitTests();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
