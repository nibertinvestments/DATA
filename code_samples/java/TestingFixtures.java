/**
 * Testing: Fixtures
 * AI/ML Training Sample
 */
public class TestingFixtures {
    
    private String data;
    
    public TestingFixtures() {
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
        TestingFixtures instance = new TestingFixtures();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
