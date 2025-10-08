/**
 * Database: Query Builder
 * AI/ML Training Sample
 */
public class DatabaseQueryBuilder {
    
    private String data;
    
    public DatabaseQueryBuilder() {
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
        DatabaseQueryBuilder instance = new DatabaseQueryBuilder();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
