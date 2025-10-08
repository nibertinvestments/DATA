/**
 * Data Structures: Tree
 * AI/ML Training Sample
 */
public class DataStructuresTree {
    
    private String data;
    
    public DataStructuresTree() {
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
        DataStructuresTree instance = new DataStructuresTree();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
