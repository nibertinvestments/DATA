/**
 * Data Structures: Trie
 * AI/ML Training Sample
 */
public class DataStructuresTrie {
    
    private String data;
    
    public DataStructuresTrie() {
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
        DataStructuresTrie instance = new DataStructuresTrie();
        instance.process("example");
        System.out.println("Data: " + instance.getData());
        System.out.println("Valid: " + instance.validate());
    }
}
