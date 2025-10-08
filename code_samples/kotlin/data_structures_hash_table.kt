/**
 * Data Structures: Hash Table
 * AI/ML Training Sample
 */

class HashTable {
    var data: String = ""
        private set
    
    fun process(input: String) {
        data = input
    }
    
    fun getData(): String = data
    
    fun validate(): Boolean = data.isNotEmpty()
}

fun main() {
    val instance = HashTable()
    instance.process("example")
    println("Data: ${instance.getData()}")
    println("Valid: ${instance.validate()}")
}
