/**
 * Design Patterns: Adapter
 * AI/ML Training Sample
 */

class Adapter {
    var data: String = ""
        private set
    
    fun process(input: String) {
        data = input
    }
    
    fun getData(): String = data
    
    fun validate(): Boolean = data.isNotEmpty()
}

fun main() {
    val instance = Adapter()
    instance.process("example")
    println("Data: ${instance.getData()}")
    println("Valid: ${instance.validate()}")
}
