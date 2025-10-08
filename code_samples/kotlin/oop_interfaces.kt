/**
 * Oop: Interfaces
 * AI/ML Training Sample
 */

class Interfaces {
    var data: String = ""
        private set
    
    fun process(input: String) {
        data = input
    }
    
    fun getData(): String = data
    
    fun validate(): Boolean = data.isNotEmpty()
}

fun main() {
    val instance = Interfaces()
    instance.process("example")
    println("Data: ${instance.getData()}")
    println("Valid: ${instance.validate()}")
}
