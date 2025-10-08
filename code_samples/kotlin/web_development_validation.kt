/**
 * Web Development: Validation
 * AI/ML Training Sample
 */

class Validation {
    var data: String = ""
        private set
    
    fun process(input: String) {
        data = input
    }
    
    fun getData(): String = data
    
    fun validate(): Boolean = data.isNotEmpty()
}

fun main() {
    val instance = Validation()
    instance.process("example")
    println("Data: ${instance.getData()}")
    println("Valid: ${instance.validate()}")
}
