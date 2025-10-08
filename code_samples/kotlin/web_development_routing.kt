/**
 * Web Development: Routing
 * AI/ML Training Sample
 */

class Routing {
    var data: String = ""
        private set
    
    fun process(input: String) {
        data = input
    }
    
    fun getData(): String = data
    
    fun validate(): Boolean = data.isNotEmpty()
}

fun main() {
    val instance = Routing()
    instance.process("example")
    println("Data: ${instance.getData()}")
    println("Valid: ${instance.validate()}")
}
