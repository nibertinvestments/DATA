/**
 * Web Development: Routing
 * AI/ML Training Sample
 */

class Routing {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object RoutingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Routing()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
