/**
 * Design Patterns: Decorator
 * AI/ML Training Sample
 */

class Decorator {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object DecoratorExample {
  def main(args: Array[String]): Unit = {
    val instance = new Decorator()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
