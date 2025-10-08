/**
 * Design Patterns: Factory
 * AI/ML Training Sample
 */

class Factory {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object FactoryExample {
  def main(args: Array[String]): Unit = {
    val instance = new Factory()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
