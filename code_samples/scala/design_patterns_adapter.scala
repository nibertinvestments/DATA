/**
 * Design Patterns: Adapter
 * AI/ML Training Sample
 */

class Adapter {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object AdapterExample {
  def main(args: Array[String]): Unit = {
    val instance = new Adapter()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
