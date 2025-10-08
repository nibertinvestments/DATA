/**
 * Functional: Higher Order
 * AI/ML Training Sample
 */

class HigherOrder {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object HigherOrderExample {
  def main(args: Array[String]): Unit = {
    val instance = new HigherOrder()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
