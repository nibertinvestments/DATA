/**
 * Functional: Monads
 * AI/ML Training Sample
 */

class Monads {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object MonadsExample {
  def main(args: Array[String]): Unit = {
    val instance = new Monads()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
