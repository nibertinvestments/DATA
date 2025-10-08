/**
 * Functional: Closures
 * AI/ML Training Sample
 */

class Closures {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ClosuresExample {
  def main(args: Array[String]): Unit = {
    val instance = new Closures()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
