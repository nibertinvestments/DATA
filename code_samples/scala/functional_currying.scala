/**
 * Functional: Currying
 * AI/ML Training Sample
 */

class Currying {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object CurryingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Currying()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
