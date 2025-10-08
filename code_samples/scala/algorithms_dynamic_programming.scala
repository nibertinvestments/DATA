/**
 * Algorithms: Dynamic Programming
 * AI/ML Training Sample
 */

class DynamicProgramming {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object DynamicProgrammingExample {
  def main(args: Array[String]): Unit = {
    val instance = new DynamicProgramming()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
