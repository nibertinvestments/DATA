/**
 * Algorithms: Searching
 * AI/ML Training Sample
 */

class Searching {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object SearchingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Searching()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
