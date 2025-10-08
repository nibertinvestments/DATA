/**
 * Algorithms: Sorting
 * AI/ML Training Sample
 */

class Sorting {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object SortingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Sorting()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
