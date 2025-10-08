/**
 * Algorithms: String Algorithms
 * AI/ML Training Sample
 */

class StringAlgorithms {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object StringAlgorithmsExample {
  def main(args: Array[String]): Unit = {
    val instance = new StringAlgorithms()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
