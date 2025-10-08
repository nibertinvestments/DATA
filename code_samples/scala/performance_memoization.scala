/**
 * Performance: Memoization
 * AI/ML Training Sample
 */

class Memoization {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object MemoizationExample {
  def main(args: Array[String]): Unit = {
    val instance = new Memoization()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
