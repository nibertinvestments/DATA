/**
 * Performance: Batching
 * AI/ML Training Sample
 */

class Batching {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object BatchingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Batching()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
