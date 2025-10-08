/**
 * File Operations: Streaming
 * AI/ML Training Sample
 */

class Streaming {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object StreamingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Streaming()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
