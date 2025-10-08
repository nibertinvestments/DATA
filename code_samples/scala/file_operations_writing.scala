/**
 * File Operations: Writing
 * AI/ML Training Sample
 */

class Writing {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object WritingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Writing()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
