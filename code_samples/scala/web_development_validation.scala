/**
 * Web Development: Validation
 * AI/ML Training Sample
 */

class Validation {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ValidationExample {
  def main(args: Array[String]): Unit = {
    val instance = new Validation()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
