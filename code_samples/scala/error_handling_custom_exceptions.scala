/**
 * Error Handling: Custom Exceptions
 * AI/ML Training Sample
 */

class CustomExceptions {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object CustomExceptionsExample {
  def main(args: Array[String]): Unit = {
    val instance = new CustomExceptions()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
