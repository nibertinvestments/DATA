/**
 * Security: Input Validation
 * AI/ML Training Sample
 */

class InputValidation {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object InputValidationExample {
  def main(args: Array[String]): Unit = {
    val instance = new InputValidation()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
