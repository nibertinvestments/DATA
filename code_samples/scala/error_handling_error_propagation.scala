/**
 * Error Handling: Error Propagation
 * AI/ML Training Sample
 */

class ErrorPropagation {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ErrorPropagationExample {
  def main(args: Array[String]): Unit = {
    val instance = new ErrorPropagation()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
