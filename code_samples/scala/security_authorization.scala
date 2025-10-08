/**
 * Security: Authorization
 * AI/ML Training Sample
 */

class Authorization {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object AuthorizationExample {
  def main(args: Array[String]): Unit = {
    val instance = new Authorization()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
