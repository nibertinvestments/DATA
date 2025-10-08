/**
 * Security: Authentication
 * AI/ML Training Sample
 */

class Authentication {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object AuthenticationExample {
  def main(args: Array[String]): Unit = {
    val instance = new Authentication()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
