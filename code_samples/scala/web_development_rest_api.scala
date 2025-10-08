/**
 * Web Development: Rest Api
 * AI/ML Training Sample
 */

class RestApi {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object RestApiExample {
  def main(args: Array[String]): Unit = {
    val instance = new RestApi()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
