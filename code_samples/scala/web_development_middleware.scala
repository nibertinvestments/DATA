/**
 * Web Development: Middleware
 * AI/ML Training Sample
 */

class Middleware {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object MiddlewareExample {
  def main(args: Array[String]): Unit = {
    val instance = new Middleware()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
