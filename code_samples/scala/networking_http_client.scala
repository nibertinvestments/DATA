/**
 * Networking: Http Client
 * AI/ML Training Sample
 */

class HttpClient {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object HttpClientExample {
  def main(args: Array[String]): Unit = {
    val instance = new HttpClient()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
