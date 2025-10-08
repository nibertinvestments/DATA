/**
 * Networking: Websockets
 * AI/ML Training Sample
 */

class Websockets {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object WebsocketsExample {
  def main(args: Array[String]): Unit = {
    val instance = new Websockets()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
