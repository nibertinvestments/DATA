/**
 * Networking: Protocols
 * AI/ML Training Sample
 */

class Protocols {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ProtocolsExample {
  def main(args: Array[String]): Unit = {
    val instance = new Protocols()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
