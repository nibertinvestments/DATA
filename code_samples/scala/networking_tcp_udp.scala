/**
 * Networking: Tcp Udp
 * AI/ML Training Sample
 */

class TcpUdp {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object TcpUdpExample {
  def main(args: Array[String]): Unit = {
    val instance = new TcpUdp()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
