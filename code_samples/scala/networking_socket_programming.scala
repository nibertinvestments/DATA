/**
 * Networking: Socket Programming
 * AI/ML Training Sample
 */

class SocketProgramming {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object SocketProgrammingExample {
  def main(args: Array[String]): Unit = {
    val instance = new SocketProgramming()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
