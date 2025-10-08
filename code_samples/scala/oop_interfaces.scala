/**
 * Oop: Interfaces
 * AI/ML Training Sample
 */

class Interfaces {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object InterfacesExample {
  def main(args: Array[String]): Unit = {
    val instance = new Interfaces()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
