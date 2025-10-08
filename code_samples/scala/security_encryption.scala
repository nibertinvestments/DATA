/**
 * Security: Encryption
 * AI/ML Training Sample
 */

class Encryption {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object EncryptionExample {
  def main(args: Array[String]): Unit = {
    val instance = new Encryption()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
