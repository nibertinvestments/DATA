/**
 * Oop: Encapsulation
 * AI/ML Training Sample
 */

class Encapsulation {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object EncapsulationExample {
  def main(args: Array[String]): Unit = {
    val instance = new Encapsulation()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
