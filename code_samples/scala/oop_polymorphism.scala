/**
 * Oop: Polymorphism
 * AI/ML Training Sample
 */

class Polymorphism {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object PolymorphismExample {
  def main(args: Array[String]): Unit = {
    val instance = new Polymorphism()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
