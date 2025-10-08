/**
 * Utilities: String Manipulation
 * AI/ML Training Sample
 */

class StringManipulation {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object StringManipulationExample {
  def main(args: Array[String]): Unit = {
    val instance = new StringManipulation()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
