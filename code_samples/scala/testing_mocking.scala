/**
 * Testing: Mocking
 * AI/ML Training Sample
 */

class Mocking {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object MockingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Mocking()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
