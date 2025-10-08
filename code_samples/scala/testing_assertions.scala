/**
 * Testing: Assertions
 * AI/ML Training Sample
 */

class Assertions {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object AssertionsExample {
  def main(args: Array[String]): Unit = {
    val instance = new Assertions()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
