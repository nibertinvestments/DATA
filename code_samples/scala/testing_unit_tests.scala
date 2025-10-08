/**
 * Testing: Unit Tests
 * AI/ML Training Sample
 */

class UnitTests {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object UnitTestsExample {
  def main(args: Array[String]): Unit = {
    val instance = new UnitTests()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
