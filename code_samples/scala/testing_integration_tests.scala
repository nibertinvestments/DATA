/**
 * Testing: Integration Tests
 * AI/ML Training Sample
 */

class IntegrationTests {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object IntegrationTestsExample {
  def main(args: Array[String]): Unit = {
    val instance = new IntegrationTests()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
