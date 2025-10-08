/**
 * Testing: Fixtures
 * AI/ML Training Sample
 */

class Fixtures {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object FixturesExample {
  def main(args: Array[String]): Unit = {
    val instance = new Fixtures()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
