/**
 * Performance: Caching
 * AI/ML Training Sample
 */

class Caching {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object CachingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Caching()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
