/**
 * Design Patterns: Singleton
 * AI/ML Training Sample
 */

class Singleton {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object SingletonExample {
  def main(args: Array[String]): Unit = {
    val instance = new Singleton()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
