/**
 * Utilities: Collections
 * AI/ML Training Sample
 */

class Collections {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object CollectionsExample {
  def main(args: Array[String]): Unit = {
    val instance = new Collections()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
