/**
 * Utilities: Regex
 * AI/ML Training Sample
 */

class Regex {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object RegexExample {
  def main(args: Array[String]): Unit = {
    val instance = new Regex()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
