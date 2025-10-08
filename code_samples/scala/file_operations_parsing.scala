/**
 * File Operations: Parsing
 * AI/ML Training Sample
 */

class Parsing {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ParsingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Parsing()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
