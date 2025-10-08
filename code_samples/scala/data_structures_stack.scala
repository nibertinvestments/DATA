/**
 * Data Structures: Stack
 * AI/ML Training Sample
 */

class Stack {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object StackExample {
  def main(args: Array[String]): Unit = {
    val instance = new Stack()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
