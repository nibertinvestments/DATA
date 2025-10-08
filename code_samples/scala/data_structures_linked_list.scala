/**
 * Data Structures: Linked List
 * AI/ML Training Sample
 */

class LinkedList {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object LinkedListExample {
  def main(args: Array[String]): Unit = {
    val instance = new LinkedList()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
