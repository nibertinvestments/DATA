/**
 * Data Structures: Queue
 * AI/ML Training Sample
 */

class Queue {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object QueueExample {
  def main(args: Array[String]): Unit = {
    val instance = new Queue()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
