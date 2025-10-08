/**
 * Data Structures: Heap
 * AI/ML Training Sample
 */

class Heap {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object HeapExample {
  def main(args: Array[String]): Unit = {
    val instance = new Heap()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
