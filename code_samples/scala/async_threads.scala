/**
 * Async: Threads
 * AI/ML Training Sample
 */

class Threads {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ThreadsExample {
  def main(args: Array[String]): Unit = {
    val instance = new Threads()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
