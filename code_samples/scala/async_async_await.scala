/**
 * Async: Async Await
 * AI/ML Training Sample
 */

class AsyncAwait {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object AsyncAwaitExample {
  def main(args: Array[String]): Unit = {
    val instance = new AsyncAwait()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
