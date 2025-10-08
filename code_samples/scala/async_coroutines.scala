/**
 * Async: Coroutines
 * AI/ML Training Sample
 */

class Coroutines {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object CoroutinesExample {
  def main(args: Array[String]): Unit = {
    val instance = new Coroutines()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
