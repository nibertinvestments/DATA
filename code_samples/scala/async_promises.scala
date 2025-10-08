/**
 * Async: Promises
 * AI/ML Training Sample
 */

class Promises {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object PromisesExample {
  def main(args: Array[String]): Unit = {
    val instance = new Promises()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
