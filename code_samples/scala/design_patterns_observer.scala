/**
 * Design Patterns: Observer
 * AI/ML Training Sample
 */

class Observer {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ObserverExample {
  def main(args: Array[String]): Unit = {
    val instance = new Observer()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
