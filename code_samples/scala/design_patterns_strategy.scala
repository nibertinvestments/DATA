/**
 * Design Patterns: Strategy
 * AI/ML Training Sample
 */

class Strategy {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object StrategyExample {
  def main(args: Array[String]): Unit = {
    val instance = new Strategy()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
