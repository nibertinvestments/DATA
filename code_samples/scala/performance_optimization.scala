/**
 * Performance: Optimization
 * AI/ML Training Sample
 */

class Optimization {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object OptimizationExample {
  def main(args: Array[String]): Unit = {
    val instance = new Optimization()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
