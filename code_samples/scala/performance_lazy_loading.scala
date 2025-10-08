/**
 * Performance: Lazy Loading
 * AI/ML Training Sample
 */

class LazyLoading {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object LazyLoadingExample {
  def main(args: Array[String]): Unit = {
    val instance = new LazyLoading()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
