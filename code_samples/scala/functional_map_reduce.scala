/**
 * Functional: Map Reduce
 * AI/ML Training Sample
 */

class MapReduce {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object MapReduceExample {
  def main(args: Array[String]): Unit = {
    val instance = new MapReduce()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
