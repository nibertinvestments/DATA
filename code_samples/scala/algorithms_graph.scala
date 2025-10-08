/**
 * Algorithms: Graph
 * AI/ML Training Sample
 */

class Graph {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object GraphExample {
  def main(args: Array[String]): Unit = {
    val instance = new Graph()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
