/**
 * Database: Query Builder
 * AI/ML Training Sample
 */

class QueryBuilder {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object QueryBuilderExample {
  def main(args: Array[String]): Unit = {
    val instance = new QueryBuilder()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
