/**
 * Database: Orm
 * AI/ML Training Sample
 */

class Orm {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object OrmExample {
  def main(args: Array[String]): Unit = {
    val instance = new Orm()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
