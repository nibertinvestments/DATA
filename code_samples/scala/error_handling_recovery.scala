/**
 * Error Handling: Recovery
 * AI/ML Training Sample
 */

class Recovery {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object RecoveryExample {
  def main(args: Array[String]): Unit = {
    val instance = new Recovery()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
