/**
 * Error Handling: Try Catch
 * AI/ML Training Sample
 */

class TryCatch {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object TryCatchExample {
  def main(args: Array[String]): Unit = {
    val instance = new TryCatch()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
