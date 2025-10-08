/**
 * File Operations: Reading
 * AI/ML Training Sample
 */

class Reading {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object ReadingExample {
  def main(args: Array[String]): Unit = {
    val instance = new Reading()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
