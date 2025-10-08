/**
 * Data Structures: Hash Table
 * AI/ML Training Sample
 */

class HashTable {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object HashTableExample {
  def main(args: Array[String]): Unit = {
    val instance = new HashTable()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
