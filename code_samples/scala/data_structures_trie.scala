/**
 * Data Structures: Trie
 * AI/ML Training Sample
 */

class Trie {
  private var data: String = ""
  
  def process(input: String): Unit = {
    data = input
  }
  
  def getData: String = data
  
  def validate: Boolean = data.nonEmpty
}

object TrieExample {
  def main(args: Array[String]): Unit = {
    val instance = new Trie()
    instance.process("example")
    println(s"Data: ${instance.getData}")
    println(s"Valid: ${instance.validate}")
  }
}
