// Data Structures: Trie
// AI/ML Training Sample

class Trie {
  String _data = '';
  
  void process(String input) {
    _data = input;
  }
  
  String getData() => _data;
  
  bool validate() => _data.isNotEmpty;
}

void main() {
  final instance = Trie();
  instance.process('example');
  print('Data: ${instance.getData()}');
  print('Valid: ${instance.validate()}');
}
