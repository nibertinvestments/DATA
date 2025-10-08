// Data Structures: Stack
// AI/ML Training Sample

class Stack {
  String _data = '';
  
  void process(String input) {
    _data = input;
  }
  
  String getData() => _data;
  
  bool validate() => _data.isNotEmpty;
}

void main() {
  final instance = Stack();
  instance.process('example');
  print('Data: ${instance.getData()}');
  print('Valid: ${instance.validate()}');
}
