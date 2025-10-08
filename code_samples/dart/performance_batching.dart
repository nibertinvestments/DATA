// Performance: Batching
// AI/ML Training Sample

class Batching {
  String _data = '';
  
  void process(String input) {
    _data = input;
  }
  
  String getData() => _data;
  
  bool validate() => _data.isNotEmpty;
}

void main() {
  final instance = Batching();
  instance.process('example');
  print('Data: ${instance.getData()}');
  print('Valid: ${instance.validate()}');
}
