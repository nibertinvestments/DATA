// Error Handling: Custom Exceptions
// AI/ML Training Sample

class CustomExceptions {
  String _data = '';
  
  void process(String input) {
    _data = input;
  }
  
  String getData() => _data;
  
  bool validate() => _data.isNotEmpty;
}

void main() {
  final instance = CustomExceptions();
  instance.process('example');
  print('Data: ${instance.getData()}');
  print('Valid: ${instance.validate()}');
}
