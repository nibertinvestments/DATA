// Networking: Socket Programming
// AI/ML Training Sample

class SocketProgramming {
  String _data = '';
  
  void process(String input) {
    _data = input;
  }
  
  String getData() => _data;
  
  bool validate() => _data.isNotEmpty;
}

void main() {
  final instance = SocketProgramming();
  instance.process('example');
  print('Data: ${instance.getData()}');
  print('Valid: ${instance.validate()}');
}
