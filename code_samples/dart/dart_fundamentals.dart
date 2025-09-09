// Basic Dart examples for ML/AI training

// Data structures and collections
import 'dart:math';
import 'dart:io';

// Custom class demonstrating object-oriented programming
class Person {
  String name;
  int age;
  String email;

  Person(this.name, this.age, this.email);

  // Named constructor
  Person.guest() : name = 'Guest', age = 0, email = 'guest@example.com';

  // Getters and setters
  bool get isAdult => age >= 18;
  
  set updateAge(int newAge) {
    if (newAge >= 0) age = newAge;
  }

  // Method with optional parameters
  String greet([String greeting = 'Hello']) {
    return '$greeting, I\'m $name and I\'m $age years old.';
  }

  // Override toString
  @override
  String toString() => 'Person(name: $name, age: $age, email: $email)';
}

// Generic class example
class Stack<T> {
  List<T> _items = [];

  void push(T item) => _items.add(item);

  T? pop() => _items.isNotEmpty ? _items.removeLast() : null;

  T? peek() => _items.isNotEmpty ? _items.last : null;

  bool get isEmpty => _items.isEmpty;
  int get length => _items.length;

  @override
  String toString() => 'Stack: $_items';
}

// Mixin example
mixin Flyable {
  void fly() => print('Flying!');
  double altitude = 0.0;
}

class Bird with Flyable {
  String species;
  
  Bird(this.species);

  void chirp() => print('$species is chirping!');
}

// Abstract class and inheritance
abstract class Animal {
  String name;
  
  Animal(this.name);
  
  void makeSound(); // Abstract method
  
  void sleep() => print('$name is sleeping');
}

class Dog extends Animal {
  String breed;
  
  Dog(String name, this.breed) : super(name);
  
  @override
  void makeSound() => print('$name the $breed says Woof!');
  
  void fetch() => print('$name is fetching the ball!');
}

// Enum example
enum TaskStatus { pending, inProgress, completed, cancelled }

class Task {
  String title;
  TaskStatus status;
  DateTime createdAt;
  DateTime? completedAt;

  Task(this.title) : status = TaskStatus.pending, createdAt = DateTime.now();

  void complete() {
    status = TaskStatus.completed;
    completedAt = DateTime.now();
  }

  Duration? get duration {
    if (completedAt != null) {
      return completedAt!.difference(createdAt);
    }
    return null;
  }

  @override
  String toString() {
    String statusStr = status.toString().split('.').last;
    return 'Task: $title [$statusStr]';
  }
}

// Future and async programming
Future<String> fetchUserData(int userId) async {
  // Simulate network delay
  await Future.delayed(Duration(seconds: 1));
  
  if (userId <= 0) {
    throw ArgumentError('User ID must be positive');
  }
  
  return 'User data for ID: $userId';
}

Future<List<String>> fetchMultipleUsers(List<int> userIds) async {
  List<Future<String>> futures = userIds.map(fetchUserData).toList();
  return await Future.wait(futures);
}

// Stream example
Stream<int> countStream(int max) async* {
  for (int i = 1; i <= max; i++) {
    await Future.delayed(Duration(milliseconds: 100));
    yield i;
  }
}

// Extension methods
extension StringExtensions on String {
  bool get isEmail => contains('@') && contains('.');
  
  String get capitalize {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1).toLowerCase()}';
  }
  
  String truncate(int maxLength) {
    if (length <= maxLength) return this;
    return '${substring(0, maxLength)}...';
  }
}

// Higher-order functions and functional programming
List<T> filter<T>(List<T> list, bool Function(T) predicate) {
  return list.where(predicate).toList();
}

List<R> map<T, R>(List<T> list, R Function(T) transform) {
  return list.map(transform).toList();
}

T reduce<T>(List<T> list, T Function(T, T) combine) {
  if (list.isEmpty) throw StateError('Cannot reduce empty list');
  return list.reduce(combine);
}

// Pattern matching (Dart 3.0+)
String describeNumber(dynamic value) {
  return switch (value) {
    int i when i < 0 => 'Negative integer: $i',
    int i when i == 0 => 'Zero',
    int i when i > 0 => 'Positive integer: $i',
    double d => 'Floating point: $d',
    String s => 'String: "$s"',
    _ => 'Unknown type'
  };
}

// Record types (Dart 3.0+)
({String name, int age}) createPersonRecord(String name, int age) {
  return (name: name, age: age);
}

// Main function demonstrating all concepts
void main() async {
  print('Dart Programming Examples');
  print('=========================\n');

  // Basic object creation and usage
  var person1 = Person('Alice', 25, 'alice@example.com');
  var person2 = Person.guest();
  
  print('Person 1: $person1');
  print('Person 2: $person2');
  print('Is person1 an adult? ${person1.isAdult}');
  print('Greeting: ${person1.greet('Hi')}\n');

  // Generic collections
  var numberStack = Stack<int>();
  numberStack.push(1);
  numberStack.push(2);
  numberStack.push(3);
  
  print('Stack: $numberStack');
  print('Popped: ${numberStack.pop()}');
  print('Peek: ${numberStack.peek()}\n');

  // Inheritance and polymorphism
  var dog = Dog('Buddy', 'Golden Retriever');
  dog.makeSound();
  dog.fetch();
  dog.sleep();
  
  var bird = Bird('Eagle');
  bird.chirp();
  bird.fly();
  print('Bird altitude: ${bird.altitude}\n');

  // Enum and task management
  var task = Task('Learn Dart');
  print('Created: $task');
  task.status = TaskStatus.inProgress;
  print('Updated: $task');
  task.complete();
  print('Completed: $task');
  print('Duration: ${task.duration}\n');

  // Collections and functional programming
  var numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  
  var evenNumbers = filter(numbers, (n) => n % 2 == 0);
  print('Even numbers: $evenNumbers');
  
  var squares = map(numbers, (n) => n * n);
  print('Squares: $squares');
  
  var sum = reduce(numbers, (a, b) => a + b);
  print('Sum: $sum\n');

  // String extensions
  var email = 'user@example.com';
  var text = 'hello world';
  var longText = 'This is a very long text that needs to be truncated';
  
  print('Is email valid? ${email.isEmail}');
  print('Capitalized: ${text.capitalize}');
  print('Truncated: ${longText.truncate(20)}\n');

  // Async programming
  try {
    print('Fetching user data...');
    var userData = await fetchUserData(123);
    print('Result: $userData');
    
    var multipleUsers = await fetchMultipleUsers([1, 2, 3]);
    print('Multiple users: $multipleUsers\n');
  } catch (e) {
    print('Error: $e\n');
  }

  // Streams
  print('Counting stream:');
  await for (var number in countStream(5)) {
    print('Count: $number');
  }
  print();

  // Pattern matching
  var values = [42, -5, 3.14, 'hello', true];
  for (var value in values) {
    print(describeNumber(value));
  }
  print();

  // Records
  var personRecord = createPersonRecord('Bob', 30);
  print('Person record: ${personRecord.name}, age ${personRecord.age}');

  // Map operations
  var userMap = {
    'name': 'Charlie',
    'age': 28,
    'city': 'New York'
  };
  
  print('\nUser map:');
  userMap.forEach((key, value) => print('$key: $value'));
  
  // Set operations
  var set1 = {1, 2, 3, 4, 5};
  var set2 = {4, 5, 6, 7, 8};
  
  print('\nSet operations:');
  print('Set 1: $set1');
  print('Set 2: $set2');
  print('Union: ${set1.union(set2)}');
  print('Intersection: ${set1.intersection(set2)}');
  print('Difference: ${set1.difference(set2)}');
  
  print('\nAll Dart examples completed!');
}