# Basic Rust Dataset - Systems Programming with Safety

## Dataset 1: Hello World and Basic Structure
```rust
// Simple Hello World
fn main() {
    println!("Hello, World!");
}

// Hello World with functions
fn greet() {
    println!("Hello, World!");
}

fn greet_with_name(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    greet();
    greet_with_name("Rust");
}
```

## Dataset 2: Variables, Mutability, and Data Types
```rust
fn main() {
    // Immutable variables (default)
    let message = "Hello Rust";
    let count = 42;
    let price = 3.14;
    let is_active = true;
    
    // Mutable variables
    let mut name = String::from("John");
    let mut age = 30;
    
    // Shadowing
    let spaces = "   ";
    let spaces = spaces.len();
    
    // Type annotations
    let number: i32 = 100;
    let decimal: f64 = 3.14159;
    
    // Arrays and tuples
    let numbers: [i32; 5] = [1, 2, 3, 4, 5];
    let person: (String, i32) = ("Alice".to_string(), 25);
    
    // Vectors (dynamic arrays)
    let mut fruits = vec!["apple", "banana", "cherry"];
    fruits.push("date");
    
    // Output
    println!("Message: {}", message);
    println!("Count: {}", count);
    println!("Price: {}", price);
    println!("Is Active: {}", is_active);
    println!("Name: {}", name);
    println!("Age: {}", age);
    println!("Spaces count: {}", spaces);
    println!("Numbers: {:?}", numbers);
    println!("Person: {:?}", person);
    println!("Fruits: {:?}", fruits);
    
    // Modifying mutable variables
    name.push_str(" Doe");
    age += 1;
    println!("Updated name: {}", name);
    println!("Updated age: {}", age);
}
```

## Dataset 3: Control Structures
```rust
fn main() {
    // If expressions
    let age = 18;
    if age >= 18 {
        println!("Adult");
    } else if age >= 13 {
        println!("Teenager");
    } else {
        println!("Child");
    }
    
    // If as expression
    let status = if age >= 18 { "adult" } else { "minor" };
    println!("Status: {}", status);
    
    // Loop
    let mut counter = 0;
    loop {
        counter += 1;
        println!("Loop iteration: {}", counter);
        if counter == 3 {
            break;
        }
    }
    
    // While loop
    let mut count = 0;
    while count < 5 {
        println!("Count: {}", count);
        count += 1;
    }
    
    // For loop with range
    for i in 0..5 {
        println!("Number: {}", i);
    }
    
    // For loop with collection
    let numbers = vec![1, 2, 3, 4, 5];
    for (index, value) in numbers.iter().enumerate() {
        println!("Index: {}, Value: {}", index, value);
    }
    
    // Match expression (pattern matching)
    let day = "Monday";
    match day {
        "Monday" => println!("Start of the week"),
        "Friday" => println!("TGIF!"),
        _ => println!("Regular day"),
    }
    
    // Match with numbers
    let number = 3;
    match number {
        1 => println!("One"),
        2 | 3 => println!("Two or Three"),
        4..=10 => println!("Four to Ten"),
        _ => println!("Something else"),
    }
}
```

## Dataset 4: Functions and Ownership
```rust
// Function with parameters and return value
fn add(a: i32, b: i32) -> i32 {
    a + b // No semicolon for expression return
}

// Function that takes ownership
fn take_ownership(s: String) {
    println!("Taking ownership of: {}", s);
} // s goes out of scope and memory is freed

// Function that borrows (doesn't take ownership)
fn borrow_string(s: &String) {
    println!("Borrowing: {}", s);
}

// Function that returns ownership
fn give_ownership() -> String {
    String::from("Hello from function")
}

// Function with mutable reference
fn change_string(s: &mut String) {
    s.push_str(", World!");
}

fn main() {
    // Basic function call
    let result = add(5, 3);
    println!("Add result: {}", result);
    
    // Ownership example
    let s1 = String::from("Hello");
    take_ownership(s1); // s1 is moved, can't use it after this
    // println!("{}", s1); // This would cause a compile error
    
    // Borrowing example
    let s2 = String::from("Hello");
    borrow_string(&s2);
    println!("s2 is still valid: {}", s2); // s2 is still valid
    
    // Getting ownership from function
    let s3 = give_ownership();
    println!("Got ownership: {}", s3);
    
    // Mutable reference
    let mut s4 = String::from("Hello");
    change_string(&mut s4);
    println!("Changed string: {}", s4);
    
    // Cloning to avoid move
    let s5 = String::from("Clone me");
    let s6 = s5.clone();
    println!("Original: {}, Clone: {}", s5, s6);
}
```

## Dataset 5: Structs and Implementation Blocks
```rust
// Basic struct
struct Person {
    name: String,
    age: u32,
    email: String,
}

// Tuple struct
struct Point(i32, i32, i32);

// Unit struct
struct Unit;

// Implementation block for methods
impl Person {
    // Associated function (constructor)
    fn new(name: String, age: u32, email: String) -> Person {
        Person { name, age, email }
    }
    
    // Method that borrows self
    fn introduce(&self) -> String {
        format!("Hi, I'm {} and I'm {} years old", self.name, self.age)
    }
    
    // Method that borrows self mutably
    fn have_birthday(&mut self) {
        self.age += 1;
        println!("Happy birthday! Now I'm {}", self.age);
    }
    
    // Method that takes ownership of self
    fn into_email(self) -> String {
        self.email
    }
}

fn main() {
    // Creating structs
    let mut person1 = Person {
        name: String::from("Alice"),
        age: 25,
        email: String::from("alice@example.com"),
    };
    
    // Using constructor
    let person2 = Person::new(
        String::from("Bob"),
        30,
        String::from("bob@example.com"),
    );
    
    // Struct update syntax
    let person3 = Person {
        name: String::from("Carol"),
        ..person2 // Copy remaining fields from person2
    };
    
    // Method calls
    println!("{}", person1.introduce());
    person1.have_birthday();
    
    println!("{}", person2.introduce());
    println!("{}", person3.introduce());
    
    // Tuple struct
    let point = Point(10, 20, 30);
    println!("Point: ({}, {}, {})", point.0, point.1, point.2);
    
    // Taking ownership
    let email = person1.into_email();
    println!("Email: {}", email);
    // person1 is no longer valid after this
}
```

## Dataset 6: Enums and Pattern Matching
```rust
// Basic enum
#[derive(Debug)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

// Enum with data
#[derive(Debug)]
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

// Enum with different types
#[derive(Debug)]
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn process(&self) {
        match self {
            Message::Quit => println!("Quit message received"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(text) => println!("Write: {}", text),
            Message::ChangeColor(r, g, b) => println!("Change color to RGB({}, {}, {})", r, g, b),
        }
    }
}

fn main() {
    // Using basic enum
    let direction = Direction::Up;
    println!("Direction: {:?}", direction);
    
    // Pattern matching with enum
    match direction {
        Direction::Up => println!("Going up!"),
        Direction::Down => println!("Going down!"),
        Direction::Left => println!("Going left!"),
        Direction::Right => println!("Going right!"),
    }
    
    // Enum with data
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));
    
    println!("Home: {:?}", home);
    println!("Loopback: {:?}", loopback);
    
    // Complex enum usage
    let messages = vec![
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("Hello, Rust!")),
        Message::ChangeColor(255, 0, 0),
    ];
    
    for message in messages {
        message.process();
    }
    
    // Option enum (built-in)
    let some_number = Some(5);
    let some_string = Some("a string");
    let absent_number: Option<i32> = None;
    
    match some_number {
        Some(value) => println!("Got a value: {}", value),
        None => println!("No value"),
    }
    
    // Using if let for cleaner code
    if let Some(value) = some_string {
        println!("Got string: {}", value);
    }
    
    // Result enum for error handling
    let result: Result<i32, &str> = Ok(42);
    match result {
        Ok(value) => println!("Success: {}", value),
        Err(error) => println!("Error: {}", error),
    }
}
```

## Dataset 7: Collections and Iterators
```rust
use std::collections::HashMap;

fn main() {
    // Vectors
    let mut numbers = vec![1, 2, 3, 4, 5];
    numbers.push(6);
    numbers.push(7);
    
    println!("Numbers: {:?}", numbers);
    println!("First element: {:?}", numbers.get(0));
    println!("Length: {}", numbers.len());
    
    // Iterating over vectors
    for number in &numbers {
        println!("Number: {}", number);
    }
    
    // Iterator methods
    let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
    println!("Doubled: {:?}", doubled);
    
    let evens: Vec<&i32> = numbers.iter().filter(|&x| x % 2 == 0).collect();
    println!("Evens: {:?}", evens);
    
    let sum: i32 = numbers.iter().sum();
    println!("Sum: {}", sum);
    
    // Strings
    let mut s = String::from("Hello");
    s.push_str(", World!");
    s.push('!');
    
    println!("String: {}", s);
    println!("Length: {}", s.len());
    
    // String slices
    let hello = &s[0..5];
    let world = &s[7..12];
    println!("Hello: {}, World: {}", hello, world);
    
    // Hash maps
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    
    // Another way to create HashMap
    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let initial_scores = vec![10, 50];
    let mut scores2: HashMap<_, _> = teams.into_iter().zip(initial_scores.into_iter()).collect();
    
    // Accessing values
    let team_name = String::from("Blue");
    let score = scores.get(&team_name);
    match score {
        Some(value) => println!("Blue team score: {}", value),
        None => println!("Team not found"),
    }
    
    // Iterating over HashMap
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
    
    // Updating HashMap
    scores.insert(String::from("Blue"), 25); // Overwrite
    scores.entry(String::from("Red")).or_insert(30); // Insert if not exists
    
    println!("Updated scores: {:?}", scores);
}
```

## Dataset 8: Error Handling
```rust
use std::fs::File;
use std::io::{self, Read};

// Custom error type
#[derive(Debug)]
enum MathError {
    DivisionByZero,
    NegativeSquareRoot,
}

impl std::fmt::Display for MathError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MathError::DivisionByZero => write!(f, "Division by zero"),
            MathError::NegativeSquareRoot => write!(f, "Square root of negative number"),
        }
    }
}

impl std::error::Error for MathError {}

// Function that returns Result
fn divide(a: f64, b: f64) -> Result<f64, MathError> {
    if b == 0.0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}

fn square_root(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        Err(MathError::NegativeSquareRoot)
    } else {
        Ok(x.sqrt())
    }
}

// Function that uses ? operator
fn calculate(a: f64, b: f64) -> Result<f64, MathError> {
    let quotient = divide(a, b)?;
    let result = square_root(quotient)?;
    Ok(result)
}

// Function that reads file and handles multiple error types
fn read_file_content(filename: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn main() {
    // Basic Result handling
    match divide(10.0, 2.0) {
        Ok(result) => println!("Division result: {}", result),
        Err(error) => println!("Error: {}", error),
    }
    
    match divide(10.0, 0.0) {
        Ok(result) => println!("Division result: {}", result),
        Err(error) => println!("Error: {}", error),
    }
    
    // Using unwrap (panics on error)
    // let result = divide(10.0, 0.0).unwrap(); // This would panic
    
    // Using unwrap_or for default value
    let result = divide(10.0, 0.0).unwrap_or(0.0);
    println!("Result with default: {}", result);
    
    // Using ? operator
    match calculate(16.0, 4.0) {
        Ok(result) => println!("Calculate result: {}", result),
        Err(error) => println!("Calculate error: {}", error),
    }
    
    match calculate(16.0, 0.0) {
        Ok(result) => println!("Calculate result: {}", result),
        Err(error) => println!("Calculate error: {}", error),
    }
    
    // File reading example
    match read_file_content("nonexistent.txt") {
        Ok(content) => println!("File content: {}", content),
        Err(error) => println!("File read error: {}", error),
    }
    
    // Option handling
    let numbers = vec![1, 2, 3, 4, 5];
    
    // Using match
    match numbers.get(10) {
        Some(value) => println!("Value at index 10: {}", value),
        None => println!("No value at index 10"),
    }
    
    // Using if let
    if let Some(value) = numbers.get(2) {
        println!("Value at index 2: {}", value);
    }
    
    // Using unwrap_or
    let value = numbers.get(10).unwrap_or(&0);
    println!("Value with default: {}", value);
}
```

## Dataset 9: Traits and Generics
```rust
// Basic trait
trait Summary {
    fn summarize(&self) -> String;
    
    // Default implementation
    fn author(&self) -> String {
        String::from("Unknown author")
    }
}

// Struct implementing trait
struct Article {
    headline: String,
    content: String,
    author: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{} by {}", self.headline, self.author)
    }
    
    fn author(&self) -> String {
        self.author.clone()
    }
}

struct Tweet {
    username: String,
    content: String,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}

// Generic function
fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
    let mut largest = list[0];
    for &item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Generic struct
#[derive(Debug)]
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
    
    fn x(&self) -> &T {
        &self.x
    }
}

impl<T: std::fmt::Display> Point<T> {
    fn display(&self) {
        println!("Point({}, {})", self.x, self.y);
    }
}

// Trait bounds
fn notify<T: Summary>(item: &T) {
    println!("Breaking news: {}", item.summarize());
}

// Multiple trait bounds
fn display_and_summarize<T: Summary + std::fmt::Display>(item: &T) {
    println!("Display: {}", item);
    println!("Summary: {}", item.summarize());
}

fn main() {
    // Using traits
    let article = Article {
        headline: String::from("Rust is Amazing"),
        content: String::from("Rust provides memory safety without garbage collection..."),
        author: String::from("Jane Doe"),
    };
    
    let tweet = Tweet {
        username: String::from("rustlang"),
        content: String::from("Announcing Rust 1.70!"),
    };
    
    println!("Article summary: {}", article.summarize());
    println!("Tweet summary: {}", tweet.summarize());
    println!("Article author: {}", article.author());
    println!("Tweet author: {}", tweet.author()); // Uses default implementation
    
    // Using generic function
    let numbers = vec![34, 50, 25, 100, 65];
    let largest_num = largest(&numbers);
    println!("Largest number: {}", largest_num);
    
    let chars = vec!['y', 'm', 'a', 'q'];
    let largest_char = largest(&chars);
    println!("Largest char: {}", largest_char);
    
    // Using generic struct
    let integer_point = Point::new(5, 10);
    let float_point = Point::new(1.0, 4.0);
    
    println!("Integer point: {:?}", integer_point);
    println!("Float point: {:?}", float_point);
    println!("Integer point x: {}", integer_point.x());
    
    integer_point.display();
    float_point.display();
    
    // Using trait bounds
    notify(&article);
    notify(&tweet);
}
```

## Dataset 10: Smart Pointers and Concurrency
```rust
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// Box - heap allocation
fn box_example() {
    let b = Box::new(5);
    println!("Box value: {}", b);
    
    // Recursive type using Box
    #[derive(Debug)]
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }
    
    use List::{Cons, Nil};
    
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("List: {:?}", list);
}

// Rc - Reference counting
fn rc_example() {
    let data = Rc::new(vec![1, 2, 3, 4, 5]);
    let data1 = Rc::clone(&data);
    let data2 = Rc::clone(&data);
    
    println!("Data: {:?}", data);
    println!("Reference count: {}", Rc::strong_count(&data));
    
    drop(data1);
    println!("Reference count after drop: {}", Rc::strong_count(&data));
}

// RefCell - Interior mutability
fn refcell_example() {
    let data = RefCell::new(vec![1, 2, 3]);
    
    // Borrowing mutably
    data.borrow_mut().push(4);
    data.borrow_mut().push(5);
    
    println!("RefCell data: {:?}", data.borrow());
}

// Arc and Mutex for thread safety
fn thread_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            println!("Thread {} incremented counter", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final counter value: {}", *counter.lock().unwrap());
}

// Thread communication with channels
fn channel_example() {
    use std::sync::mpsc;
    
    let (tx, rx) = mpsc::channel();
    
    // Spawn multiple senders
    for i in 0..5 {
        let tx = tx.clone();
        thread::spawn(move || {
            let message = format!("Message from thread {}", i);
            tx.send(message).unwrap();
            thread::sleep(Duration::from_millis(100));
        });
    }
    
    // Drop the original sender
    drop(tx);
    
    // Receive messages
    for received in rx {
        println!("Received: {}", received);
    }
}

fn main() {
    println!("=== Box Example ===");
    box_example();
    
    println!("\n=== Rc Example ===");
    rc_example();
    
    println!("\n=== RefCell Example ===");
    refcell_example();
    
    println!("\n=== Thread Example ===");
    thread_example();
    
    println!("\n=== Channel Example ===");
    channel_example();
    
    // Basic thread spawning
    println!("\n=== Basic Threading ===");
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("Spawned thread: {}", i);
            thread::sleep(Duration::from_millis(50));
        }
    });
    
    for i in 1..5 {
        println!("Main thread: {}", i);
        thread::sleep(Duration::from_millis(50));
    }
    
    handle.join().unwrap();
    println!("All threads completed");
}
```