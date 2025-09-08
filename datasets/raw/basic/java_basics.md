# Basic Java Dataset - Object-Oriented Programming Fundamentals

## Dataset 1: Hello World and Basic Structure
```java
// Simple Hello World program
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}

// Hello World with methods
public class Greeting {
    public static void main(String[] args) {
        Greeting greeter = new Greeting();
        greeter.sayHello();
        greeter.sayHello("Java");
    }
    
    public void sayHello() {
        System.out.println("Hello, World!");
    }
    
    public void sayHello(String name) {
        System.out.println("Hello, " + name + "!");
    }
}
```

## Dataset 2: Variables and Data Types
```java
public class DataTypes {
    public static void main(String[] args) {
        // Primitive data types
        byte byteVar = 127;
        short shortVar = 32767;
        int intVar = 2147483647;
        long longVar = 9223372036854775807L;
        float floatVar = 3.14f;
        double doubleVar = 3.14159;
        char charVar = 'A';
        boolean boolVar = true;
        
        // Reference types
        String stringVar = "Hello Java";
        int[] arrayVar = {1, 2, 3, 4, 5};
        
        // Output
        System.out.println("Byte: " + byteVar);
        System.out.println("Short: " + shortVar);
        System.out.println("Int: " + intVar);
        System.out.println("Long: " + longVar);
        System.out.println("Float: " + floatVar);
        System.out.println("Double: " + doubleVar);
        System.out.println("Char: " + charVar);
        System.out.println("Boolean: " + boolVar);
        System.out.println("String: " + stringVar);
        System.out.println("Array length: " + arrayVar.length);
    }
}
```

## Dataset 3: Control Structures
```java
public class ControlStructures {
    public static void main(String[] args) {
        // If-else statements
        int age = 18;
        if (age >= 18) {
            System.out.println("Adult");
        } else if (age >= 13) {
            System.out.println("Teenager");
        } else {
            System.out.println("Child");
        }
        
        // For loops
        for (int i = 0; i < 5; i++) {
            System.out.println("Number: " + i);
        }
        
        // Enhanced for loop
        int[] numbers = {1, 2, 3, 4, 5};
        for (int number : numbers) {
            System.out.println("Enhanced for: " + number);
        }
        
        // While loop
        int count = 0;
        while (count < 5) {
            System.out.println("Count: " + count);
            count++;
        }
        
        // Switch statement
        String day = "Monday";
        switch (day) {
            case "Monday":
                System.out.println("Start of the week");
                break;
            case "Friday":
                System.out.println("TGIF!");
                break;
            default:
                System.out.println("Regular day");
        }
    }
}
```

## Dataset 4: Methods and Functions
```java
public class Methods {
    public static void main(String[] args) {
        Methods obj = new Methods();
        
        // Method calls
        int sum = obj.add(5, 3);
        System.out.println("Sum: " + sum);
        
        obj.greet("Alice");
        obj.printNumbers(1, 2, 3, 4, 5);
        
        // Static method call
        int product = multiply(4, 5);
        System.out.println("Product: " + product);
    }
    
    // Instance method
    public int add(int a, int b) {
        return a + b;
    }
    
    // Method with void return type
    public void greet(String name) {
        System.out.println("Hello, " + name + "!");
    }
    
    // Variable arguments (varargs)
    public void printNumbers(int... numbers) {
        for (int number : numbers) {
            System.out.print(number + " ");
        }
        System.out.println();
    }
    
    // Static method
    public static int multiply(int a, int b) {
        return a * b;
    }
    
    // Method overloading
    public double add(double a, double b) {
        return a + b;
    }
}
```

## Dataset 5: Classes and Objects
```java
// Basic class definition
class Person {
    private String name;
    private int age;
    
    // Constructor
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // Default constructor
    public Person() {
        this.name = "Unknown";
        this.age = 0;
    }
    
    // Getter methods
    public String getName() {
        return name;
    }
    
    public int getAge() {
        return age;
    }
    
    // Setter methods
    public void setName(String name) {
        this.name = name;
    }
    
    public void setAge(int age) {
        if (age >= 0) {
            this.age = age;
        }
    }
    
    // Instance method
    public void introduce() {
        System.out.println("Hi, I'm " + name + " and I'm " + age + " years old");
    }
    
    public void haveBirthday() {
        age++;
        System.out.println("Happy birthday! Now I'm " + age);
    }
}

public class ClassesAndObjects {
    public static void main(String[] args) {
        // Creating objects
        Person person1 = new Person("Alice", 25);
        Person person2 = new Person();
        
        // Using objects
        person1.introduce();
        person1.haveBirthday();
        
        person2.setName("Bob");
        person2.setAge(30);
        person2.introduce();
    }
}
```

## Dataset 6: Arrays and Collections
```java
import java.util.*;

public class ArraysAndCollections {
    public static void main(String[] args) {
        // Arrays
        int[] numbers = {1, 2, 3, 4, 5};
        String[] fruits = new String[3];
        fruits[0] = "Apple";
        fruits[1] = "Banana";
        fruits[2] = "Cherry";
        
        // Array iteration
        for (int i = 0; i < numbers.length; i++) {
            System.out.println("numbers[" + i + "] = " + numbers[i]);
        }
        
        for (String fruit : fruits) {
            System.out.println("Fruit: " + fruit);
        }
        
        // ArrayList
        ArrayList<String> list = new ArrayList<>();
        list.add("First");
        list.add("Second");
        list.add("Third");
        list.remove(1);
        
        for (String item : list) {
            System.out.println("List item: " + item);
        }
        
        // HashMap
        HashMap<String, Integer> map = new HashMap<>();
        map.put("Apple", 5);
        map.put("Banana", 3);
        map.put("Cherry", 8);
        
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}
```

## Dataset 7: String Manipulation
```java
public class StringManipulation {
    public static void main(String[] args) {
        String text = "Hello, World!";
        
        // String methods
        System.out.println("Original: " + text);
        System.out.println("Uppercase: " + text.toUpperCase());
        System.out.println("Lowercase: " + text.toLowerCase());
        System.out.println("Length: " + text.length());
        System.out.println("Character at 0: " + text.charAt(0));
        System.out.println("Substring: " + text.substring(0, 5));
        System.out.println("Replace: " + text.replace("World", "Java"));
        
        // String comparison
        String str1 = "Hello";
        String str2 = "hello";
        System.out.println("Equals: " + str1.equals(str2));
        System.out.println("Equals ignore case: " + str1.equalsIgnoreCase(str2));
        
        // String splitting
        String sentence = "Java,is,awesome";
        String[] words = sentence.split(",");
        for (String word : words) {
            System.out.println("Word: " + word);
        }
        
        // StringBuilder for efficiency
        StringBuilder sb = new StringBuilder();
        sb.append("Hello");
        sb.append(", ");
        sb.append("World!");
        System.out.println("StringBuilder result: " + sb.toString());
    }
}
```

## Dataset 8: Exception Handling
```java
public class ExceptionHandling {
    public static void main(String[] args) {
        // Basic try-catch
        try {
            int result = 10 / 0;
            System.out.println("Result: " + result);
        } catch (ArithmeticException e) {
            System.out.println("Cannot divide by zero!");
        }
        
        // Multiple catch blocks
        try {
            String str = null;
            int number = Integer.parseInt("abc");
            System.out.println(str.length());
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format: " + e.getMessage());
        } catch (NullPointerException e) {
            System.out.println("Null pointer exception: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("General exception: " + e.getMessage());
        } finally {
            System.out.println("This always executes");
        }
        
        // Throwing exceptions
        try {
            validateAge(-5);
        } catch (IllegalArgumentException e) {
            System.out.println("Validation error: " + e.getMessage());
        }
    }
    
    public static void validateAge(int age) {
        if (age < 0) {
            throw new IllegalArgumentException("Age cannot be negative");
        }
        System.out.println("Valid age: " + age);
    }
}
```

## Dataset 9: Inheritance and Polymorphism
```java
// Base class
class Animal {
    protected String name;
    protected int age;
    
    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public void makeSound() {
        System.out.println(name + " makes a sound");
    }
    
    public void eat() {
        System.out.println(name + " is eating");
    }
}

// Derived class
class Dog extends Animal {
    private String breed;
    
    public Dog(String name, int age, String breed) {
        super(name, age);
        this.breed = breed;
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " barks: Woof!");
    }
    
    public void fetch() {
        System.out.println(name + " is fetching the ball");
    }
}

class Cat extends Animal {
    public Cat(String name, int age) {
        super(name, age);
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " meows: Meow!");
    }
}

public class InheritanceExample {
    public static void main(String[] args) {
        Animal[] animals = {
            new Dog("Buddy", 3, "Golden Retriever"),
            new Cat("Whiskers", 2),
            new Animal("Generic", 1)
        };
        
        for (Animal animal : animals) {
            animal.makeSound();
            animal.eat();
            
            // Polymorphism
            if (animal instanceof Dog) {
                ((Dog) animal).fetch();
            }
        }
    }
}
```

## Dataset 10: Input/Output and File Handling
```java
import java.io.*;
import java.util.Scanner;

public class InputOutput {
    public static void main(String[] args) {
        // Console input
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter your name: ");
        String name = scanner.nextLine();
        System.out.println("Hello, " + name + "!");
        
        System.out.print("Enter your age: ");
        int age = scanner.nextInt();
        System.out.println("You are " + age + " years old");
        
        // File writing
        try (PrintWriter writer = new PrintWriter(new FileWriter("output.txt"))) {
            writer.println("Hello, File!");
            writer.println("This is written from Java");
            System.out.println("File written successfully");
        } catch (IOException e) {
            System.out.println("Error writing file: " + e.getMessage());
        }
        
        // File reading
        try (BufferedReader reader = new BufferedReader(new FileReader("output.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("Read from file: " + line);
            }
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
        
        scanner.close();
    }
}
```