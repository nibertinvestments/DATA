# Basic C# Dataset - Object-Oriented .NET Programming

## Dataset 1: Hello World and Basic Structure
```csharp
using System;

// Simple Hello World
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}

// Hello World with methods
using System;

class Greeting
{
    static void Main(string[] args)
    {
        SayHello();
        SayHello("C#");
    }
    
    static void SayHello()
    {
        Console.WriteLine("Hello, World!");
    }
    
    static void SayHello(string name)
    {
        Console.WriteLine($"Hello, {name}!");
    }
}
```

## Dataset 2: Variables and Data Types
```csharp
using System;
using System.Collections.Generic;

class DataTypes
{
    static void Main(string[] args)
    {
        // Value types
        int intVar = 42;
        double doubleVar = 3.14;
        float floatVar = 3.14f;
        decimal decimalVar = 99.99m;
        char charVar = 'A';
        bool boolVar = true;
        
        // Reference types
        string stringVar = "Hello C#";
        int[] arrayVar = {1, 2, 3, 4, 5};
        List<string> listVar = new List<string> {"apple", "banana", "cherry"};
        
        // Nullable types
        int? nullableInt = null;
        
        // var keyword (type inference)
        var inferredString = "This is a string";
        var inferredNumber = 100;
        
        // Output
        Console.WriteLine($"Integer: {intVar}");
        Console.WriteLine($"Double: {doubleVar}");
        Console.WriteLine($"Float: {floatVar}");
        Console.WriteLine($"Decimal: {decimalVar}");
        Console.WriteLine($"Character: {charVar}");
        Console.WriteLine($"Boolean: {boolVar}");
        Console.WriteLine($"String: {stringVar}");
        Console.WriteLine($"Array length: {arrayVar.Length}");
        Console.WriteLine($"List count: {listVar.Count}");
        Console.WriteLine($"Nullable int: {nullableInt ?? -1}");
        Console.WriteLine($"Inferred string: {inferredString}");
        Console.WriteLine($"Inferred number: {inferredNumber}");
    }
}
```

## Dataset 3: Control Structures
```csharp
using System;

class ControlStructures
{
    static void Main(string[] args)
    {
        // If-else statements
        int age = 18;
        if (age >= 18)
        {
            Console.WriteLine("Adult");
        }
        else if (age >= 13)
        {
            Console.WriteLine("Teenager");
        }
        else
        {
            Console.WriteLine("Child");
        }
        
        // Ternary operator
        string status = age >= 18 ? "adult" : "minor";
        Console.WriteLine($"Status: {status}");
        
        // For loops
        for (int i = 0; i < 5; i++)
        {
            Console.WriteLine($"Number: {i}");
        }
        
        // Foreach loop
        int[] numbers = {1, 2, 3, 4, 5};
        foreach (int number in numbers)
        {
            Console.WriteLine($"Foreach: {number}");
        }
        
        // While loop
        int count = 0;
        while (count < 5)
        {
            Console.WriteLine($"Count: {count}");
            count++;
        }
        
        // Do-while loop
        int doCount = 0;
        do
        {
            Console.WriteLine($"Do-while: {doCount}");
            doCount++;
        } while (doCount < 3);
        
        // Switch statement
        string day = "Monday";
        switch (day)
        {
            case "Monday":
                Console.WriteLine("Start of the week");
                break;
            case "Friday":
                Console.WriteLine("TGIF!");
                break;
            default:
                Console.WriteLine("Regular day");
                break;
        }
        
        // Switch expression (C# 8.0+)
        string dayMessage = day switch
        {
            "Monday" => "Start of the week",
            "Friday" => "TGIF!",
            _ => "Regular day"
        };
        Console.WriteLine($"Switch expression: {dayMessage}");
    }
}
```

## Dataset 4: Methods and Parameters
```csharp
using System;

class Methods
{
    static void Main(string[] args)
    {
        // Method calls
        int sum = Add(5, 3);
        Console.WriteLine($"Sum: {sum}");
        
        Greet("Alice");
        
        // Method with out parameter
        int quotient, remainder;
        Divide(10, 3, out quotient, out remainder);
        Console.WriteLine($"Quotient: {quotient}, Remainder: {remainder}");
        
        // Method with ref parameter
        int number = 10;
        ModifyValue(ref number);
        Console.WriteLine($"Modified value: {number}");
        
        // Optional parameters
        PrintMessage("Hello");
        PrintMessage("Hello", "World");
        
        // Named parameters
        CreateUser(name: "Bob", age: 30, email: "bob@example.com");
        
        // Method overloading
        Console.WriteLine($"Add integers: {Add(1, 2)}");
        Console.WriteLine($"Add doubles: {Add(1.5, 2.5)}");
        
        // Params keyword
        int total = Sum(1, 2, 3, 4, 5);
        Console.WriteLine($"Total: {total}");
    }
    
    static int Add(int a, int b)
    {
        return a + b;
    }
    
    static double Add(double a, double b)
    {
        return a + b;
    }
    
    static void Greet(string name)
    {
        Console.WriteLine($"Hello, {name}!");
    }
    
    static void Divide(int dividend, int divisor, out int quotient, out int remainder)
    {
        quotient = dividend / divisor;
        remainder = dividend % divisor;
    }
    
    static void ModifyValue(ref int value)
    {
        value *= 2;
    }
    
    static void PrintMessage(string message, string suffix = "!")
    {
        Console.WriteLine($"{message}{suffix}");
    }
    
    static void CreateUser(string name, int age, string email)
    {
        Console.WriteLine($"User: {name}, Age: {age}, Email: {email}");
    }
    
    static int Sum(params int[] numbers)
    {
        int total = 0;
        foreach (int number in numbers)
        {
            total += number;
        }
        return total;
    }
}
```

## Dataset 5: Classes and Objects
```csharp
using System;

// Basic class
public class Person
{
    // Fields
    private string name;
    private int age;
    
    // Properties
    public string Name
    {
        get { return name; }
        set { name = value; }
    }
    
    public int Age
    {
        get { return age; }
        set { age = value >= 0 ? value : 0; }
    }
    
    // Auto-implemented property
    public string Email { get; set; }
    
    // Constructor
    public Person(string name, int age)
    {
        this.name = name;
        this.age = age;
    }
    
    // Default constructor
    public Person() : this("Unknown", 0)
    {
    }
    
    // Methods
    public void Introduce()
    {
        Console.WriteLine($"Hi, I'm {name} and I'm {age} years old");
    }
    
    public void HaveBirthday()
    {
        age++;
        Console.WriteLine($"Happy birthday! Now I'm {age}");
    }
    
    // Static method
    public static void ShowSpecies()
    {
        Console.WriteLine("Human");
    }
}

// Inheritance
public class Student : Person
{
    public string StudentId { get; set; }
    public List<string> Courses { get; set; }
    
    public Student(string name, int age, string studentId) : base(name, age)
    {
        StudentId = studentId;
        Courses = new List<string>();
    }
    
    public void AddCourse(string course)
    {
        Courses.Add(course);
    }
    
    // Override method
    public new void Introduce()
    {
        base.Introduce();
        Console.WriteLine($"My student ID is {StudentId}");
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Creating objects
        Person person1 = new Person("Alice", 25);
        person1.Email = "alice@example.com";
        
        Person person2 = new Person();
        person2.Name = "Bob";
        person2.Age = 30;
        
        // Using methods
        person1.Introduce();
        person1.HaveBirthday();
        
        person2.Introduce();
        
        // Static method
        Person.ShowSpecies();
        
        // Inheritance
        Student student = new Student("Carol", 20, "S12345");
        student.AddCourse("C# Programming");
        student.AddCourse("Data Structures");
        student.Introduce();
        
        Console.WriteLine($"Courses: {string.Join(", ", student.Courses)}");
    }
}
```

## Dataset 6: Collections and LINQ
```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Collections
{
    static void Main(string[] args)
    {
        // List<T>
        List<int> numbers = new List<int> {1, 2, 3, 4, 5};
        numbers.Add(6);
        numbers.AddRange(new[] {7, 8, 9});
        
        Console.WriteLine("Numbers:");
        foreach (int number in numbers)
        {
            Console.Write($"{number} ");
        }
        Console.WriteLine();
        
        // Dictionary<TKey, TValue>
        Dictionary<string, int> ages = new Dictionary<string, int>
        {
            {"Alice", 25},
            {"Bob", 30},
            {"Carol", 35}
        };
        
        ages["David"] = 40;
        
        Console.WriteLine("Ages:");
        foreach (KeyValuePair<string, int> kvp in ages)
        {
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");
        }
        
        // LINQ queries
        var evenNumbers = numbers.Where(n => n % 2 == 0);
        Console.WriteLine($"Even numbers: {string.Join(", ", evenNumbers)}");
        
        var doubledNumbers = numbers.Select(n => n * 2);
        Console.WriteLine($"Doubled numbers: {string.Join(", ", doubledNumbers)}");
        
        var sum = numbers.Sum();
        Console.WriteLine($"Sum: {sum}");
        
        var average = numbers.Average();
        Console.WriteLine($"Average: {average:F2}");
        
        var orderedNumbers = numbers.OrderByDescending(n => n);
        Console.WriteLine($"Descending order: {string.Join(", ", orderedNumbers)}");
        
        // Complex LINQ query
        var people = new[]
        {
            new { Name = "Alice", Age = 25, City = "New York" },
            new { Name = "Bob", Age = 30, City = "London" },
            new { Name = "Carol", Age = 35, City = "New York" },
            new { Name = "David", Age = 40, City = "Paris" }
        };
        
        var newYorkers = people
            .Where(p => p.City == "New York")
            .OrderBy(p => p.Age)
            .Select(p => new { p.Name, p.Age });
        
        Console.WriteLine("New Yorkers:");
        foreach (var person in newYorkers)
        {
            Console.WriteLine($"{person.Name} ({person.Age})");
        }
        
        // Method chaining
        var result = numbers
            .Where(n => n > 3)
            .Select(n => n * 2)
            .OrderByDescending(n => n)
            .Take(3);
        
        Console.WriteLine($"Filtered and transformed: {string.Join(", ", result)}");
    }
}
```

## Dataset 7: String Manipulation
```csharp
using System;
using System.Text;

class StringManipulation
{
    static void Main(string[] args)
    {
        string text = "Hello, World!";
        
        // String methods
        Console.WriteLine($"Original: {text}");
        Console.WriteLine($"Uppercase: {text.ToUpper()}");
        Console.WriteLine($"Lowercase: {text.ToLower()}");
        Console.WriteLine($"Length: {text.Length}");
        Console.WriteLine($"Substring: {text.Substring(0, 5)}");
        Console.WriteLine($"Replace: {text.Replace("World", "C#")}");
        Console.WriteLine($"Contains: {text.Contains("World")}");
        Console.WriteLine($"Starts with: {text.StartsWith("Hello")}");
        Console.WriteLine($"Ends with: {text.EndsWith("!")}");
        
        // String interpolation
        string name = "Alice";
        int age = 25;
        Console.WriteLine($"Name: {name}, Age: {age}");
        
        // String formatting
        Console.WriteLine("Name: {0}, Age: {1}", name, age);
        Console.WriteLine(string.Format("Name: {0}, Age: {1}", name, age));
        
        // String splitting and joining
        string sentence = "The,quick,brown,fox";
        string[] words = sentence.Split(',');
        Console.WriteLine("Words:");
        foreach (string word in words)
        {
            Console.WriteLine($"  {word}");
        }
        
        string joined = string.Join(" ", words);
        Console.WriteLine($"Joined: {joined}");
        
        // StringBuilder for efficient string building
        StringBuilder sb = new StringBuilder();
        sb.Append("Hello");
        sb.Append(", ");
        sb.Append("World!");
        sb.AppendLine();
        sb.AppendFormat("Name: {0}, Age: {1}", name, age);
        
        Console.WriteLine("StringBuilder result:");
        Console.WriteLine(sb.ToString());
        
        // String comparison
        string str1 = "Hello";
        string str2 = "hello";
        Console.WriteLine($"Equals: {str1.Equals(str2)}");
        Console.WriteLine($"Equals ignore case: {str1.Equals(str2, StringComparison.OrdinalIgnoreCase)}");
        Console.WriteLine($"Compare: {string.Compare(str1, str2)}");
        
        // Escape sequences
        string escaped = "Line 1\nLine 2\tTabbed\n\"Quoted\"";
        Console.WriteLine("Escaped string:");
        Console.WriteLine(escaped);
        
        // Verbatim strings
        string verbatim = @"C:\Users\Alice\Documents\file.txt";
        Console.WriteLine($"Verbatim string: {verbatim}");
    }
}
```

## Dataset 8: Exception Handling
```csharp
using System;

// Custom exception
public class ValidationException : Exception
{
    public ValidationException(string message) : base(message) { }
    public ValidationException(string message, Exception innerException) : base(message, innerException) { }
}

class ExceptionHandling
{
    static void Main(string[] args)
    {
        // Basic try-catch
        try
        {
            int result = 10 / 0;
            Console.WriteLine($"Result: {result}");
        }
        catch (DivideByZeroException ex)
        {
            Console.WriteLine($"Division by zero error: {ex.Message}");
        }
        
        // Multiple catch blocks
        try
        {
            string input = null;
            int number = int.Parse("abc");
            Console.WriteLine(input.Length);
        }
        catch (FormatException ex)
        {
            Console.WriteLine($"Format error: {ex.Message}");
        }
        catch (ArgumentNullException ex)
        {
            Console.WriteLine($"Null argument error: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"General error: {ex.Message}");
        }
        finally
        {
            Console.WriteLine("This always executes");
        }
        
        // Throwing exceptions
        try
        {
            ValidateAge(-5);
        }
        catch (ValidationException ex)
        {
            Console.WriteLine($"Validation error: {ex.Message}");
        }
        
        // Using statement for automatic resource disposal
        try
        {
            using (var file = new System.IO.StringWriter())
            {
                file.WriteLine("Hello, file!");
                Console.WriteLine("File operation completed");
            } // file.Dispose() called automatically
        }
        catch (Exception ex)
        {
            Console.WriteLine($"File error: {ex.Message}");
        }
        
        // TryParse pattern
        string numberString = "123";
        if (int.TryParse(numberString, out int parsedNumber))
        {
            Console.WriteLine($"Parsed successfully: {parsedNumber}");
        }
        else
        {
            Console.WriteLine("Failed to parse");
        }
    }
    
    static void ValidateAge(int age)
    {
        if (age < 0)
        {
            throw new ValidationException("Age cannot be negative");
        }
        Console.WriteLine($"Valid age: {age}");
    }
}
```

## Dataset 9: Interfaces and Abstract Classes
```csharp
using System;

// Interface
public interface IShape
{
    double Area { get; }
    double Perimeter { get; }
    void Draw();
}

// Another interface
public interface IMovable
{
    void Move(int x, int y);
}

// Abstract class
public abstract class Shape : IShape
{
    public abstract double Area { get; }
    public abstract double Perimeter { get; }
    
    public virtual void Draw()
    {
        Console.WriteLine($"Drawing a {GetType().Name}");
    }
    
    public void ShowInfo()
    {
        Console.WriteLine($"Area: {Area:F2}, Perimeter: {Perimeter:F2}");
    }
}

// Concrete class implementing interface and inheriting from abstract class
public class Rectangle : Shape, IMovable
{
    public double Width { get; set; }
    public double Height { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    
    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }
    
    public override double Area => Width * Height;
    public override double Perimeter => 2 * (Width + Height);
    
    public override void Draw()
    {
        base.Draw();
        Console.WriteLine($"Rectangle at ({X}, {Y}) with dimensions {Width}x{Height}");
    }
    
    public void Move(int x, int y)
    {
        X = x;
        Y = y;
        Console.WriteLine($"Rectangle moved to ({X}, {Y})");
    }
}

public class Circle : Shape, IMovable
{
    public double Radius { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    
    public Circle(double radius)
    {
        Radius = radius;
    }
    
    public override double Area => Math.PI * Radius * Radius;
    public override double Perimeter => 2 * Math.PI * Radius;
    
    public override void Draw()
    {
        Console.WriteLine($"Drawing a circle at ({X}, {Y}) with radius {Radius}");
    }
    
    public void Move(int x, int y)
    {
        X = x;
        Y = y;
        Console.WriteLine($"Circle moved to ({X}, {Y})");
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Polymorphism with interfaces and abstract classes
        IShape[] shapes = {
            new Rectangle(10, 5),
            new Circle(7)
        };
        
        foreach (IShape shape in shapes)
        {
            shape.Draw();
            shape.ShowInfo();
            
            // Type checking and casting
            if (shape is IMovable movable)
            {
                movable.Move(10, 20);
            }
            
            Console.WriteLine();
        }
        
        // Using interface references
        IMovable movableObject = new Rectangle(5, 3);
        movableObject.Move(100, 200);
        
        // Pattern matching (C# 7.0+)
        foreach (IShape shape in shapes)
        {
            switch (shape)
            {
                case Rectangle rect:
                    Console.WriteLine($"Rectangle with area {rect.Area}");
                    break;
                case Circle circle:
                    Console.WriteLine($"Circle with radius {circle.Radius}");
                    break;
                default:
                    Console.WriteLine("Unknown shape");
                    break;
            }
        }
    }
}
```

## Dataset 10: File I/O and Async Programming
```csharp
using System;
using System.IO;
using System.Threading.Tasks;

class FileIOAndAsync
{
    static async Task Main(string[] args)
    {
        // Synchronous file operations
        Console.WriteLine("=== Synchronous File I/O ===");
        
        string fileName = "sample.txt";
        string content = "Hello, File!\nThis is written from C#\nLine 3";
        
        // Writing to file
        File.WriteAllText(fileName, content);
        Console.WriteLine("File written successfully");
        
        // Reading from file
        string readContent = File.ReadAllText(fileName);
        Console.WriteLine("File content:");
        Console.WriteLine(readContent);
        
        // Reading lines
        string[] lines = File.ReadAllLines(fileName);
        Console.WriteLine("\nReading line by line:");
        for (int i = 0; i < lines.Length; i++)
        {
            Console.WriteLine($"Line {i + 1}: {lines[i]}");
        }
        
        // Using StreamWriter and StreamReader
        using (StreamWriter writer = new StreamWriter("stream_output.txt"))
        {
            writer.WriteLine("Line 1");
            writer.WriteLine("Line 2");
            writer.WriteLine($"Current time: {DateTime.Now}");
        }
        
        using (StreamReader reader = new StreamReader("stream_output.txt"))
        {
            string line;
            Console.WriteLine("\nStream reader output:");
            while ((line = reader.ReadLine()) != null)
            {
                Console.WriteLine(line);
            }
        }
        
        // Asynchronous file operations
        Console.WriteLine("\n=== Asynchronous File I/O ===");
        
        await WriteFileAsync("async_file.txt", "This is async content");
        string asyncContent = await ReadFileAsync("async_file.txt");
        Console.WriteLine($"Async file content: {asyncContent}");
        
        // Asynchronous method demonstration
        Console.WriteLine("\n=== Async Methods ===");
        
        Task<string> task1 = GetDataAsync("Task 1", 1000);
        Task<string> task2 = GetDataAsync("Task 2", 2000);
        Task<string> task3 = GetDataAsync("Task 3", 1500);
        
        // Wait for all tasks to complete
        string[] results = await Task.WhenAll(task1, task2, task3);
        
        Console.WriteLine("All tasks completed:");
        foreach (string result in results)
        {
            Console.WriteLine(result);
        }
        
        // Sequential vs Parallel execution
        Console.WriteLine("\n=== Sequential vs Parallel ===");
        
        var watch = System.Diagnostics.Stopwatch.StartNew();
        
        // Sequential
        await GetDataAsync("Sequential 1", 500);
        await GetDataAsync("Sequential 2", 500);
        await GetDataAsync("Sequential 3", 500);
        
        watch.Stop();
        Console.WriteLine($"Sequential execution time: {watch.ElapsedMilliseconds}ms");
        
        watch.Restart();
        
        // Parallel
        await Task.WhenAll(
            GetDataAsync("Parallel 1", 500),
            GetDataAsync("Parallel 2", 500),
            GetDataAsync("Parallel 3", 500)
        );
        
        watch.Stop();
        Console.WriteLine($"Parallel execution time: {watch.ElapsedMilliseconds}ms");
    }
    
    static async Task WriteFileAsync(string fileName, string content)
    {
        using (StreamWriter writer = new StreamWriter(fileName))
        {
            await writer.WriteAsync(content);
        }
        Console.WriteLine($"Async write to {fileName} completed");
    }
    
    static async Task<string> ReadFileAsync(string fileName)
    {
        using (StreamReader reader = new StreamReader(fileName))
        {
            string content = await reader.ReadToEndAsync();
            Console.WriteLine($"Async read from {fileName} completed");
            return content;
        }
    }
    
    static async Task<string> GetDataAsync(string taskName, int delayMs)
    {
        Console.WriteLine($"{taskName} started");
        await Task.Delay(delayMs); // Simulate async work
        string result = $"{taskName} completed after {delayMs}ms";
        Console.WriteLine(result);
        return result;
    }
}
```