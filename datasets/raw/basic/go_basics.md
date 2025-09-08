# Basic Go Dataset - Concurrent Programming Fundamentals

## Dataset 1: Hello World and Package Basics
```go
package main

import "fmt"

// Simple Hello World
func main() {
    fmt.Println("Hello, World!")
}

// Hello World with functions
package main

import "fmt"

func greet() {
    fmt.Println("Hello, World!")
}

func greetWithName(name string) {
    fmt.Printf("Hello, %s!\n", name)
}

func main() {
    greet()
    greetWithName("Go")
}
```

## Dataset 2: Variables and Data Types
```go
package main

import "fmt"

func main() {
    // Variable declarations
    var message string = "Hello Go"
    var count int = 42
    var price float64 = 3.14
    var isActive bool = true
    
    // Short variable declaration
    name := "John"
    age := 30
    
    // Multiple variable declaration
    var (
        firstName = "Alice"
        lastName  = "Smith"
        id        = 12345
    )
    
    // Constants
    const Pi = 3.14159
    const Language = "Go"
    
    // Arrays and slices
    var numbers [5]int = [5]int{1, 2, 3, 4, 5}
    fruits := []string{"apple", "banana", "cherry"}
    
    // Maps
    person := map[string]interface{}{
        "name": "Bob",
        "age":  25,
    }
    
    // Output
    fmt.Println("Message:", message)
    fmt.Println("Count:", count)
    fmt.Println("Price:", price)
    fmt.Println("Is Active:", isActive)
    fmt.Println("Name:", name)
    fmt.Println("Age:", age)
    fmt.Printf("Full name: %s %s\n", firstName, lastName)
    fmt.Println("Numbers:", numbers)
    fmt.Println("Fruits:", fruits)
    fmt.Println("Person:", person)
}
```

## Dataset 3: Control Structures
```go
package main

import "fmt"

func main() {
    // If-else statements
    age := 18
    if age >= 18 {
        fmt.Println("Adult")
    } else if age >= 13 {
        fmt.Println("Teenager")
    } else {
        fmt.Println("Child")
    }
    
    // If with initialization
    if x := 10; x > 5 {
        fmt.Println("x is greater than 5")
    }
    
    // For loops (Go's only loop construct)
    for i := 0; i < 5; i++ {
        fmt.Printf("Number: %d\n", i)
    }
    
    // For as while loop
    count := 0
    for count < 5 {
        fmt.Printf("Count: %d\n", count)
        count++
    }
    
    // For range
    numbers := []int{1, 2, 3, 4, 5}
    for index, value := range numbers {
        fmt.Printf("Index: %d, Value: %d\n", index, value)
    }
    
    // Range over map
    person := map[string]int{"Alice": 25, "Bob": 30}
    for name, age := range person {
        fmt.Printf("%s is %d years old\n", name, age)
    }
    
    // Switch statement
    day := "Monday"
    switch day {
    case "Monday":
        fmt.Println("Start of the week")
    case "Friday":
        fmt.Println("TGIF!")
    default:
        fmt.Println("Regular day")
    }
    
    // Switch with expression
    switch time := 14; {
    case time < 12:
        fmt.Println("Good morning")
    case time < 18:
        fmt.Println("Good afternoon")
    default:
        fmt.Println("Good evening")
    }
}
```

## Dataset 4: Functions and Multiple Return Values
```go
package main

import "fmt"

// Basic function
func add(a, b int) int {
    return a + b
}

// Function with multiple return values
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Named return values
func getPersonInfo() (name string, age int) {
    name = "Alice"
    age = 30
    return // naked return
}

// Variadic function
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Higher-order function
func applyOperation(numbers []int, operation func(int) int) []int {
    result := make([]int, len(numbers))
    for i, num := range numbers {
        result[i] = operation(num)
    }
    return result
}

func main() {
    // Function calls
    result := add(5, 3)
    fmt.Println("Add result:", result)
    
    // Multiple return values
    quotient, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Division result:", quotient)
    }
    
    // Named returns
    name, age := getPersonInfo()
    fmt.Printf("Person: %s, Age: %d\n", name, age)
    
    // Variadic function
    total := sum(1, 2, 3, 4, 5)
    fmt.Println("Sum:", total)
    
    // Higher-order function with anonymous function
    numbers := []int{1, 2, 3, 4, 5}
    doubled := applyOperation(numbers, func(x int) int {
        return x * 2
    })
    fmt.Println("Doubled:", doubled)
}
```

## Dataset 5: Structs and Methods
```go
package main

import "fmt"

// Struct definition
type Person struct {
    Name string
    Age  int
    Email string
}

// Method with value receiver
func (p Person) Introduce() string {
    return fmt.Sprintf("Hi, I'm %s and I'm %d years old", p.Name, p.Age)
}

// Method with pointer receiver
func (p *Person) HaveBirthday() {
    p.Age++
    fmt.Printf("Happy birthday! Now I'm %d\n", p.Age)
}

// Embedded struct
type Employee struct {
    Person
    Department string
    Salary     float64
}

// Method for embedded struct
func (e Employee) GetInfo() string {
    return fmt.Sprintf("%s works in %s department", e.Name, e.Department)
}

func main() {
    // Creating structs
    person1 := Person{
        Name:  "Alice",
        Age:   25,
        Email: "alice@example.com",
    }
    
    person2 := Person{"Bob", 30, "bob@example.com"}
    
    var person3 Person
    person3.Name = "Charlie"
    person3.Age = 35
    
    // Using methods
    fmt.Println(person1.Introduce())
    person1.HaveBirthday()
    
    fmt.Println(person2.Introduce())
    
    // Embedded struct
    employee := Employee{
        Person: Person{
            Name:  "Diana",
            Age:   28,
            Email: "diana@company.com",
        },
        Department: "Engineering",
        Salary:     75000,
    }
    
    fmt.Println(employee.Introduce()) // Method from embedded struct
    fmt.Println(employee.GetInfo())   // Method from Employee
}
```

## Dataset 6: Interfaces
```go
package main

import "fmt"

// Interface definition
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Interface for displaying
type Stringer interface {
    String() string
}

// Rectangle struct
type Rectangle struct {
    Width  float64
    Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func (r Rectangle) String() string {
    return fmt.Sprintf("Rectangle(%.2f x %.2f)", r.Width, r.Height)
}

// Circle struct
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14159 * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * 3.14159 * c.Radius
}

func (c Circle) String() string {
    return fmt.Sprintf("Circle(radius: %.2f)", c.Radius)
}

// Function that accepts interface
func printShapeInfo(s Shape) {
    fmt.Printf("Shape: %v\n", s)
    fmt.Printf("Area: %.2f\n", s.Area())
    fmt.Printf("Perimeter: %.2f\n", s.Perimeter())
    
    // Type assertion
    if stringer, ok := s.(Stringer); ok {
        fmt.Printf("String representation: %s\n", stringer.String())
    }
    fmt.Println()
}

func main() {
    rectangle := Rectangle{Width: 10, Height: 5}
    circle := Circle{Radius: 7}
    
    // Interface usage
    shapes := []Shape{rectangle, circle}
    
    for _, shape := range shapes {
        printShapeInfo(shape)
    }
    
    // Empty interface
    var anything interface{}
    anything = 42
    fmt.Printf("Value: %v, Type: %T\n", anything, anything)
    
    anything = "Hello"
    fmt.Printf("Value: %v, Type: %T\n", anything, anything)
    
    // Type switch
    switch v := anything.(type) {
    case int:
        fmt.Printf("Integer: %d\n", v)
    case string:
        fmt.Printf("String: %s\n", v)
    default:
        fmt.Printf("Unknown type: %T\n", v)
    }
}
```

## Dataset 7: Slices and Maps
```go
package main

import "fmt"

func main() {
    // Slice operations
    numbers := []int{1, 2, 3, 4, 5}
    fmt.Println("Original slice:", numbers)
    
    // Append to slice
    numbers = append(numbers, 6, 7, 8)
    fmt.Println("After append:", numbers)
    
    // Slice of slice
    subset := numbers[2:5]
    fmt.Println("Subset [2:5]:", subset)
    
    // Make slice with capacity
    dynamicSlice := make([]int, 3, 10)
    fmt.Printf("Length: %d, Capacity: %d\n", len(dynamicSlice), cap(dynamicSlice))
    
    // Copy slices
    source := []int{1, 2, 3}
    destination := make([]int, len(source))
    copy(destination, source)
    fmt.Println("Copied slice:", destination)
    
    // 2D slice
    matrix := [][]int{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    }
    fmt.Println("Matrix:", matrix)
    
    // Map operations
    ages := map[string]int{
        "Alice": 25,
        "Bob":   30,
        "Carol": 35,
    }
    
    // Add to map
    ages["David"] = 40
    
    // Check if key exists
    if age, exists := ages["Alice"]; exists {
        fmt.Printf("Alice is %d years old\n", age)
    }
    
    // Delete from map
    delete(ages, "Bob")
    
    // Iterate over map
    for name, age := range ages {
        fmt.Printf("%s: %d\n", name, age)
    }
    
    // Make map
    scores := make(map[string]int)
    scores["test1"] = 95
    scores["test2"] = 87
    fmt.Println("Scores:", scores)
}
```

## Dataset 8: Error Handling
```go
package main

import (
    "errors"
    "fmt"
)

// Custom error type
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error in %s: %s", e.Field, e.Message)
}

// Function that returns an error
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Function with custom error
func validateAge(age int) error {
    if age < 0 {
        return ValidationError{
            Field:   "age",
            Message: "cannot be negative",
        }
    }
    if age > 150 {
        return ValidationError{
            Field:   "age",
            Message: "cannot be greater than 150",
        }
    }
    return nil
}

// Function with multiple potential errors
func processUser(name string, age int) error {
    if name == "" {
        return fmt.Errorf("name cannot be empty")
    }
    
    if err := validateAge(age); err != nil {
        return fmt.Errorf("age validation failed: %w", err)
    }
    
    fmt.Printf("User %s (%d years old) processed successfully\n", name, age)
    return nil
}

func main() {
    // Basic error handling
    result, err := divide(10, 2)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Division result: %.2f\n", result)
    }
    
    // Error with zero division
    _, err = divide(10, 0)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    }
    
    // Custom error handling
    err = validateAge(-5)
    if err != nil {
        fmt.Printf("Validation error: %v\n", err)
        
        // Type assertion for custom error
        if validationErr, ok := err.(ValidationError); ok {
            fmt.Printf("Field: %s, Message: %s\n", validationErr.Field, validationErr.Message)
        }
    }
    
    // Multiple error scenarios
    testCases := []struct {
        name string
        age  int
    }{
        {"Alice", 25},
        {"", 30},
        {"Bob", -5},
        {"Carol", 200},
    }
    
    for _, tc := range testCases {
        if err := processUser(tc.name, tc.age); err != nil {
            fmt.Printf("Processing failed: %v\n", err)
        }
    }
}
```

## Dataset 9: Goroutines and Channels
```go
package main

import (
    "fmt"
    "time"
)

// Simple goroutine function
func sayHello(name string) {
    for i := 0; i < 3; i++ {
        fmt.Printf("Hello from %s: %d\n", name, i)
        time.Sleep(100 * time.Millisecond)
    }
}

// Function that sends to channel
func sendNumbers(ch chan int) {
    for i := 1; i <= 5; i++ {
        ch <- i
        fmt.Printf("Sent: %d\n", i)
        time.Sleep(100 * time.Millisecond)
    }
    close(ch)
}

// Function that receives from channel
func receiveNumbers(ch chan int) {
    for number := range ch {
        fmt.Printf("Received: %d\n", number)
    }
    fmt.Println("Channel closed")
}

// Worker function for worker pool
func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, job)
        time.Sleep(100 * time.Millisecond)
        results <- job * 2
    }
}

func main() {
    // Basic goroutines
    fmt.Println("Starting goroutines:")
    go sayHello("Goroutine 1")
    go sayHello("Goroutine 2")
    
    // Wait for goroutines to complete
    time.Sleep(500 * time.Millisecond)
    
    // Channels
    fmt.Println("\nChannel communication:")
    ch := make(chan int)
    
    go sendNumbers(ch)
    go receiveNumbers(ch)
    
    time.Sleep(1 * time.Second)
    
    // Buffered channel
    fmt.Println("\nBuffered channel:")
    bufferedCh := make(chan string, 2)
    bufferedCh <- "First"
    bufferedCh <- "Second"
    
    fmt.Println(<-bufferedCh)
    fmt.Println(<-bufferedCh)
    
    // Select statement
    fmt.Println("\nSelect statement:")
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go func() {
        time.Sleep(200 * time.Millisecond)
        ch1 <- "Message from ch1"
    }()
    
    go func() {
        time.Sleep(100 * time.Millisecond)
        ch2 <- "Message from ch2"
    }()
    
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println("Received:", msg1)
        case msg2 := <-ch2:
            fmt.Println("Received:", msg2)
        case <-time.After(300 * time.Millisecond):
            fmt.Println("Timeout")
        }
    }
    
    // Worker pool pattern
    fmt.Println("\nWorker pool:")
    jobs := make(chan int, 10)
    results := make(chan int, 10)
    
    // Start workers
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }
    
    // Send jobs
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)
    
    // Collect results
    for r := 1; r <= 5; r++ {
        result := <-results
        fmt.Printf("Result: %d\n", result)
    }
}
```

## Dataset 10: File I/O and JSON
```go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "os"
)

// Struct for JSON example
type Person struct {
    Name  string `json:"name"`
    Age   int    `json:"age"`
    Email string `json:"email"`
}

func main() {
    // Writing to file
    file, err := os.Create("output.txt")
    if err != nil {
        fmt.Printf("Error creating file: %v\n", err)
        return
    }
    defer file.Close()
    
    _, err = file.WriteString("Hello, File!\n")
    if err != nil {
        fmt.Printf("Error writing to file: %v\n", err)
        return
    }
    
    _, err = file.WriteString("This is written from Go\n")
    if err != nil {
        fmt.Printf("Error writing to file: %v\n", err)
        return
    }
    
    fmt.Println("File written successfully")
    
    // Reading from file
    content, err := ioutil.ReadFile("output.txt")
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    
    fmt.Println("File content:")
    fmt.Print(string(content))
    
    // Reading file line by line
    file, err = os.Open("output.txt")
    if err != nil {
        fmt.Printf("Error opening file: %v\n", err)
        return
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    lineNumber := 1
    fmt.Println("\nReading line by line:")
    for scanner.Scan() {
        fmt.Printf("Line %d: %s\n", lineNumber, scanner.Text())
        lineNumber++
    }
    
    if err := scanner.Err(); err != nil {
        fmt.Printf("Error reading file: %v\n", err)
    }
    
    // JSON marshaling
    person := Person{
        Name:  "Alice",
        Age:   30,
        Email: "alice@example.com",
    }
    
    jsonData, err := json.Marshal(person)
    if err != nil {
        fmt.Printf("Error marshaling JSON: %v\n", err)
        return
    }
    
    fmt.Printf("\nJSON: %s\n", jsonData)
    
    // JSON unmarshaling
    jsonString := `{"name":"Bob","age":25,"email":"bob@example.com"}`
    var parsedPerson Person
    
    err = json.Unmarshal([]byte(jsonString), &parsedPerson)
    if err != nil {
        fmt.Printf("Error unmarshaling JSON: %v\n", err)
        return
    }
    
    fmt.Printf("Parsed person: %+v\n", parsedPerson)
    
    // Writing JSON to file
    people := []Person{
        {"Alice", 30, "alice@example.com"},
        {"Bob", 25, "bob@example.com"},
        {"Carol", 35, "carol@example.com"},
    }
    
    jsonFile, err := os.Create("people.json")
    if err != nil {
        fmt.Printf("Error creating JSON file: %v\n", err)
        return
    }
    defer jsonFile.Close()
    
    encoder := json.NewEncoder(jsonFile)
    encoder.SetIndent("", "  ")
    
    err = encoder.Encode(people)
    if err != nil {
        fmt.Printf("Error encoding JSON: %v\n", err)
        return
    }
    
    fmt.Println("JSON file written successfully")
}
```