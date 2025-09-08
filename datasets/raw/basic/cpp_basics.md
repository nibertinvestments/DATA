# Basic C++ Dataset - System Programming Fundamentals

## Dataset 1: Hello World and Basic Structure
```cpp
#include <iostream>
#include <string>

// Simple Hello World
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

// Hello World with functions
#include <iostream>

void greet() {
    std::cout << "Hello, World!" << std::endl;
}

void greet(const std::string& name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

int main() {
    greet();
    greet("C++");
    return 0;
}
```

## Dataset 2: Variables and Data Types
```cpp
#include <iostream>
#include <string>
#include <vector>

int main() {
    // Fundamental types
    int intVar = 42;
    float floatVar = 3.14f;
    double doubleVar = 3.14159;
    char charVar = 'A';
    bool boolVar = true;
    
    // String type
    std::string stringVar = "Hello C++";
    
    // Arrays
    int array[5] = {1, 2, 3, 4, 5};
    
    // Vectors (dynamic arrays)
    std::vector<int> vector = {1, 2, 3, 4, 5};
    
    // Output
    std::cout << "Integer: " << intVar << std::endl;
    std::cout << "Float: " << floatVar << std::endl;
    std::cout << "Double: " << doubleVar << std::endl;
    std::cout << "Character: " << charVar << std::endl;
    std::cout << "Boolean: " << boolVar << std::endl;
    std::cout << "String: " << stringVar << std::endl;
    std::cout << "Array size: " << sizeof(array)/sizeof(array[0]) << std::endl;
    std::cout << "Vector size: " << vector.size() << std::endl;
    
    return 0;
}
```

## Dataset 3: Control Structures
```cpp
#include <iostream>

int main() {
    // If-else statements
    int age = 18;
    if (age >= 18) {
        std::cout << "Adult" << std::endl;
    } else if (age >= 13) {
        std::cout << "Teenager" << std::endl;
    } else {
        std::cout << "Child" << std::endl;
    }
    
    // For loops
    for (int i = 0; i < 5; i++) {
        std::cout << "Number: " << i << std::endl;
    }
    
    // Range-based for loop (C++11)
    int numbers[] = {1, 2, 3, 4, 5};
    for (int number : numbers) {
        std::cout << "Range-based for: " << number << std::endl;
    }
    
    // While loop
    int count = 0;
    while (count < 5) {
        std::cout << "Count: " << count << std::endl;
        count++;
    }
    
    // Switch statement
    int choice = 1;
    switch (choice) {
        case 1:
            std::cout << "Choice one" << std::endl;
            break;
        case 2:
            std::cout << "Choice two" << std::endl;
            break;
        default:
            std::cout << "Default choice" << std::endl;
    }
    
    return 0;
}
```

## Dataset 4: Functions and Function Overloading
```cpp
#include <iostream>

// Function declaration
int add(int a, int b);
void greet(const std::string& name = "World");

// Function overloading
int multiply(int a, int b);
double multiply(double a, double b);

// Template function
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    // Function calls
    int sum = add(5, 3);
    std::cout << "Sum: " << sum << std::endl;
    
    greet();
    greet("Alice");
    
    // Overloaded functions
    std::cout << "Int multiply: " << multiply(3, 4) << std::endl;
    std::cout << "Double multiply: " << multiply(3.5, 2.0) << std::endl;
    
    // Template function
    std::cout << "Max int: " << maximum(10, 20) << std::endl;
    std::cout << "Max double: " << maximum(3.14, 2.71) << std::endl;
    
    return 0;
}

// Function definitions
int add(int a, int b) {
    return a + b;
}

void greet(const std::string& name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

int multiply(int a, int b) {
    return a * b;
}

double multiply(double a, double b) {
    return a * b;
}
```

## Dataset 5: Classes and Objects
```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;
    
public:
    // Constructor
    Person(const std::string& name, int age) : name(name), age(age) {}
    
    // Default constructor
    Person() : name("Unknown"), age(0) {}
    
    // Destructor
    ~Person() {
        std::cout << "Person " << name << " destroyed" << std::endl;
    }
    
    // Getter methods
    std::string getName() const { return name; }
    int getAge() const { return age; }
    
    // Setter methods
    void setName(const std::string& newName) { name = newName; }
    void setAge(int newAge) { 
        if (newAge >= 0) age = newAge; 
    }
    
    // Instance methods
    void introduce() const {
        std::cout << "Hi, I'm " << name << " and I'm " << age << " years old" << std::endl;
    }
    
    void haveBirthday() {
        age++;
        std::cout << "Happy birthday! Now I'm " << age << std::endl;
    }
};

int main() {
    // Creating objects
    Person person1("Alice", 25);
    Person person2;
    
    // Using objects
    person1.introduce();
    person1.haveBirthday();
    
    person2.setName("Bob");
    person2.setAge(30);
    person2.introduce();
    
    return 0;
}
```

## Dataset 6: Arrays, Vectors, and STL Containers
```cpp
#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <algorithm>

int main() {
    // C-style arrays
    int cArray[5] = {1, 2, 3, 4, 5};
    
    // std::array (C++11)
    std::array<int, 5> stdArray = {1, 2, 3, 4, 5};
    
    // std::vector
    std::vector<int> vector = {1, 2, 3, 4, 5};
    vector.push_back(6);
    vector.pop_back();
    
    // Iterating through containers
    std::cout << "C-array: ";
    for (int i = 0; i < 5; i++) {
        std::cout << cArray[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "std::array: ";
    for (const auto& element : stdArray) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Vector: ";
    for (const auto& element : vector) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // std::map
    std::map<std::string, int> fruits;
    fruits["apple"] = 5;
    fruits["banana"] = 3;
    fruits["cherry"] = 8;
    
    std::cout << "Map contents:" << std::endl;
    for (const auto& pair : fruits) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // std::set
    std::set<int> uniqueNumbers = {3, 1, 4, 1, 5, 9, 2, 6};
    std::cout << "Set contents: ";
    for (const auto& number : uniqueNumbers) {
        std::cout << number << " ";
    }
    std::cout << std::endl;
    
    // STL algorithms
    std::sort(vector.begin(), vector.end());
    auto it = std::find(vector.begin(), vector.end(), 3);
    if (it != vector.end()) {
        std::cout << "Found 3 at position: " << std::distance(vector.begin(), it) << std::endl;
    }
    
    return 0;
}
```

## Dataset 7: Pointers and References
```cpp
#include <iostream>

int main() {
    // Basic variables
    int number = 42;
    
    // Pointers
    int* ptr = &number;
    std::cout << "Value: " << number << std::endl;
    std::cout << "Address: " << &number << std::endl;
    std::cout << "Pointer value: " << ptr << std::endl;
    std::cout << "Dereferenced pointer: " << *ptr << std::endl;
    
    // Modifying through pointer
    *ptr = 100;
    std::cout << "Modified value: " << number << std::endl;
    
    // References
    int& ref = number;
    std::cout << "Reference value: " << ref << std::endl;
    ref = 200;
    std::cout << "Modified through reference: " << number << std::endl;
    
    // Dynamic memory allocation
    int* dynamicArray = new int[5];
    for (int i = 0; i < 5; i++) {
        dynamicArray[i] = i + 1;
    }
    
    std::cout << "Dynamic array: ";
    for (int i = 0; i < 5; i++) {
        std::cout << dynamicArray[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up dynamic memory
    delete[] dynamicArray;
    
    return 0;
}
```

## Dataset 8: String Manipulation
```cpp
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

int main() {
    std::string text = "Hello, World!";
    
    // String methods
    std::cout << "Original: " << text << std::endl;
    std::cout << "Length: " << text.length() << std::endl;
    std::cout << "Substring: " << text.substr(0, 5) << std::endl;
    std::cout << "Find 'World': " << text.find("World") << std::endl;
    
    // String modification
    std::string modifiedText = text;
    std::replace(modifiedText.begin(), modifiedText.end(), 'o', '0');
    std::cout << "Modified: " << modifiedText << std::endl;
    
    // String concatenation
    std::string firstName = "John";
    std::string lastName = "Doe";
    std::string fullName = firstName + " " + lastName;
    std::cout << "Full name: " << fullName << std::endl;
    
    // String stream
    std::stringstream ss;
    ss << "Number: " << 42 << ", Pi: " << 3.14;
    std::cout << "String stream: " << ss.str() << std::endl;
    
    // C-string interaction
    const char* cString = text.c_str();
    std::cout << "C-string: " << cString << std::endl;
    
    // Case conversion (using algorithm)
    std::string upperText = text;
    std::transform(upperText.begin(), upperText.end(), upperText.begin(), ::toupper);
    std::cout << "Uppercase: " << upperText << std::endl;
    
    return 0;
}
```

## Dataset 9: File I/O
```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Writing to file
    std::ofstream outFile("output.txt");
    if (outFile.is_open()) {
        outFile << "Hello, File!" << std::endl;
        outFile << "This is written from C++" << std::endl;
        outFile << "Number: " << 42 << std::endl;
        outFile.close();
        std::cout << "File written successfully" << std::endl;
    } else {
        std::cout << "Unable to open file for writing" << std::endl;
    }
    
    // Reading from file
    std::ifstream inFile("output.txt");
    if (inFile.is_open()) {
        std::string line;
        std::cout << "Reading from file:" << std::endl;
        while (std::getline(inFile, line)) {
            std::cout << line << std::endl;
        }
        inFile.close();
    } else {
        std::cout << "Unable to open file for reading" << std::endl;
    }
    
    // Binary file operations
    std::ofstream binaryOut("data.bin", std::ios::binary);
    if (binaryOut.is_open()) {
        int numbers[] = {1, 2, 3, 4, 5};
        binaryOut.write(reinterpret_cast<char*>(numbers), sizeof(numbers));
        binaryOut.close();
        std::cout << "Binary file written" << std::endl;
    }
    
    std::ifstream binaryIn("data.bin", std::ios::binary);
    if (binaryIn.is_open()) {
        int readNumbers[5];
        binaryIn.read(reinterpret_cast<char*>(readNumbers), sizeof(readNumbers));
        binaryIn.close();
        
        std::cout << "Read from binary file: ";
        for (int number : readNumbers) {
            std::cout << number << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
```

## Dataset 10: Exception Handling and RAII
```cpp
#include <iostream>
#include <stdexcept>
#include <memory>

// Custom exception class
class CustomException : public std::exception {
private:
    std::string message;
    
public:
    CustomException(const std::string& msg) : message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
};

// RAII example class
class Resource {
private:
    std::string name;
    
public:
    Resource(const std::string& name) : name(name) {
        std::cout << "Resource " << name << " acquired" << std::endl;
    }
    
    ~Resource() {
        std::cout << "Resource " << name << " released" << std::endl;
    }
    
    void use() {
        std::cout << "Using resource " << name << std::endl;
    }
};

void riskyFunction(int value) {
    if (value < 0) {
        throw std::invalid_argument("Value cannot be negative");
    }
    if (value > 100) {
        throw CustomException("Value too large");
    }
    std::cout << "Value is valid: " << value << std::endl;
}

int main() {
    // Exception handling
    try {
        riskyFunction(50);
        riskyFunction(-5);
    } catch (const std::invalid_argument& e) {
        std::cout << "Invalid argument: " << e.what() << std::endl;
    } catch (const CustomException& e) {
        std::cout << "Custom exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
    }
    
    // RAII with automatic cleanup
    {
        Resource resource("TestResource");
        resource.use();
        // Resource automatically destroyed at end of scope
    }
    
    // Smart pointers (RAII for dynamic memory)
    {
        std::unique_ptr<Resource> smartRes = std::make_unique<Resource>("SmartResource");
        smartRes->use();
        // Memory automatically freed when smartRes goes out of scope
    }
    
    return 0;
}
```