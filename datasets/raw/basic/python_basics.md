# Basic Python Dataset - Hello World and Fundamentals

## Dataset 1: Hello World Variations
```python
# Simple hello world
print("Hello, World!")

# Hello world with variables
name = "World"
print(f"Hello, {name}!")

# Hello world with function
def greet(name="World"):
    return f"Hello, {name}!"

print(greet())
print(greet("Python"))

# Hello world with class
class Greeter:
    def __init__(self, greeting="Hello"):
        self.greeting = greeting
    
    def greet(self, name="World"):
        return f"{self.greeting}, {name}!"

greeter = Greeter()
print(greeter.greet())
```

## Dataset 2: Basic Variables and Types
```python
# Basic data types
integer_var = 42
float_var = 3.14
string_var = "Python"
boolean_var = True
list_var = [1, 2, 3, 4, 5]
dict_var = {"name": "John", "age": 30}
tuple_var = (1, 2, 3)
set_var = {1, 2, 3, 4, 5}

# Type checking
print(type(integer_var))
print(type(float_var))
print(type(string_var))
print(type(boolean_var))
print(type(list_var))
print(type(dict_var))
print(type(tuple_var))
print(type(set_var))
```

## Dataset 3: Basic Control Structures
```python
# If-else statements
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# For loops
for i in range(5):
    print(f"Number: {i}")

# While loops
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# List comprehension
squares = [x**2 for x in range(10)]
print(squares)
```

## Dataset 4: Basic Functions
```python
# Simple function
def add(a, b):
    return a + b

# Function with default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Function with variable arguments
def sum_all(*args):
    return sum(args)

# Function with keyword arguments
def create_profile(**kwargs):
    return kwargs

# Lambda functions
square = lambda x: x**2
print(square(5))

# Examples
print(add(3, 5))
print(greet("Alice"))
print(greet("Bob", "Hi"))
print(sum_all(1, 2, 3, 4, 5))
print(create_profile(name="John", age=30, city="New York"))
```

## Dataset 5: Basic Input/Output
```python
# Basic input/output
name = input("Enter your name: ")
print(f"Hello, {name}!")

# File reading
try:
    with open("sample.txt", "r") as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("File not found")

# File writing
with open("output.txt", "w") as file:
    file.write("Hello, File!")

# JSON handling
import json

data = {"name": "John", "age": 30}
json_string = json.dumps(data)
parsed_data = json.loads(json_string)
print(parsed_data)
```

## Dataset 6: Basic Error Handling
```python
# Try-except blocks
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("Invalid number format!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always executes")
```

## Dataset 7: Basic String Operations
```python
# String manipulation
text = "Hello, World!"
print(text.upper())
print(text.lower())
print(text.replace("World", "Python"))
print(text.split(","))

# String formatting
name = "Alice"
age = 25
print(f"Name: {name}, Age: {age}")
print("Name: {}, Age: {}".format(name, age))
print("Name: %s, Age: %d" % (name, age))

# String methods
email = "user@example.com"
print(email.startswith("user"))
print(email.endswith(".com"))
print(email.find("@"))
```

## Dataset 8: Basic List Operations
```python
# List creation and manipulation
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.insert(0, 0)
numbers.remove(3)
popped = numbers.pop()

# List methods
fruits = ["apple", "banana", "cherry"]
fruits.sort()
fruits.reverse()

# List slicing
print(numbers[1:4])
print(numbers[:3])
print(numbers[2:])
print(numbers[-1])

# List iteration
for fruit in fruits:
    print(fruit)

for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
```

## Dataset 9: Basic Dictionary Operations
```python
# Dictionary creation and manipulation
person = {"name": "John", "age": 30, "city": "New York"}
person["occupation"] = "Developer"
person.update({"salary": 75000})

# Dictionary methods
print(person.keys())
print(person.values())
print(person.items())

# Dictionary iteration
for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
print(squares)
```

## Dataset 10: Basic Object-Oriented Programming
```python
# Simple class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! Now I'm {self.age}"

# Class inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def study(self, subject):
        return f"{self.name} is studying {subject}"

# Using classes
person = Person("Alice", 25)
print(person.introduce())
print(person.have_birthday())

student = Student("Bob", 20, "S12345")
print(student.introduce())
print(student.study("Python"))
```