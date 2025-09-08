# Basic JavaScript Dataset - Fundamentals and Core Concepts

## Dataset 1: Hello World and Basic Output
```javascript
// Simple hello world
console.log("Hello, World!");

// Hello world with variables
const name = "World";
console.log(`Hello, ${name}!`);

// Hello world with function
function greet(name = "World") {
    return `Hello, ${name}!`;
}

console.log(greet());
console.log(greet("JavaScript"));

// Arrow function version
const greetArrow = (name = "World") => `Hello, ${name}!`;
console.log(greetArrow("ES6"));
```

## Dataset 2: Variables and Data Types
```javascript
// Variable declarations
var oldStyle = "var declaration";
let modernStyle = "let declaration";
const constant = "const declaration";

// Basic data types
const number = 42;
const decimal = 3.14;
const string = "JavaScript";
const boolean = true;
const array = [1, 2, 3, 4, 5];
const object = { name: "John", age: 30 };
const nullValue = null;
const undefinedValue = undefined;

// Type checking
console.log(typeof number);
console.log(typeof string);
console.log(typeof boolean);
console.log(typeof array);
console.log(typeof object);
console.log(typeof nullValue);
console.log(typeof undefinedValue);
```

## Dataset 3: Control Structures
```javascript
// If-else statements
const age = 18;
if (age >= 18) {
    console.log("Adult");
} else if (age >= 13) {
    console.log("Teenager");
} else {
    console.log("Child");
}

// For loops
for (let i = 0; i < 5; i++) {
    console.log(`Number: ${i}`);
}

// While loops
let count = 0;
while (count < 5) {
    console.log(`Count: ${count}`);
    count++;
}

// Switch statement
const day = "Monday";
switch (day) {
    case "Monday":
        console.log("Start of the week");
        break;
    case "Friday":
        console.log("TGIF!");
        break;
    default:
        console.log("Regular day");
}
```

## Dataset 4: Functions
```javascript
// Function declaration
function add(a, b) {
    return a + b;
}

// Function expression
const subtract = function(a, b) {
    return a - b;
};

// Arrow functions
const multiply = (a, b) => a * b;
const square = x => x * x;

// Function with default parameters
function greet(name, greeting = "Hello") {
    return `${greeting}, ${name}!`;
}

// Rest parameters
function sumAll(...numbers) {
    return numbers.reduce((sum, num) => sum + num, 0);
}

// Examples
console.log(add(3, 5));
console.log(subtract(10, 4));
console.log(multiply(3, 4));
console.log(square(5));
console.log(greet("Alice"));
console.log(greet("Bob", "Hi"));
console.log(sumAll(1, 2, 3, 4, 5));
```

## Dataset 5: Arrays and Array Methods
```javascript
// Array creation and manipulation
const fruits = ["apple", "banana", "cherry"];
fruits.push("date");
fruits.unshift("elderberry");
const popped = fruits.pop();
const shifted = fruits.shift();

// Array methods
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);
const evens = numbers.filter(x => x % 2 === 0);
const sum = numbers.reduce((acc, num) => acc + num, 0);

// Array iteration
fruits.forEach((fruit, index) => {
    console.log(`${index}: ${fruit}`);
});

// Finding elements
const found = numbers.find(x => x > 3);
const foundIndex = numbers.findIndex(x => x > 3);

console.log("Doubled:", doubled);
console.log("Evens:", evens);
console.log("Sum:", sum);
console.log("Found:", found);
console.log("Found index:", foundIndex);
```

## Dataset 6: Objects and Object Methods
```javascript
// Object creation
const person = {
    name: "John",
    age: 30,
    city: "New York",
    greet: function() {
        return `Hello, I'm ${this.name}`;
    }
};

// Object manipulation
person.occupation = "Developer";
person.salary = 75000;
delete person.city;

// Object methods
console.log(Object.keys(person));
console.log(Object.values(person));
console.log(Object.entries(person));

// Object iteration
for (const key in person) {
    if (person.hasOwnProperty(key)) {
        console.log(`${key}: ${person[key]}`);
    }
}

// Destructuring
const { name, age } = person;
console.log(name, age);

// Object shorthand
const createUser = (name, email) => ({ name, email });
console.log(createUser("Alice", "alice@example.com"));
```

## Dataset 7: String Methods
```javascript
// String manipulation
const text = "Hello, World!";
console.log(text.toUpperCase());
console.log(text.toLowerCase());
console.log(text.replace("World", "JavaScript"));
console.log(text.split(","));

// Template literals
const name = "Alice";
const age = 25;
console.log(`Name: ${name}, Age: ${age}`);

// String methods
const email = "user@example.com";
console.log(email.startsWith("user"));
console.log(email.endsWith(".com"));
console.log(email.indexOf("@"));
console.log(email.slice(0, 4));
console.log(email.substring(5, 12));

// String searching
const sentence = "The quick brown fox jumps over the lazy dog";
console.log(sentence.includes("fox"));
console.log(sentence.match(/\b\w{4}\b/g)); // 4-letter words
```

## Dataset 8: Error Handling
```javascript
// Try-catch blocks
try {
    const result = riskyOperation();
    console.log(result);
} catch (error) {
    console.error("An error occurred:", error.message);
} finally {
    console.log("This always executes");
}

function riskyOperation() {
    if (Math.random() > 0.5) {
        throw new Error("Random error occurred");
    }
    return "Success!";
}

// Custom error handling
function divide(a, b) {
    try {
        if (b === 0) {
            throw new Error("Division by zero is not allowed");
        }
        return a / b;
    } catch (error) {
        console.error("Division error:", error.message);
        return null;
    }
}

console.log(divide(10, 2));
console.log(divide(10, 0));
```

## Dataset 9: Basic DOM Manipulation (Browser Environment)
```javascript
// DOM selection
const element = document.getElementById("myElement");
const elements = document.getElementsByClassName("myClass");
const querySelector = document.querySelector(".my-class");
const querySelectorAll = document.querySelectorAll("div.container");

// DOM manipulation
if (element) {
    element.textContent = "New text content";
    element.innerHTML = "<strong>Bold text</strong>";
    element.style.color = "blue";
    element.classList.add("new-class");
    element.classList.remove("old-class");
    element.classList.toggle("active");
}

// Event handling
element?.addEventListener("click", function(event) {
    console.log("Element clicked!", event);
});

// Creating elements
const newDiv = document.createElement("div");
newDiv.textContent = "I'm a new div";
newDiv.className = "dynamic-content";
document.body.appendChild(newDiv);
```

## Dataset 10: Async Programming Basics
```javascript
// Promises
function fetchData() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (Math.random() > 0.3) {
                resolve("Data fetched successfully");
            } else {
                reject(new Error("Failed to fetch data"));
            }
        }, 1000);
    });
}

// Using promises
fetchData()
    .then(data => {
        console.log("Success:", data);
    })
    .catch(error => {
        console.error("Error:", error.message);
    });

// Async/await
async function getData() {
    try {
        const data = await fetchData();
        console.log("Async success:", data);
        return data;
    } catch (error) {
        console.error("Async error:", error.message);
        throw error;
    }
}

getData();

// Multiple promises
Promise.all([
    Promise.resolve("First"),
    Promise.resolve("Second"),
    Promise.resolve("Third")
]).then(results => {
    console.log("All results:", results);
});
```