# Basic TypeScript Dataset - Modern JavaScript with Types

## Dataset 1: Hello World and Basic Types
```typescript
// Simple Hello World
console.log("Hello, World!");

// Hello World with types
function greet(name: string): string {
    return `Hello, ${name}!`;
}

const greeting: string = greet("TypeScript");
console.log(greeting);

// Arrow function with types
const greetArrow = (name: string = "World"): string => `Hello, ${name}!`;
console.log(greetArrow());
console.log(greetArrow("Developer"));
```

## Dataset 2: Basic Types and Type Annotations
```typescript
// Primitive types
const message: string = "Hello TypeScript";
const count: number = 42;
const isActive: boolean = true;
const value: null = null;
const notDefined: undefined = undefined;

// Array types
const numbers: number[] = [1, 2, 3, 4, 5];
const fruits: Array<string> = ["apple", "banana", "cherry"];

// Tuple types
const person: [string, number] = ["Alice", 25];
const coordinates: [number, number, number] = [10, 20, 30];

// Object type
const user: { name: string; age: number; email?: string } = {
    name: "John",
    age: 30
};

// Type outputs
console.log("Message:", message);
console.log("Count:", count);
console.log("Is Active:", isActive);
console.log("Numbers:", numbers);
console.log("Person:", person);
console.log("User:", user);
```

## Dataset 3: Interfaces and Type Aliases
```typescript
// Interface definition
interface User {
    id: number;
    name: string;
    email: string;
    isActive?: boolean;
}

// Type alias
type Status = "pending" | "approved" | "rejected";
type ID = string | number;

// Using interfaces
const createUser = (userData: User): User => {
    return {
        id: userData.id,
        name: userData.name,
        email: userData.email,
        isActive: userData.isActive ?? true
    };
};

const newUser: User = createUser({
    id: 1,
    name: "Alice Smith",
    email: "alice@example.com"
});

// Using type aliases
const updateStatus = (id: ID, status: Status): void => {
    console.log(`User ${id} status updated to ${status}`);
};

updateStatus(1, "approved");
updateStatus("user-123", "pending");

console.log("New user:", newUser);
```

## Dataset 4: Functions with Types
```typescript
// Function with typed parameters and return type
function add(a: number, b: number): number {
    return a + b;
}

// Function with optional parameters
function greet(name: string, greeting?: string): string {
    return `${greeting || "Hello"}, ${name}!`;
}

// Function with default parameters
function createMessage(text: string, prefix: string = "Info"): string {
    return `[${prefix}] ${text}`;
}

// Function with rest parameters
function sum(...numbers: number[]): number {
    return numbers.reduce((total, num) => total + num, 0);
}

// Function overloads
function process(value: string): string;
function process(value: number): number;
function process(value: string | number): string | number {
    if (typeof value === "string") {
        return value.toUpperCase();
    }
    return value * 2;
}

// Higher-order function
function applyOperation(
    numbers: number[],
    operation: (num: number) => number
): number[] {
    return numbers.map(operation);
}

// Examples
console.log("Add:", add(5, 3));
console.log("Greet:", greet("Alice"));
console.log("Greet with custom:", greet("Bob", "Hi"));
console.log("Message:", createMessage("System ready"));
console.log("Sum:", sum(1, 2, 3, 4, 5));
console.log("Process string:", process("hello"));
console.log("Process number:", process(5));

const doubled = applyOperation([1, 2, 3, 4], x => x * 2);
console.log("Doubled:", doubled);
```

## Dataset 5: Classes with TypeScript
```typescript
// Basic class with access modifiers
class Person {
    private _name: string;
    private _age: number;
    public readonly id: number;
    
    constructor(name: string, age: number, id: number) {
        this._name = name;
        this._age = age;
        this.id = id;
    }
    
    // Getter and setter
    get name(): string {
        return this._name;
    }
    
    set name(newName: string) {
        if (newName.length > 0) {
            this._name = newName;
        }
    }
    
    get age(): number {
        return this._age;
    }
    
    set age(newAge: number) {
        if (newAge >= 0) {
            this._age = newAge;
        }
    }
    
    // Methods
    introduce(): string {
        return `Hi, I'm ${this._name} and I'm ${this._age} years old`;
    }
    
    haveBirthday(): void {
        this._age++;
        console.log(`Happy birthday! Now I'm ${this._age}`);
    }
}

// Inheritance
class Student extends Person {
    private _studentId: string;
    private _courses: string[];
    
    constructor(name: string, age: number, id: number, studentId: string) {
        super(name, age, id);
        this._studentId = studentId;
        this._courses = [];
    }
    
    get studentId(): string {
        return this._studentId;
    }
    
    addCourse(course: string): void {
        this._courses.push(course);
    }
    
    getCourses(): string[] {
        return [...this._courses];
    }
    
    // Override method
    introduce(): string {
        return `${super.introduce()}. My student ID is ${this._studentId}`;
    }
}

// Using classes
const person = new Person("Alice", 25, 1);
console.log(person.introduce());
person.haveBirthday();

const student = new Student("Bob", 20, 2, "S12345");
student.addCourse("TypeScript");
student.addCourse("React");
console.log(student.introduce());
console.log("Courses:", student.getCourses());
```

## Dataset 6: Generics
```typescript
// Generic function
function identity<T>(arg: T): T {
    return arg;
}

// Generic function with constraints
interface Lengthwise {
    length: number;
}

function loggingIdentity<T extends Lengthwise>(arg: T): T {
    console.log("Length:", arg.length);
    return arg;
}

// Generic class
class Container<T> {
    private _value: T;
    
    constructor(value: T) {
        this._value = value;
    }
    
    getValue(): T {
        return this._value;
    }
    
    setValue(newValue: T): void {
        this._value = newValue;
    }
}

// Generic interface
interface Repository<T> {
    create(item: T): T;
    findById(id: string): T | null;
    update(id: string, item: Partial<T>): T | null;
    delete(id: string): boolean;
}

// Using generics
const stringIdentity = identity<string>("Hello");
const numberIdentity = identity<number>(42);
const booleanIdentity = identity(true); // Type inference

console.log("String identity:", stringIdentity);
console.log("Number identity:", numberIdentity);
console.log("Boolean identity:", booleanIdentity);

loggingIdentity("Hello World");
loggingIdentity([1, 2, 3]);

const stringContainer = new Container<string>("TypeScript");
const numberContainer = new Container<number>(100);

console.log("String container:", stringContainer.getValue());
console.log("Number container:", numberContainer.getValue());
```

## Dataset 7: Enums and Literal Types
```typescript
// Numeric enum
enum Direction {
    Up,
    Down,
    Left,
    Right
}

// String enum
enum Color {
    Red = "red",
    Green = "green",
    Blue = "blue"
}

// Const enum (for performance)
const enum HttpStatus {
    OK = 200,
    NotFound = 404,
    InternalServerError = 500
}

// Literal types
type Theme = "light" | "dark" | "auto";
type Size = "small" | "medium" | "large";

// Using enums and literal types
function move(direction: Direction): string {
    switch (direction) {
        case Direction.Up:
            return "Moving up";
        case Direction.Down:
            return "Moving down";
        case Direction.Left:
            return "Moving left";
        case Direction.Right:
            return "Moving right";
        default:
            return "Unknown direction";
    }
}

function setTheme(theme: Theme): void {
    console.log(`Theme set to: ${theme}`);
}

function createButton(size: Size, color: Color): string {
    return `Button: ${size} ${color}`;
}

// Examples
console.log(move(Direction.Up));
console.log(move(Direction.Right));

setTheme("dark");
setTheme("light");

console.log(createButton("large", Color.Blue));
console.log("HTTP Status:", HttpStatus.OK);
```

## Dataset 8: Union and Intersection Types
```typescript
// Union types
type StringOrNumber = string | number;
type Status = "loading" | "success" | "error";

function formatValue(value: StringOrNumber): string {
    if (typeof value === "string") {
        return value.toUpperCase();
    }
    return value.toString();
}

function handleStatus(status: Status): string {
    switch (status) {
        case "loading":
            return "Please wait...";
        case "success":
            return "Operation completed successfully";
        case "error":
            return "An error occurred";
        default:
            const exhaustiveCheck: never = status;
            return exhaustiveCheck;
    }
}

// Intersection types
interface Timestamped {
    timestamp: Date;
}

interface Tagged {
    tag: string;
}

type TimestampedTag = Timestamped & Tagged;

function createTimestampedTag(tag: string): TimestampedTag {
    return {
        tag,
        timestamp: new Date()
    };
}

// Discriminated unions
interface Circle {
    kind: "circle";
    radius: number;
}

interface Rectangle {
    kind: "rectangle";
    width: number;
    height: number;
}

interface Triangle {
    kind: "triangle";
    base: number;
    height: number;
}

type Shape = Circle | Rectangle | Triangle;

function calculateArea(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle":
            return (shape.base * shape.height) / 2;
        default:
            const exhaustiveCheck: never = shape;
            return exhaustiveCheck;
    }
}

// Examples
console.log("Format string:", formatValue("hello"));
console.log("Format number:", formatValue(42));

console.log("Status loading:", handleStatus("loading"));
console.log("Status success:", handleStatus("success"));

const taggedItem = createTimestampedTag("example");
console.log("Tagged item:", taggedItem);

const circle: Circle = { kind: "circle", radius: 5 };
const rectangle: Rectangle = { kind: "rectangle", width: 10, height: 20 };

console.log("Circle area:", calculateArea(circle));
console.log("Rectangle area:", calculateArea(rectangle));
```

## Dataset 9: Async/Await with Types
```typescript
// Promise with types
function fetchData(): Promise<string> {
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

// Generic promise function
function fetchGenericData<T>(data: T): Promise<T> {
    return new Promise((resolve) => {
        setTimeout(() => resolve(data), 500);
    });
}

// Async function with error handling
async function getData(): Promise<string | null> {
    try {
        const data = await fetchData();
        console.log("Success:", data);
        return data;
    } catch (error) {
        console.error("Error:", error instanceof Error ? error.message : "Unknown error");
        return null;
    }
}

// Multiple async operations
async function fetchMultipleData(): Promise<void> {
    try {
        const [userData, settingsData] = await Promise.all([
            fetchGenericData({ id: 1, name: "Alice" }),
            fetchGenericData({ theme: "dark", language: "en" })
        ]);
        
        console.log("User data:", userData);
        console.log("Settings data:", settingsData);
    } catch (error) {
        console.error("Failed to fetch multiple data:", error);
    }
}

// Async generator
async function* generateNumbers(): AsyncGenerator<number, void, unknown> {
    for (let i = 1; i <= 5; i++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        yield i;
    }
}

// Examples
getData().then(result => {
    console.log("Final result:", result);
});

fetchMultipleData();

// Using async generator
(async () => {
    console.log("Async generator:");
    for await (const number of generateNumbers()) {
        console.log("Generated number:", number);
    }
})();
```

## Dataset 10: Utility Types and Advanced Types
```typescript
// Base interface for examples
interface User {
    id: number;
    name: string;
    email: string;
    age: number;
    isActive: boolean;
}

// Utility types
type PartialUser = Partial<User>; // All properties optional
type RequiredUser = Required<User>; // All properties required
type UserEmail = Pick<User, "name" | "email">; // Only selected properties
type UserWithoutId = Omit<User, "id">; // Exclude selected properties
type UserRecord = Record<string, User>; // Record type

// Conditional types
type NonNullable<T> = T extends null | undefined ? never : T;
type ApiResponse<T> = T extends string ? { message: T } : { data: T };

// Mapped types
type ReadonlyUser = {
    readonly [K in keyof User]: User[K];
};

type OptionalUser = {
    [K in keyof User]?: User[K];
};

// Template literal types
type EventNames = "click" | "hover" | "focus";
type EventHandlers = {
    [K in EventNames as `on${Capitalize<K>}`]: () => void;
};

// Using utility types
function updateUser(id: number, updates: PartialUser): User {
    const existingUser: User = {
        id: 1,
        name: "John Doe",
        email: "john@example.com",
        age: 30,
        isActive: true
    };
    
    return { ...existingUser, ...updates, id };
}

function createUserProfile(data: UserWithoutId): User {
    return {
        id: Math.floor(Math.random() * 1000),
        ...data
    };
}

function processApiResponse<T>(data: T): ApiResponse<T> {
    if (typeof data === "string") {
        return { message: data } as ApiResponse<T>;
    }
    return { data } as ApiResponse<T>;
}

// Examples
const updatedUser = updateUser(1, { name: "Jane Doe", age: 31 });
console.log("Updated user:", updatedUser);

const newProfile = createUserProfile({
    name: "Alice Smith",
    email: "alice@example.com",
    age: 25,
    isActive: true
});
console.log("New profile:", newProfile);

const stringResponse = processApiResponse("Hello World");
const dataResponse = processApiResponse({ id: 1, value: "test" });
console.log("String response:", stringResponse);
console.log("Data response:", dataResponse);

// Event handlers example
const eventHandlers: EventHandlers = {
    onClick: () => console.log("Clicked"),
    onHover: () => console.log("Hovered"),
    onFocus: () => console.log("Focused")
};

eventHandlers.onClick();
```