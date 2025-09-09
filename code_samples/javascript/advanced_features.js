// Advanced JavaScript programming concepts for ML/AI training

// ES6+ Modern JavaScript Features
console.log("=== Modern JavaScript Features ===");

// Destructuring assignments
const person = { name: 'Alice', age: 30, city: 'New York', country: 'USA' };
const { name, age, ...location } = person;
console.log(`${name} is ${age} years old and lives in`, location);

const numbers = [1, 2, 3, 4, 5];
const [first, second, ...rest] = numbers;
console.log(`First: ${first}, Second: ${second}, Rest:`, rest);

// Template literals and tagged templates
const formatCurrency = (strings, ...values) => {
    return strings.reduce((result, string, i) => {
        const value = values[i] ? `$${values[i].toFixed(2)}` : '';
        return result + string + value;
    }, '');
};

const price = 99.99;
const tax = 8.25;
const message = formatCurrency`Total cost: ${price} plus tax: ${tax}`;
console.log(message);

// Arrow functions and lexical this
class Timer {
    constructor() {
        this.seconds = 0;
        this.intervalId = null;
    }
    
    start() {
        this.intervalId = setInterval(() => {
            this.seconds++;
            console.log(`Timer: ${this.seconds} seconds`);
        }, 1000);
    }
    
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
}

// Classes and inheritance
class Shape {
    constructor(name) {
        this.name = name;
    }
    
    area() {
        throw new Error('Area method must be implemented');
    }
    
    describe() {
        return `This is a ${this.name} with area ${this.area()}`;
    }
    
    static compareAreas(shape1, shape2) {
        return shape1.area() - shape2.area();
    }
}

class Rectangle extends Shape {
    constructor(width, height) {
        super('rectangle');
        this.width = width;
        this.height = height;
    }
    
    area() {
        return this.width * this.height;
    }
    
    get perimeter() {
        return 2 * (this.width + this.height);
    }
}

class Circle extends Shape {
    constructor(radius) {
        super('circle');
        this.radius = radius;
    }
    
    area() {
        return Math.PI * this.radius ** 2;
    }
    
    get circumference() {
        return 2 * Math.PI * this.radius;
    }
}

// Promises and async/await
console.log("\n=== Asynchronous Programming ===");

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchUserData(userId) {
    await delay(100); // Simulate network delay
    
    if (userId <= 0) {
        throw new Error('Invalid user ID');
    }
    
    return {
        id: userId,
        name: `User ${userId}`,
        email: `user${userId}@example.com`,
        created: new Date().toISOString()
    };
}

async function fetchMultipleUsers(userIds) {
    const promises = userIds.map(id => fetchUserData(id));
    
    try {
        const users = await Promise.all(promises);
        return users;
    } catch (error) {
        console.error('Error fetching users:', error.message);
        throw error;
    }
}

// Generators and iterators
function* fibonacci() {
    let a = 0, b = 1;
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

function* range(start, end, step = 1) {
    for (let i = start; i < end; i += step) {
        yield i;
    }
}

// Higher-order functions and functional programming
const functional = {
    // Currying
    curry: (fn) => {
        return function curried(...args) {
            if (args.length >= fn.length) {
                return fn.apply(this, args);
            } else {
                return function(...args2) {
                    return curried.apply(this, args.concat(args2));
                };
            }
        };
    },
    
    // Compose functions
    compose: (...fns) => (value) => fns.reduceRight((acc, fn) => fn(acc), value),
    
    // Pipe functions
    pipe: (...fns) => (value) => fns.reduce((acc, fn) => fn(acc), value),
    
    // Memoization
    memoize: (fn) => {
        const cache = new Map();
        return function(...args) {
            const key = JSON.stringify(args);
            if (cache.has(key)) {
                return cache.get(key);
            }
            const result = fn.apply(this, args);
            cache.set(key, result);
            return result;
        };
    }
};

// Advanced array methods
console.log("\n=== Advanced Array Operations ===");

const data = [
    { name: 'Alice', department: 'Engineering', salary: 90000 },
    { name: 'Bob', department: 'Sales', salary: 70000 },
    { name: 'Charlie', department: 'Engineering', salary: 95000 },
    { name: 'Diana', department: 'Marketing', salary: 75000 },
    { name: 'Eve', department: 'Engineering', salary: 88000 }
];

// Complex filtering and mapping
const highEarners = data
    .filter(emp => emp.salary > 80000)
    .map(emp => ({
        ...emp,
        grade: emp.salary > 90000 ? 'Senior' : 'Mid-level'
    }))
    .sort((a, b) => b.salary - a.salary);

console.log('High earners:', highEarners);

// Grouping data
const groupBy = (array, key) => {
    return array.reduce((groups, item) => {
        const group = item[key];
        groups[group] = groups[group] || [];
        groups[group].push(item);
        return groups;
    }, {});
};

const departmentGroups = groupBy(data, 'department');
console.log('Grouped by department:', departmentGroups);

// Set and Map data structures
console.log("\n=== Advanced Data Structures ===");

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }
    
    get(key) {
        if (this.cache.has(key)) {
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value);
            return value;
        }
        return -1;
    }
    
    put(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }
    
    size() {
        return this.cache.size;
    }
}

// WeakMap and WeakSet examples
const objectMetadata = new WeakMap();
const processedObjects = new WeakSet();

class DataProcessor {
    process(obj) {
        if (processedObjects.has(obj)) {
            return objectMetadata.get(obj);
        }
        
        const metadata = {
            processed: new Date(),
            id: Math.random().toString(36).substr(2, 9)
        };
        
        objectMetadata.set(obj, metadata);
        processedObjects.add(obj);
        
        return metadata;
    }
}

// Proxy for dynamic behavior
const createValidator = (target, validators) => {
    return new Proxy(target, {
        set(obj, prop, value) {
            if (validators[prop]) {
                const isValid = validators[prop](value);
                if (!isValid) {
                    throw new Error(`Invalid value for ${prop}: ${value}`);
                }
            }
            obj[prop] = value;
            return true;
        }
    });
};

const userValidators = {
    age: value => typeof value === 'number' && value >= 0 && value <= 150,
    email: value => typeof value === 'string' && value.includes('@')
};

// Module pattern and closures
const createModule = (() => {
    let privateCounter = 0;
    const privateData = new Map();
    
    return {
        increment() {
            privateCounter++;
            return privateCounter;
        },
        
        decrement() {
            privateCounter--;
            return privateCounter;
        },
        
        getCount() {
            return privateCounter;
        },
        
        setData(key, value) {
            privateData.set(key, value);
        },
        
        getData(key) {
            return privateData.get(key);
        }
    };
})();

// Error handling and custom errors
class ValidationError extends Error {
    constructor(message, field) {
        super(message);
        this.name = 'ValidationError';
        this.field = field;
    }
}

class APIError extends Error {
    constructor(message, status, code) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.code = code;
    }
}

async function validateAndProcessData(data) {
    try {
        if (!data.email || !data.email.includes('@')) {
            throw new ValidationError('Invalid email format', 'email');
        }
        
        if (!data.age || data.age < 0) {
            throw new ValidationError('Invalid age', 'age');
        }
        
        // Simulate API call
        await delay(50);
        
        return {
            success: true,
            processedData: {
                ...data,
                id: Math.random().toString(36).substr(2, 9),
                processed: new Date().toISOString()
            }
        };
        
    } catch (error) {
        if (error instanceof ValidationError) {
            console.error(`Validation error in field ${error.field}: ${error.message}`);
        } else {
            console.error('Unexpected error:', error);
        }
        throw error;
    }
}

// Regular expressions and string manipulation
console.log("\n=== Regular Expressions and String Processing ===");

const textProcessor = {
    extractEmails: (text) => {
        const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
        return text.match(emailRegex) || [];
    },
    
    extractPhones: (text) => {
        const phoneRegex = /(\+1-?)?(\d{3})-?(\d{3})-?(\d{4})/g;
        return text.match(phoneRegex) || [];
    },
    
    slugify: (text) => {
        return text
            .toLowerCase()
            .trim()
            .replace(/[^\w\s-]/g, '')
            .replace(/[\s_-]+/g, '-')
            .replace(/^-+|-+$/g, '');
    },
    
    capitalizeWords: (text) => {
        return text.replace(/\b\w/g, char => char.toUpperCase());
    },
    
    truncate: (text, length, suffix = '...') => {
        return text.length > length ? text.substring(0, length) + suffix : text;
    }
};

// Event system implementation
class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
        return this;
    }
    
    off(event, listenerToRemove) {
        if (!this.events[event]) return this;
        
        this.events[event] = this.events[event].filter(
            listener => listener !== listenerToRemove
        );
        return this;
    }
    
    emit(event, ...args) {
        if (!this.events[event]) return false;
        
        this.events[event].forEach(listener => {
            listener.apply(this, args);
        });
        return true;
    }
    
    once(event, listener) {
        const onceListener = (...args) => {
            this.off(event, onceListener);
            listener.apply(this, args);
        };
        return this.on(event, onceListener);
    }
}

// Performance measurement utilities
const performance = {
    measure: (name, fn) => {
        const start = Date.now();
        const result = fn();
        const end = Date.now();
        console.log(`${name} took ${end - start}ms`);
        return result;
    },
    
    measureAsync: async (name, asyncFn) => {
        const start = Date.now();
        const result = await asyncFn();
        const end = Date.now();
        console.log(`${name} took ${end - start}ms`);
        return result;
    },
    
    benchmark: (functions, iterations = 1000) => {
        const results = {};
        
        for (const [name, fn] of Object.entries(functions)) {
            const start = Date.now();
            for (let i = 0; i < iterations; i++) {
                fn();
            }
            const end = Date.now();
            results[name] = end - start;
        }
        
        return results;
    }
};

// Main execution and demonstration
async function main() {
    console.log("\n=== Execution Examples ===");
    
    // Test shapes
    const rectangle = new Rectangle(5, 10);
    const circle = new Circle(3);
    
    console.log(rectangle.describe());
    console.log(circle.describe());
    console.log(`Rectangle perimeter: ${rectangle.perimeter}`);
    console.log(`Circle circumference: ${circle.circumference.toFixed(2)}`);
    
    // Test async operations
    try {
        const users = await fetchMultipleUsers([1, 2, 3]);
        console.log('Fetched users:', users);
    } catch (error) {
        console.error('Failed to fetch users:', error.message);
    }
    
    // Test generators
    const fib = fibonacci();
    const fibNumbers = [];
    for (let i = 0; i < 10; i++) {
        fibNumbers.push(fib.next().value);
    }
    console.log('First 10 Fibonacci numbers:', fibNumbers);
    
    // Test range generator
    const rangeNumbers = [...range(1, 10, 2)];
    console.log('Range 1-10 step 2:', rangeNumbers);
    
    // Test functional programming
    const add = (a, b) => a + b;
    const multiply = (a, b) => a * b;
    const curriedAdd = functional.curry(add);
    
    console.log('Curried add:', curriedAdd(5)(3));
    
    const addThenMultiply = functional.pipe(
        x => x + 5,
        x => x * 2,
        x => x - 1
    );
    
    console.log('Pipe result:', addThenMultiply(10));
    
    // Test LRU Cache
    const cache = new LRUCache(3);
    cache.put('a', 1);
    cache.put('b', 2);
    cache.put('c', 3);
    console.log('Cache get a:', cache.get('a'));
    cache.put('d', 4); // This should evict 'b'
    console.log('Cache get b (should be -1):', cache.get('b'));
    
    // Test validation
    try {
        const user = createValidator({}, userValidators);
        user.age = 25;
        user.email = 'test@example.com';
        console.log('Validated user:', user);
        
        // This should throw an error
        user.age = -5;
    } catch (error) {
        console.log('Validation error caught:', error.message);
    }
    
    // Test text processing
    const sampleText = "Contact us at support@example.com or call 555-123-4567";
    console.log('Extracted emails:', textProcessor.extractEmails(sampleText));
    console.log('Extracted phones:', textProcessor.extractPhones(sampleText));
    console.log('Slugified:', textProcessor.slugify("Hello World! This is a Test."));
    
    // Test event emitter
    const emitter = new EventEmitter();
    emitter.on('test', (data) => console.log('Test event:', data));
    emitter.emit('test', { message: 'Hello from event system!' });
    
    // Test performance measurement
    performance.measure('Array creation', () => {
        return new Array(1000).fill(0).map((_, i) => i * 2);
    });
    
    console.log('\nAll JavaScript examples completed!');
}

// Run the main function
main().catch(console.error);