/**
 * Intermediate JavaScript Programming Examples
 * ============================================
 * 
 * This file demonstrates intermediate JavaScript concepts including:
 * - Advanced ES6+ features and modern syntax
 * - Design patterns implementation
 * - Functional programming techniques
 * - Asynchronous programming patterns
 * - Object-oriented programming with classes
 * - Performance optimization techniques
 * - Testing strategies and mocking
 * - Browser APIs and Node.js features
 */

// Advanced ES6+ Features and Syntax
// =================================

/**
 * Custom iterator implementation for Fibonacci sequence
 */
class FibonacciIterator {
    constructor(maxCount = Infinity) {
        this.maxCount = maxCount;
        this.current = 0;
        this.next = 1;
        this.count = 0;
    }

    [Symbol.iterator]() {
        return this;
    }

    next() {
        if (this.count >= this.maxCount) {
            return { done: true };
        }

        const value = this.current;
        [this.current, this.next] = [this.next, this.current + this.next];
        this.count++;

        return { value, done: false };
    }
}

/**
 * Generator function for prime numbers
 */
function* primeGenerator(limit = Infinity) {
    const primes = [];
    let candidate = 2;

    while (candidate <= limit) {
        let isPrime = true;
        
        for (const prime of primes) {
            if (prime * prime > candidate) break;
            if (candidate % prime === 0) {
                isPrime = false;
                break;
            }
        }

        if (isPrime) {
            primes.push(candidate);
            yield candidate;
        }
        
        candidate++;
    }
}

/**
 * Advanced destructuring and spread operator examples
 */
const advancedDestructuring = () => {
    // Object destructuring with renaming and defaults
    const user = { id: 1, name: 'John', preferences: { theme: 'dark' } };
    const { 
        id: userId, 
        name = 'Unknown', 
        email = 'no-email@example.com',
        preferences: { theme = 'light', notifications = true } = {}
    } = user;

    // Array destructuring with rest elements
    const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const [first, second, ...middle, secondLast, last] = numbers;

    // Function parameter destructuring
    const processOrder = ({ 
        orderId, 
        items = [], 
        customer: { name: customerName, email } = {},
        shipping = { method: 'standard', cost: 0 }
    }) => {
        return {
            orderId,
            itemCount: items.length,
            customerName,
            email,
            shippingMethod: shipping.method,
            shippingCost: shipping.cost
        };
    };

    return { userId, name, theme, notifications, first, last, processOrder };
};

// Design Patterns Implementation
// =============================

/**
 * Singleton pattern using ES6 classes
 */
class ConfigManager {
    constructor() {
        if (ConfigManager.instance) {
            return ConfigManager.instance;
        }

        this.config = new Map();
        this.subscribers = new Set();
        ConfigManager.instance = this;
    }

    set(key, value) {
        const oldValue = this.config.get(key);
        this.config.set(key, value);
        this.notifySubscribers(key, value, oldValue);
    }

    get(key, defaultValue = null) {
        return this.config.get(key) ?? defaultValue;
    }

    subscribe(callback) {
        this.subscribers.add(callback);
        return () => this.subscribers.delete(callback);
    }

    notifySubscribers(key, newValue, oldValue) {
        this.subscribers.forEach(callback => {
            callback({ key, newValue, oldValue });
        });
    }
}

/**
 * Observer pattern with modern JavaScript
 */
class EventEmitter {
    constructor() {
        this.events = new Map();
    }

    on(event, listener) {
        if (!this.events.has(event)) {
            this.events.set(event, new Set());
        }
        this.events.get(event).add(listener);

        // Return unsubscribe function
        return () => this.off(event, listener);
    }

    off(event, listener) {
        const listeners = this.events.get(event);
        if (listeners) {
            listeners.delete(listener);
            if (listeners.size === 0) {
                this.events.delete(event);
            }
        }
    }

    emit(event, ...args) {
        const listeners = this.events.get(event);
        if (listeners) {
            listeners.forEach(listener => {
                try {
                    listener(...args);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    once(event, listener) {
        const onceWrapper = (...args) => {
            this.off(event, onceWrapper);
            listener(...args);
        };
        this.on(event, onceWrapper);
    }
}

/**
 * Factory pattern with validation
 */
class UserFactory {
    static types = {
        ADMIN: 'admin',
        USER: 'user',
        GUEST: 'guest'
    };

    static create(type, userData) {
        const validators = {
            [this.types.ADMIN]: this.validateAdmin,
            [this.types.USER]: this.validateUser,
            [this.types.GUEST]: this.validateGuest
        };

        const UserClass = {
            [this.types.ADMIN]: AdminUser,
            [this.types.USER]: RegularUser,
            [this.types.GUEST]: GuestUser
        };

        if (!validators[type]) {
            throw new Error(`Unknown user type: ${type}`);
        }

        if (!validators[type](userData)) {
            throw new Error(`Invalid data for user type: ${type}`);
        }

        return new UserClass[type](userData);
    }

    static validateAdmin(data) {
        return data.name && data.email && data.permissions;
    }

    static validateUser(data) {
        return data.name && data.email;
    }

    static validateGuest(data) {
        return data.sessionId;
    }
}

class User {
    constructor(data) {
        this.id = data.id || this.generateId();
        this.createdAt = new Date();
    }

    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
}

class AdminUser extends User {
    constructor(data) {
        super(data);
        this.name = data.name;
        this.email = data.email;
        this.permissions = data.permissions;
        this.type = 'admin';
    }

    hasPermission(permission) {
        return this.permissions.includes(permission);
    }
}

class RegularUser extends User {
    constructor(data) {
        super(data);
        this.name = data.name;
        this.email = data.email;
        this.type = 'user';
    }
}

class GuestUser extends User {
    constructor(data) {
        super(data);
        this.sessionId = data.sessionId;
        this.type = 'guest';
    }
}

// Functional Programming Techniques
// =================================

/**
 * Higher-order functions and currying
 */
const curry = (fn) => {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...args2) {
                return curried.apply(this, args.concat(args2));
            };
        }
    };
};

/**
 * Function composition utilities
 */
const compose = (...fns) => (value) => fns.reduceRight((acc, fn) => fn(acc), value);
const pipe = (...fns) => (value) => fns.reduce((acc, fn) => fn(acc), value);

/**
 * Monadic operations for error handling
 */
class Maybe {
    constructor(value) {
        this.value = value;
    }

    static of(value) {
        return new Maybe(value);
    }

    static nothing() {
        return new Maybe(null);
    }

    isNothing() {
        return this.value === null || this.value === undefined;
    }

    map(fn) {
        return this.isNothing() ? Maybe.nothing() : Maybe.of(fn(this.value));
    }

    flatMap(fn) {
        return this.isNothing() ? Maybe.nothing() : fn(this.value);
    }

    filter(predicate) {
        return this.isNothing() || !predicate(this.value) ? Maybe.nothing() : this;
    }

    getOrElse(defaultValue) {
        return this.isNothing() ? defaultValue : this.value;
    }
}

/**
 * Functional array processing utilities
 */
const arrayUtils = {
    chunk: (array, size) => {
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    },

    partition: (array, predicate) => {
        return array.reduce(
            ([pass, fail], item) => 
                predicate(item) ? [[...pass, item], fail] : [pass, [...fail, item]],
            [[], []]
        );
    },

    groupBy: (array, keyFn) => {
        return array.reduce((groups, item) => {
            const key = keyFn(item);
            groups[key] = groups[key] || [];
            groups[key].push(item);
            return groups;
        }, {});
    },

    unique: (array, keyFn = x => x) => {
        const seen = new Set();
        return array.filter(item => {
            const key = keyFn(item);
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    },

    zip: (...arrays) => {
        const length = Math.min(...arrays.map(arr => arr.length));
        return Array.from({ length }, (_, i) => arrays.map(arr => arr[i]));
    }
};

// Asynchronous Programming Patterns
// =================================

/**
 * Promise-based HTTP client with retry logic
 */
class HTTPClient {
    constructor(baseURL = '', defaultOptions = {}) {
        this.baseURL = baseURL;
        this.defaultOptions = {
            timeout: 10000,
            retries: 3,
            retryDelay: 1000,
            ...defaultOptions
        };
    }

    async request(url, options = {}) {
        const config = { ...this.defaultOptions, ...options };
        const fullURL = this.baseURL + url;
        
        for (let attempt = 1; attempt <= config.retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.timeout);

                const response = await fetch(fullURL, {
                    ...config,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                return await response.json();
            } catch (error) {
                if (attempt === config.retries) {
                    throw error;
                }

                console.warn(`Request attempt ${attempt} failed:`, error.message);
                await this.delay(config.retryDelay * attempt);
            }
        }
    }

    async get(url, options = {}) {
        return this.request(url, { ...options, method: 'GET' });
    }

    async post(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            body: JSON.stringify(data)
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Async iterator for paginated data
 */
class PaginatedIterator {
    constructor(fetchPage, options = {}) {
        this.fetchPage = fetchPage;
        this.currentPage = options.startPage || 1;
        this.pageSize = options.pageSize || 10;
        this.hasMore = true;
    }

    async *[Symbol.asyncIterator]() {
        while (this.hasMore) {
            try {
                const data = await this.fetchPage(this.currentPage, this.pageSize);
                
                if (!data || data.length === 0) {
                    this.hasMore = false;
                    return;
                }

                for (const item of data) {
                    yield item;
                }

                this.currentPage++;
                this.hasMore = data.length === this.pageSize;
            } catch (error) {
                throw new Error(`Failed to fetch page ${this.currentPage}: ${error.message}`);
            }
        }
    }
}

/**
 * Promise pool for concurrent execution with limit
 */
class PromisePool {
    constructor(concurrency = 5) {
        this.concurrency = concurrency;
        this.running = new Set();
        this.queue = [];
    }

    async add(promiseFactory) {
        return new Promise((resolve, reject) => {
            this.queue.push({
                promiseFactory,
                resolve,
                reject
            });
            this.process();
        });
    }

    async process() {
        if (this.running.size >= this.concurrency || this.queue.length === 0) {
            return;
        }

        const { promiseFactory, resolve, reject } = this.queue.shift();
        const promise = promiseFactory();
        this.running.add(promise);

        try {
            const result = await promise;
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running.delete(promise);
            this.process();
        }
    }

    async addAll(promiseFactories) {
        return Promise.all(promiseFactories.map(factory => this.add(factory)));
    }
}

// Advanced Object-Oriented Programming
// ===================================

/**
 * Mixin pattern for shared functionality
 */
const Timestamped = (superclass) => class extends superclass {
    constructor(...args) {
        super(...args);
        this.createdAt = new Date();
        this.updatedAt = new Date();
    }

    touch() {
        this.updatedAt = new Date();
    }

    getAge() {
        return Date.now() - this.createdAt.getTime();
    }
};

const Validatable = (superclass) => class extends superclass {
    constructor(...args) {
        super(...args);
        this.validationErrors = [];
    }

    addValidationRule(field, rule, message) {
        if (!this.validationRules) {
            this.validationRules = {};
        }
        if (!this.validationRules[field]) {
            this.validationRules[field] = [];
        }
        this.validationRules[field].push({ rule, message });
    }

    validate() {
        this.validationErrors = [];
        
        if (!this.validationRules) return true;

        for (const [field, rules] of Object.entries(this.validationRules)) {
            const value = this[field];
            for (const { rule, message } of rules) {
                if (!rule(value)) {
                    this.validationErrors.push({ field, message });
                }
            }
        }

        return this.validationErrors.length === 0;
    }

    isValid() {
        return this.validationErrors.length === 0;
    }
};

/**
 * Advanced class with mixins
 */
class Product extends Timestamped(Validatable(class {})) {
    constructor(name, price, category) {
        super();
        this.name = name;
        this.price = price;
        this.category = category;

        // Add validation rules
        this.addValidationRule('name', value => value && value.trim().length > 0, 'Name is required');
        this.addValidationRule('price', value => value > 0, 'Price must be positive');
        this.addValidationRule('category', value => value && value.trim().length > 0, 'Category is required');
    }

    updatePrice(newPrice) {
        if (newPrice > 0) {
            this.price = newPrice;
            this.touch();
        }
    }

    toJSON() {
        return {
            name: this.name,
            price: this.price,
            category: this.category,
            createdAt: this.createdAt,
            updatedAt: this.updatedAt
        };
    }
}

// Performance Optimization Techniques
// ===================================

/**
 * Memoization decorator
 */
const memoize = (fn, keyGenerator = (...args) => JSON.stringify(args)) => {
    const cache = new Map();
    
    return function memoized(...args) {
        const key = keyGenerator(...args);
        
        if (cache.has(key)) {
            return cache.get(key);
        }
        
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
};

/**
 * Debounce and throttle utilities
 */
const debounce = (fn, delay) => {
    let timeoutId;
    
    return function debounced(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
};

const throttle = (fn, limit) => {
    let inThrottle;
    
    return function throttled(...args) {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};

/**
 * Lazy evaluation utility
 */
class Lazy {
    constructor(computation) {
        this.computation = computation;
        this.computed = false;
        this.value = undefined;
    }

    get() {
        if (!this.computed) {
            this.value = this.computation();
            this.computed = true;
        }
        return this.value;
    }

    map(fn) {
        return new Lazy(() => fn(this.get()));
    }

    static of(value) {
        return new Lazy(() => value);
    }
}

/**
 * Performance monitoring utility
 */
class PerformanceMonitor {
    constructor() {
        this.metrics = new Map();
    }

    measure(name, fn) {
        const start = performance.now();
        
        try {
            const result = fn();
            const end = performance.now();
            this.recordMetric(name, end - start, 'success');
            return result;
        } catch (error) {
            const end = performance.now();
            this.recordMetric(name, end - start, 'error');
            throw error;
        }
    }

    async measureAsync(name, asyncFn) {
        const start = performance.now();
        
        try {
            const result = await asyncFn();
            const end = performance.now();
            this.recordMetric(name, end - start, 'success');
            return result;
        } catch (error) {
            const end = performance.now();
            this.recordMetric(name, end - start, 'error');
            throw error;
        }
    }

    recordMetric(name, duration, status) {
        if (!this.metrics.has(name)) {
            this.metrics.set(name, []);
        }
        
        this.metrics.get(name).push({
            duration,
            status,
            timestamp: Date.now()
        });
    }

    getStats(name) {
        const measurements = this.metrics.get(name) || [];
        const durations = measurements.map(m => m.duration);
        
        if (durations.length === 0) {
            return null;
        }

        const sorted = durations.sort((a, b) => a - b);
        const sum = durations.reduce((a, b) => a + b, 0);
        
        return {
            count: durations.length,
            min: Math.min(...durations),
            max: Math.max(...durations),
            avg: sum / durations.length,
            median: sorted[Math.floor(sorted.length / 2)],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            successRate: measurements.filter(m => m.status === 'success').length / measurements.length
        };
    }
}

// Testing Utilities and Mocking
// =============================

/**
 * Simple testing framework
 */
class TestRunner {
    constructor() {
        this.tests = [];
        this.results = [];
    }

    test(description, testFn) {
        this.tests.push({ description, testFn });
    }

    async run() {
        console.log(`Running ${this.tests.length} tests...\n`);
        
        for (const { description, testFn } of this.tests) {
            try {
                await testFn();
                this.results.push({ description, status: 'PASS' });
                console.log(`✅ ${description}`);
            } catch (error) {
                this.results.push({ description, status: 'FAIL', error: error.message });
                console.log(`❌ ${description}: ${error.message}`);
            }
        }

        this.printSummary();
    }

    printSummary() {
        const passed = this.results.filter(r => r.status === 'PASS').length;
        const failed = this.results.filter(r => r.status === 'FAIL').length;
        
        console.log(`\nTest Summary: ${passed} passed, ${failed} failed`);
    }
}

/**
 * Mock factory for testing
 */
class MockFactory {
    static createMock(original = {}) {
        const mock = { ...original };
        const callLog = new Map();

        return new Proxy(mock, {
            get(target, prop) {
                if (prop === '__callLog') {
                    return callLog;
                }
                
                if (prop === '__reset') {
                    return () => {
                        callLog.clear();
                        Object.keys(target).forEach(key => {
                            if (typeof target[key] === 'function') {
                                target[key] = () => {};
                            }
                        });
                    };
                }

                if (typeof target[prop] === 'function') {
                    return (...args) => {
                        if (!callLog.has(prop)) {
                            callLog.set(prop, []);
                        }
                        callLog.get(prop).push(args);
                        return target[prop](...args);
                    };
                }

                return target[prop];
            },

            set(target, prop, value) {
                target[prop] = value;
                return true;
            }
        });
    }
}

// Main Demo Function
// ==================

async function main() {
    console.log('=== Intermediate JavaScript Programming Examples ===\n');

    // 1. ES6+ Features Demo
    console.log('1. ES6+ Features:');
    const { userId, name, theme } = advancedDestructuring();
    console.log(`   User: ${name} (ID: ${userId}, Theme: ${theme})`);

    const fibIterator = new FibonacciIterator(10);
    const fibSequence = [...fibIterator];
    console.log(`   Fibonacci: ${fibSequence.join(', ')}`);

    const primes = [...primeGenerator(30)];
    console.log(`   Primes up to 30: ${primes.join(', ')}`);

    // 2. Design Patterns Demo
    console.log('\n2. Design Patterns:');
    
    // Singleton
    const config1 = new ConfigManager();
    const config2 = new ConfigManager();
    console.log(`   Singleton test: ${config1 === config2}`);

    // Observer
    const eventEmitter = new EventEmitter();
    eventEmitter.on('test', (data) => console.log(`   Event received: ${data}`));
    eventEmitter.emit('test', 'Hello from EventEmitter!');

    // Factory
    const admin = UserFactory.create(UserFactory.types.ADMIN, {
        name: 'Admin User',
        email: 'admin@example.com',
        permissions: ['read', 'write', 'delete']
    });
    console.log(`   Created user: ${admin.name} (${admin.type})`);

    // 3. Functional Programming Demo
    console.log('\n3. Functional Programming:');
    
    const add = curry((a, b, c) => a + b + c);
    const addTen = add(10);
    const addTenAndFive = addTen(5);
    console.log(`   Curried function result: ${addTenAndFive(3)}`); // 18

    const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const [evens, odds] = arrayUtils.partition(numbers, x => x % 2 === 0);
    console.log(`   Evens: ${evens.join(', ')}, Odds: ${odds.join(', ')}`);

    const maybeValue = Maybe.of(42)
        .map(x => x * 2)
        .filter(x => x > 50)
        .map(x => x + 10);
    console.log(`   Maybe result: ${maybeValue.getOrElse('No value')}`);

    // 4. Async Programming Demo
    console.log('\n4. Async Programming:');
    
    const client = new HTTPClient('https://jsonplaceholder.typicode.com');
    
    try {
        // Simulate API call
        console.log('   Making HTTP request...');
        // In a real scenario: const data = await client.get('/posts/1');
        console.log('   HTTP request completed (simulated)');
    } catch (error) {
        console.log(`   HTTP request failed: ${error.message}`);
    }

    // Promise pool demo
    const pool = new PromisePool(3);
    const tasks = Array.from({ length: 5 }, (_, i) => 
        () => new Promise(resolve => {
            setTimeout(() => resolve(`Task ${i + 1} completed`), 100);
        })
    );

    const results = await pool.addAll(tasks);
    console.log(`   Promise pool completed ${results.length} tasks`);

    // 5. OOP with Mixins Demo
    console.log('\n5. Object-Oriented Programming:');
    
    const product = new Product('Laptop', 999.99, 'Electronics');
    console.log(`   Product created: ${product.name} - $${product.price}`);
    console.log(`   Is valid: ${product.validate()}`);
    console.log(`   Age: ${product.getAge()}ms`);

    // 6. Performance Optimization Demo
    console.log('\n6. Performance Optimization:');
    
    const monitor = new PerformanceMonitor();
    
    // Expensive function
    const expensiveCalculation = memoize((n) => {
        let result = 0;
        for (let i = 0; i < n; i++) {
            result += Math.sqrt(i);
        }
        return result;
    });

    monitor.measure('expensive_calc', () => expensiveCalculation(100000));
    monitor.measure('expensive_calc_cached', () => expensiveCalculation(100000)); // Should be faster
    
    const stats = monitor.getStats('expensive_calc');
    if (stats) {
        console.log(`   Performance stats - Avg: ${stats.avg.toFixed(2)}ms, Count: ${stats.count}`);
    }

    // Lazy evaluation demo
    const lazyValue = new Lazy(() => {
        console.log('   Computing lazy value...');
        return 42 * 2;
    });
    
    console.log(`   Lazy value (first access): ${lazyValue.get()}`);
    console.log(`   Lazy value (second access): ${lazyValue.get()}`); // No recomputation

    // 7. Testing Demo
    console.log('\n7. Testing Framework:');
    
    const testRunner = new TestRunner();
    
    testRunner.test('Array utils should partition correctly', () => {
        const [evens, odds] = arrayUtils.partition([1, 2, 3, 4], x => x % 2 === 0);
        if (evens.length !== 2 || odds.length !== 2) {
            throw new Error('Partition failed');
        }
    });

    testRunner.test('Maybe monad should handle null values', () => {
        const result = Maybe.of(null).map(x => x * 2).getOrElse(0);
        if (result !== 0) {
            throw new Error('Maybe should return default for null');
        }
    });

    await testRunner.run();

    console.log('\n=== Intermediate JavaScript Demo Complete ===');
}

// Browser-specific features (if in browser)
if (typeof window !== 'undefined') {
    // DOM manipulation utilities
    const domUtils = {
        $: (selector) => document.querySelector(selector),
        $$: (selector) => [...document.querySelectorAll(selector)],
        
        createElement: (tag, attributes = {}, children = []) => {
            const element = document.createElement(tag);
            
            Object.entries(attributes).forEach(([key, value]) => {
                if (key === 'className') {
                    element.className = value;
                } else if (key.startsWith('on')) {
                    element.addEventListener(key.slice(2).toLowerCase(), value);
                } else {
                    element.setAttribute(key, value);
                }
            });
            
            children.forEach(child => {
                if (typeof child === 'string') {
                    element.appendChild(document.createTextNode(child));
                } else {
                    element.appendChild(child);
                }
            });
            
            return element;
        },

        ready: (callback) => {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', callback);
            } else {
                callback();
            }
        }
    };

    // Intersection Observer utility
    const createObserver = (callback, options = {}) => {
        return new IntersectionObserver((entries) => {
            entries.forEach(entry => callback(entry));
        }, options);
    };
}

// Node.js specific features (if in Node.js)
if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    // Run the demo
    main().catch(console.error);
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FibonacciIterator,
        primeGenerator,
        ConfigManager,
        EventEmitter,
        UserFactory,
        HTTPClient,
        PromisePool,
        arrayUtils,
        Maybe,
        curry,
        compose,
        pipe,
        memoize,
        debounce,
        throttle,
        Lazy,
        PerformanceMonitor,
        TestRunner,
        MockFactory,
        main
    };
}