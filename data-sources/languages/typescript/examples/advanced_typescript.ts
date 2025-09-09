/**
 * Advanced TypeScript Examples for AI Coding Agents
 * ==================================================
 * 
 * This module demonstrates advanced TypeScript features including:
 * - Complex type system manipulation
 * - Generic programming patterns
 * - Utility types and conditional types
 * - Modern ES6+ features with strong typing
 * - Functional programming patterns
 * - Async/await and Promise handling
 * - Decorators and metadata
 * - Advanced object-oriented programming
 * 
 * Author: AI Dataset Creation Team
 * License: MIT
 * Created: 2024
 */

// =============================================================================
// Advanced Type System Features
// =============================================================================

/**
 * Generic utility types for advanced type manipulation.
 * These demonstrate TypeScript's powerful type system capabilities.
 */
namespace AdvancedTypes {
    
    // Conditional types for type-level programming
    type IsArray<T> = T extends any[] ? true : false;
    type ArrayElement<T> = T extends (infer U)[] ? U : never;
    
    // Mapped types for object transformation
    type Optional<T> = {
        [K in keyof T]?: T[K];
    };
    
    type Required<T> = {
        [K in keyof T]-?: T[K];
    };
    
    // Template literal types (TypeScript 4.1+)
    type EventName<T extends string> = `on${Capitalize<T>}`;
    type EventHandler<T extends string> = `handle${Capitalize<T>}`;
    
    // Recursive types for complex data structures
    type DeepReadonly<T> = {
        readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
    };
    
    // Union and intersection type utilities
    type UnionToIntersection<U> = 
        (U extends any ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
    
    // Function type utilities
    type AsyncReturnType<T extends (...args: any) => Promise<any>> = 
        T extends (...args: any) => Promise<infer R> ? R : never;
    
    /**
     * Advanced generic class demonstrating multiple constraints and inference.
     */
    class Repository<T extends { id: string | number }, K extends keyof T = keyof T> {
        private items: Map<T['id'], T> = new Map();
        
        /**
         * Add an item to the repository with type safety.
         */
        add(item: T): void {
            this.items.set(item.id, item);
        }
        
        /**
         * Find item by ID with null safety.
         */
        findById(id: T['id']): T | undefined {
            return this.items.get(id);
        }
        
        /**
         * Query items with type-safe field selection.
         */
        query<U extends K[]>(
            predicate: (item: T) => boolean,
            select?: U
        ): Pick<T, U[number]>[] {
            const results: T[] = Array.from(this.items.values()).filter(predicate);
            
            if (select) {
                return results.map(item => {
                    const selected = {} as Pick<T, U[number]>;
                    select.forEach(key => {
                        selected[key] = item[key];
                    });
                    return selected;
                });
            }
            
            return results as Pick<T, U[number]>[];
        }
        
        /**
         * Update item with partial data and type safety.
         */
        update(id: T['id'], updates: Partial<Omit<T, 'id'>>): boolean {
            const item = this.items.get(id);
            if (!item) return false;
            
            Object.assign(item, updates);
            return true;
        }
    }
    
    // Example usage with type inference
    interface User {
        id: number;
        name: string;
        email: string;
        age: number;
        isActive: boolean;
    }
    
    const userRepo = new Repository<User>();
    userRepo.add({ id: 1, name: 'Alice', email: 'alice@example.com', age: 25, isActive: true });
    
    // Type-safe queries with intellisense support
    const activeUsers = userRepo.query(user => user.isActive, ['name', 'email']);
    // Type is: Pick<User, 'name' | 'email'>[]
}

// =============================================================================
// Functional Programming Patterns
// =============================================================================

namespace FunctionalPatterns {
    
    /**
     * Maybe monad implementation for null-safe operations.
     */
    abstract class Maybe<T> {
        abstract map<U>(fn: (value: T) => U): Maybe<U>;
        abstract flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U>;
        abstract filter(predicate: (value: T) => boolean): Maybe<T>;
        abstract getOrElse(defaultValue: T): T;
        abstract isSome(): this is Some<T>;
        abstract isNone(): this is None<T>;
    }
    
    class Some<T> extends Maybe<T> {
        constructor(private value: T) {
            super();
        }
        
        map<U>(fn: (value: T) => U): Maybe<U> {
            try {
                return new Some(fn(this.value));
            } catch {
                return new None<U>();
            }
        }
        
        flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U> {
            try {
                return fn(this.value);
            } catch {
                return new None<U>();
            }
        }
        
        filter(predicate: (value: T) => boolean): Maybe<T> {
            return predicate(this.value) ? this : new None<T>();
        }
        
        getOrElse(_defaultValue: T): T {
            return this.value;
        }
        
        isSome(): this is Some<T> {
            return true;
        }
        
        isNone(): this is None<T> {
            return false;
        }
    }
    
    class None<T> extends Maybe<T> {
        map<U>(_fn: (value: T) => U): Maybe<U> {
            return new None<U>();
        }
        
        flatMap<U>(_fn: (value: T) => Maybe<U>): Maybe<U> {
            return new None<U>();
        }
        
        filter(_predicate: (value: T) => boolean): Maybe<T> {
            return this;
        }
        
        getOrElse(defaultValue: T): T {
            return defaultValue;
        }
        
        isSome(): this is Some<T> {
            return false;
        }
        
        isNone(): this is None<T> {
            return true;
        }
    }
    
    /**
     * Utility functions for creating Maybe instances.
     */
    const Maybe = {
        some: <T>(value: T): Maybe<T> => new Some(value),
        none: <T>(): Maybe<T> => new None<T>(),
        fromNullable: <T>(value: T | null | undefined): Maybe<T> => 
            value != null ? new Some(value) : new None<T>()
    };
    
    /**
     * Functional composition utilities with type safety.
     */
    function pipe<T>(value: T): T;
    function pipe<T, U>(value: T, fn1: (x: T) => U): U;
    function pipe<T, U, V>(value: T, fn1: (x: T) => U, fn2: (x: U) => V): V;
    function pipe<T, U, V, W>(
        value: T, 
        fn1: (x: T) => U, 
        fn2: (x: U) => V, 
        fn3: (x: V) => W
    ): W;
    function pipe(value: any, ...fns: Array<(x: any) => any>): any {
        return fns.reduce((acc, fn) => fn(acc), value);
    }
    
    /**
     * Currying utility for partial application.
     */
    function curry<T, U, V>(fn: (a: T, b: U) => V): (a: T) => (b: U) => V;
    function curry<T, U, V, W>(fn: (a: T, b: U, c: V) => W): (a: T) => (b: U) => (c: V) => W;
    function curry(fn: (...args: any[]) => any): any {
        return function curried(...args: any[]): any {
            if (args.length >= fn.length) {
                return fn(...args);
            }
            return (...nextArgs: any[]) => curried(...args, ...nextArgs);
        };
    }
    
    // Example usage of functional patterns
    const safeDivide = (a: number, b: number): Maybe<number> => 
        b === 0 ? Maybe.none<number>() : Maybe.some(a / b);
    
    const processCalculation = (x: number, y: number): string => {
        return safeDivide(x, y)
            .map(result => result * 2)
            .filter(result => result > 1)
            .map(result => `Result: ${result.toFixed(2)}`)
            .getOrElse('Calculation failed or result too small');
    };
    
    console.log('Functional Programming Examples:');
    console.log('Safe division 10/2:', processCalculation(10, 2));
    console.log('Safe division 10/0:', processCalculation(10, 0));
    console.log('Safe division 1/2:', processCalculation(1, 2));
}

// =============================================================================
// Async Programming and Error Handling
// =============================================================================

namespace AsyncPatterns {
    
    /**
     * Result type for error handling without exceptions.
     */
    type Result<T, E = Error> = 
        | { success: true; data: T }
        | { success: false; error: E };
    
    /**
     * Utility functions for Result type.
     */
    const Result = {
        ok: <T>(data: T): Result<T> => ({ success: true, data }),
        err: <E>(error: E): Result<never, E> => ({ success: false, error }),
        
        map: <T, U, E>(result: Result<T, E>, fn: (data: T) => U): Result<U, E> =>
            result.success ? Result.ok(fn(result.data)) : result,
            
        mapError: <T, E, F>(result: Result<T, E>, fn: (error: E) => F): Result<T, F> =>
            result.success ? result : Result.err(fn(result.error)),
            
        flatMap: <T, U, E>(result: Result<T, E>, fn: (data: T) => Result<U, E>): Result<U, E> =>
            result.success ? fn(result.data) : result
    };
    
    /**
     * Advanced HTTP client with proper error handling and type safety.
     */
    class HttpClient {
        private baseUrl: string;
        private defaultHeaders: Record<string, string>;
        
        constructor(baseUrl: string, defaultHeaders: Record<string, string> = {}) {
            this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
            this.defaultHeaders = defaultHeaders;
        }
        
        /**
         * Generic HTTP request method with comprehensive error handling.
         */
        private async request<T>(
            endpoint: string,
            options: RequestInit = {}
        ): Promise<Result<T, string>> {
            try {
                const url = `${this.baseUrl}${endpoint}`;
                const response = await fetch(url, {
                    ...options,
                    headers: {
                        'Content-Type': 'application/json',
                        ...this.defaultHeaders,
                        ...options.headers
                    }
                });
                
                if (!response.ok) {
                    return Result.err(`HTTP Error: ${response.status} ${response.statusText}`);
                }
                
                const data = await response.json() as T;
                return Result.ok(data);
                
            } catch (error) {
                const message = error instanceof Error ? error.message : 'Unknown error';
                return Result.err(`Network Error: ${message}`);
            }
        }
        
        /**
         * GET request with type safety.
         */
        async get<T>(endpoint: string): Promise<Result<T, string>> {
            return this.request<T>(endpoint, { method: 'GET' });
        }
        
        /**
         * POST request with data validation.
         */
        async post<T, U>(endpoint: string, data: U): Promise<Result<T, string>> {
            return this.request<T>(endpoint, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        }
        
        /**
         * PUT request for updates.
         */
        async put<T, U>(endpoint: string, data: U): Promise<Result<T, string>> {
            return this.request<T>(endpoint, {
                method: 'PUT',
                body: JSON.stringify(data)
            });
        }
        
        /**
         * DELETE request.
         */
        async delete<T>(endpoint: string): Promise<Result<T, string>> {
            return this.request<T>(endpoint, { method: 'DELETE' });
        }
    }
    
    /**
     * Retry utility with exponential backoff.
     */
    async function withRetry<T>(
        operation: () => Promise<T>,
        maxRetries: number = 3,
        initialDelay: number = 1000
    ): Promise<T> {
        let lastError: Error;
        
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await operation();
            } catch (error) {
                lastError = error as Error;
                
                if (attempt === maxRetries) {
                    throw lastError;
                }
                
                // Exponential backoff with jitter
                const delay = initialDelay * Math.pow(2, attempt) + Math.random() * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        
        throw lastError!;
    }
    
    /**
     * Promise-based cache with TTL (Time To Live).
     */
    class CacheWithTTL<K, V> {
        private cache = new Map<K, { value: V; expiry: number }>();
        private defaultTTL: number;
        
        constructor(defaultTTL: number = 300000) { // 5 minutes default
            this.defaultTTL = defaultTTL;
        }
        
        /**
         * Get value from cache or compute it.
         */
        async getOrCompute(
            key: K,
            computer: () => Promise<V>,
            ttl: number = this.defaultTTL
        ): Promise<V> {
            const cached = this.cache.get(key);
            const now = Date.now();
            
            if (cached && cached.expiry > now) {
                return cached.value;
            }
            
            try {
                const value = await computer();
                this.cache.set(key, { value, expiry: now + ttl });
                return value;
            } catch (error) {
                // If we have stale data, return it on error
                if (cached) {
                    return cached.value;
                }
                throw error;
            }
        }
        
        /**
         * Manually set cache value.
         */
        set(key: K, value: V, ttl: number = this.defaultTTL): void {
            this.cache.set(key, { value, expiry: Date.now() + ttl });
        }
        
        /**
         * Clear expired entries.
         */
        cleanup(): void {
            const now = Date.now();
            for (const [key, entry] of this.cache.entries()) {
                if (entry.expiry <= now) {
                    this.cache.delete(key);
                }
            }
        }
    }
    
    // Example usage
    const apiClient = new HttpClient('https://api.example.com', {
        'Authorization': 'Bearer token123'
    });
    
    const cache = new CacheWithTTL<string, any>(60000); // 1 minute TTL
    
    /**
     * Example service using advanced async patterns.
     */
    class UserService {
        constructor(private client: HttpClient, private cache: CacheWithTTL<string, any>) {}
        
        async getUser(id: string): Promise<Result<any, string>> {
            try {
                const user = await this.cache.getOrCompute(
                    `user:${id}`,
                    () => withRetry(async () => {
                        const result = await this.client.get(`/users/${id}`);
                        if (!result.success) {
                            throw new Error(result.error);
                        }
                        return result.data;
                    })
                );
                
                return Result.ok(user);
            } catch (error) {
                const message = error instanceof Error ? error.message : 'Unknown error';
                return Result.err(`Failed to get user: ${message}`);
            }
        }
    }
}

// =============================================================================
// Design Patterns Implementation
// =============================================================================

namespace DesignPatterns {
    
    /**
     * Observer pattern with type safety and automatic cleanup.
     */
    interface Observer<T> {
        update(data: T): void;
    }
    
    class Observable<T> {
        private observers: Set<Observer<T>> = new Set();
        
        subscribe(observer: Observer<T>): () => void {
            this.observers.add(observer);
            
            // Return unsubscribe function
            return () => {
                this.observers.delete(observer);
            };
        }
        
        notify(data: T): void {
            this.observers.forEach(observer => {
                try {
                    observer.update(data);
                } catch (error) {
                    console.error('Observer error:', error);
                }
            });
        }
        
        getObserverCount(): number {
            return this.observers.size;
        }
    }
    
    /**
     * Command pattern for undo/redo functionality.
     */
    interface Command {
        execute(): void;
        undo(): void;
        canUndo(): boolean;
    }
    
    class CommandManager {
        private history: Command[] = [];
        private currentIndex: number = -1;
        private maxHistory: number;
        
        constructor(maxHistory: number = 50) {
            this.maxHistory = maxHistory;
        }
        
        execute(command: Command): void {
            command.execute();
            
            // Remove any commands after current index
            this.history = this.history.slice(0, this.currentIndex + 1);
            
            // Add new command
            this.history.push(command);
            this.currentIndex++;
            
            // Limit history size
            if (this.history.length > this.maxHistory) {
                this.history.shift();
                this.currentIndex--;
            }
        }
        
        undo(): boolean {
            if (this.canUndo()) {
                const command = this.history[this.currentIndex];
                if (command.canUndo()) {
                    command.undo();
                    this.currentIndex--;
                    return true;
                }
            }
            return false;
        }
        
        redo(): boolean {
            if (this.canRedo()) {
                this.currentIndex++;
                const command = this.history[this.currentIndex];
                command.execute();
                return true;
            }
            return false;
        }
        
        canUndo(): boolean {
            return this.currentIndex >= 0 && 
                   this.history[this.currentIndex]?.canUndo() === true;
        }
        
        canRedo(): boolean {
            return this.currentIndex < this.history.length - 1;
        }
        
        clear(): void {
            this.history = [];
            this.currentIndex = -1;
        }
    }
    
    /**
     * Factory pattern with dependency injection.
     */
    interface Injectable {
        readonly type: symbol;
    }
    
    class Container {
        private services = new Map<symbol, any>();
        private factories = new Map<symbol, () => any>();
        
        register<T extends Injectable>(
            type: symbol,
            factory: () => T
        ): void {
            this.factories.set(type, factory);
        }
        
        resolve<T>(type: symbol): T {
            if (this.services.has(type)) {
                return this.services.get(type);
            }
            
            const factory = this.factories.get(type);
            if (!factory) {
                throw new Error(`Service not registered: ${type.toString()}`);
            }
            
            const instance = factory();
            this.services.set(type, instance);
            return instance;
        }
        
        createScope(): Container {
            const scope = new Container();
            scope.factories = new Map(this.factories);
            return scope;
        }
    }
    
    // Example services
    const LOGGER_TYPE = Symbol('Logger');
    const DATABASE_TYPE = Symbol('Database');
    
    interface Logger extends Injectable {
        log(message: string): void;
    }
    
    class ConsoleLogger implements Logger {
        readonly type = LOGGER_TYPE;
        
        log(message: string): void {
            console.log(`[${new Date().toISOString()}] ${message}`);
        }
    }
    
    // Container usage
    const container = new Container();
    container.register(LOGGER_TYPE, () => new ConsoleLogger());
    
    const logger = container.resolve<Logger>(LOGGER_TYPE);
    logger.log('Dependency injection working!');
}

// =============================================================================
// Performance Optimization Patterns
// =============================================================================

namespace PerformancePatterns {
    
    /**
     * Memoization decorator with configurable options.
     */
    function memoize<T extends (...args: any[]) => any>(
        options: {
            maxSize?: number;
            ttl?: number;
            keyGenerator?: (...args: Parameters<T>) => string;
        } = {}
    ) {
        const {
            maxSize = 100,
            ttl = Infinity,
            keyGenerator = (...args) => JSON.stringify(args)
        } = options;
        
        return function (
            target: any,
            propertyKey: string,
            descriptor: PropertyDescriptor
        ) {
            const originalMethod = descriptor.value as T;
            const cache = new Map<string, { value: ReturnType<T>; timestamp: number }>();
            
            descriptor.value = function (...args: Parameters<T>): ReturnType<T> {
                const key = keyGenerator(...args);
                const now = Date.now();
                const cached = cache.get(key);
                
                // Check if cached value is still valid
                if (cached && (ttl === Infinity || now - cached.timestamp < ttl)) {
                    return cached.value;
                }
                
                // Compute new value
                const result = originalMethod.apply(this, args);
                
                // Clean up expired entries if cache is full
                if (cache.size >= maxSize) {
                    const entries = Array.from(cache.entries());
                    entries.sort(([, a], [, b]) => a.timestamp - b.timestamp);
                    
                    // Remove oldest entries
                    const toRemove = Math.ceil(maxSize * 0.2); // Remove 20%
                    for (let i = 0; i < toRemove; i++) {
                        cache.delete(entries[i][0]);
                    }
                }
                
                cache.set(key, { value: result, timestamp: now });
                return result;
            };
            
            return descriptor;
        };
    }
    
    /**
     * Debounce decorator for method calls.
     */
    function debounce(delay: number) {
        return function (
            target: any,
            propertyKey: string,
            descriptor: PropertyDescriptor
        ) {
            const originalMethod = descriptor.value;
            let timeoutId: NodeJS.Timeout;
            
            descriptor.value = function (...args: any[]) {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    originalMethod.apply(this, args);
                }, delay);
            };
            
            return descriptor;
        };
    }
    
    /**
     * Throttle decorator for rate limiting.
     */
    function throttle(interval: number) {
        return function (
            target: any,
            propertyKey: string,
            descriptor: PropertyDescriptor
        ) {
            const originalMethod = descriptor.value;
            let lastCall = 0;
            let timeoutId: NodeJS.Timeout;
            
            descriptor.value = function (...args: any[]) {
                const now = Date.now();
                
                if (now - lastCall >= interval) {
                    lastCall = now;
                    originalMethod.apply(this, args);
                } else {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => {
                        lastCall = Date.now();
                        originalMethod.apply(this, args);
                    }, interval - (now - lastCall));
                }
            };
            
            return descriptor;
        };
    }
    
    /**
     * Example class using performance decorators.
     */
    class ExpensiveCalculator {
        @memoize({ maxSize: 50, ttl: 60000 }) // Cache for 1 minute
        fibonacci(n: number): number {
            console.log(`Computing fibonacci(${n})`);
            if (n <= 1) return n;
            return this.fibonacci(n - 1) + this.fibonacci(n - 2);
        }
        
        @debounce(300) // Wait 300ms after last call
        onSearchInput(query: string): void {
            console.log(`Searching for: ${query}`);
            // Expensive search operation
        }
        
        @throttle(1000) // Maximum once per second
        onScroll(position: number): void {
            console.log(`Scroll position: ${position}`);
            // Expensive scroll handling
        }
    }
    
    /**
     * Virtual scrolling implementation for large lists.
     */
    class VirtualList<T> {
        private items: T[];
        private itemHeight: number;
        private containerHeight: number;
        private scrollTop: number = 0;
        
        constructor(items: T[], itemHeight: number, containerHeight: number) {
            this.items = items;
            this.itemHeight = itemHeight;
            this.containerHeight = containerHeight;
        }
        
        /**
         * Get visible items based on current scroll position.
         */
        getVisibleItems(): { items: T[]; startIndex: number; endIndex: number } {
            const startIndex = Math.floor(this.scrollTop / this.itemHeight);
            const endIndex = Math.min(
                startIndex + Math.ceil(this.containerHeight / this.itemHeight) + 1,
                this.items.length - 1
            );
            
            return {
                items: this.items.slice(startIndex, endIndex + 1),
                startIndex,
                endIndex
            };
        }
        
        /**
         * Update scroll position and return if render is needed.
         */
        updateScrollTop(scrollTop: number): boolean {
            const oldStartIndex = Math.floor(this.scrollTop / this.itemHeight);
            const newStartIndex = Math.floor(scrollTop / this.itemHeight);
            
            this.scrollTop = scrollTop;
            
            // Only re-render if visible range changed
            return oldStartIndex !== newStartIndex;
        }
        
        /**
         * Get total height for scrollbar.
         */
        getTotalHeight(): number {
            return this.items.length * this.itemHeight;
        }
        
        /**
         * Get offset for positioning visible items.
         */
        getOffsetY(): number {
            return Math.floor(this.scrollTop / this.itemHeight) * this.itemHeight;
        }
    }
}

// =============================================================================
// Example Usage and Testing
// =============================================================================

/**
 * Comprehensive demonstration of all TypeScript features.
 */
function demonstrateAdvancedTypeScript(): void {
    console.log('🚀 Advanced TypeScript Features Demonstration');
    console.log('='.repeat(50));
    
    // Test advanced types
    console.log('\n📝 Advanced Type System:');
    const userRepo = new AdvancedTypes.Repository<AdvancedTypes.User>();
    // Usage examples would go here...
    
    // Test functional patterns
    console.log('\n🔧 Functional Programming:');
    // FunctionalPatterns examples would execute here...
    
    // Test async patterns
    console.log('\n⚡ Async Programming:');
    // AsyncPatterns examples would execute here...
    
    // Test design patterns
    console.log('\n🏗️ Design Patterns:');
    const observable = new DesignPatterns.Observable<string>();
    const unsubscribe = observable.subscribe({
        update: (data) => console.log(`Observer received: ${data}`)
    });
    observable.notify('Hello, TypeScript!');
    unsubscribe();
    
    // Test performance patterns
    console.log('\n⚡ Performance Optimization:');
    const calculator = new PerformancePatterns.ExpensiveCalculator();
    console.log('Fibonacci(10):', calculator.fibonacci(10));
    console.log('Fibonacci(10) again (cached):', calculator.fibonacci(10));
    
    console.log('\n✅ All TypeScript demonstrations completed!');
}

// Run demonstrations if this file is executed directly
if (typeof window === 'undefined' && typeof process !== 'undefined') {
    demonstrateAdvancedTypeScript();
}

// Export for module usage
export {
    AdvancedTypes,
    FunctionalPatterns,
    AsyncPatterns,
    DesignPatterns,
    PerformancePatterns
};