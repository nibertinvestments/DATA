/**
 * Advanced TypeScript Type System Examples
 * 
 * This module demonstrates sophisticated TypeScript features including:
 * - Advanced type manipulation and inference
 * - Template literal types and mapped types  
 * - Conditional types and type guards
 * - Utility types and type-level programming
 * - Strict null safety and error handling
 * 
 * These examples showcase TypeScript's powerful type system for
 * building robust, maintainable applications with compile-time safety.
 * 
 * @author AI Training Dataset
 * @version 1.0
 */

// ============================================================================
// ADVANCED TYPE MANIPULATIONS
// ============================================================================

/**
 * Deep readonly utility type that recursively makes all properties readonly
 */
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

/**
 * Deep partial utility type that recursively makes all properties optional
 */
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Extract function parameter types as a tuple
 */
type Parameters<T extends (...args: any[]) => any> = T extends (...args: infer P) => any ? P : never;

/**
 * Extract function return type
 */
type ReturnType<T extends (...args: any[]) => any> = T extends (...args: any[]) => infer R ? R : any;

/**
 * Create a union of all possible paths through an object
 */
type PathsToStringProps<T> = T extends string
  ? []
  : {
      [K in Extract<keyof T, string>]: [K, ...PathsToStringProps<T[K]>];
    }[Extract<keyof T, string>];

/**
 * Advanced string manipulation at the type level
 */
type CamelCase<S extends string> = S extends `${infer P1}_${infer P2}${infer P3}`
  ? `${P1}${Uppercase<P2>}${CamelCase<P3>}`
  : S;

type KebabCase<S extends string> = S extends `${infer C}${infer T}`
  ? `${C extends Uppercase<C> ? `-${Lowercase<C>}` : C}${KebabCase<T>}`
  : S;

// ============================================================================
// FUNCTIONAL PROGRAMMING PATTERNS
// ============================================================================

/**
 * Result type for error handling without exceptions
 */
type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Option type for nullable values
 */
type Option<T> = T | null | undefined;

/**
 * Utility functions for Result type
 */
namespace ResultUtils {
  export const ok = <T>(data: T): Result<T, never> => ({ success: true, data });
  
  export const err = <E>(error: E): Result<never, E> => ({ success: false, error });
  
  export const map = <T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => U
  ): Result<U, E> => {
    return result.success ? ok(fn(result.data)) : result;
  };
  
  export const flatMap = <T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
  ): Result<U, E> => {
    return result.success ? fn(result.data) : result;
  };
  
  export const unwrapOr = <T, E>(result: Result<T, E>, defaultValue: T): T => {
    return result.success ? result.data : defaultValue;
  };
}

/**
 * Functional pipeline for composing operations
 */
interface Pipe {
  <A>(value: A): A;
  <A, B>(value: A, fn1: (a: A) => B): B;
  <A, B, C>(value: A, fn1: (a: A) => B, fn2: (b: B) => C): C;
  <A, B, C, D>(value: A, fn1: (a: A) => B, fn2: (b: B) => C, fn3: (c: C) => D): D;
  // ... more overloads as needed
}

const pipe: Pipe = (value: any, ...fns: Function[]) => {
  return fns.reduce((acc, fn) => fn(acc), value);
};

/**
 * Curried function utilities
 */
type Curry<T> = T extends (...args: infer A) => infer R
  ? A extends [infer First, ...infer Rest]
    ? (arg: First) => Rest extends []
      ? R
      : Curry<(...args: Rest) => R>
    : R
  : never;

const curry = <T extends (...args: any[]) => any>(fn: T): Curry<T> => {
  return ((...args: any[]) => {
    if (args.length >= fn.length) {
      return fn(...args);
    }
    return curry(fn.bind(null, ...args));
  }) as Curry<T>;
};

// ============================================================================
// ADVANCED DATA STRUCTURES WITH TYPE SAFETY
// ============================================================================

/**
 * Immutable List implementation with advanced type features
 */
abstract class ImmutableList<T> {
  abstract readonly length: number;
  abstract head(): Option<T>;
  abstract tail(): ImmutableList<T>;
  abstract prepend(item: T): ImmutableList<T>;
  abstract map<U>(fn: (item: T) => U): ImmutableList<U>;
  abstract filter(predicate: (item: T) => boolean): ImmutableList<T>;
  abstract reduce<U>(fn: (acc: U, item: T) => U, initial: U): U;
  abstract toArray(): T[];
  
  // Type-safe operations
  flatMap<U>(fn: (item: T) => ImmutableList<U>): ImmutableList<U> {
    return this.reduce(
      (acc, item) => acc.concat(fn(item)),
      EmptyList.instance<U>()
    );
  }
  
  concat(other: ImmutableList<T>): ImmutableList<T> {
    return this.reduce(
      (acc, item) => acc.prepend(item),
      other
    ).reverse();
  }
  
  reverse(): ImmutableList<T> {
    return this.reduce(
      (acc, item) => acc.prepend(item),
      EmptyList.instance<T>()
    );
  }
  
  find(predicate: (item: T) => boolean): Option<T> {
    const filtered = this.filter(predicate);
    return filtered.head();
  }
  
  // Type guards for pattern matching
  isEmpty(): this is EmptyList<T> {
    return this instanceof EmptyList;
  }
  
  isNonEmpty(): this is NonEmptyList<T> {
    return this instanceof NonEmptyList;
  }
}

class EmptyList<T> extends ImmutableList<T> {
  readonly length = 0;
  
  private static _instance: EmptyList<any> = new EmptyList();
  
  static instance<T>(): EmptyList<T> {
    return EmptyList._instance;
  }
  
  head(): Option<T> {
    return null;
  }
  
  tail(): ImmutableList<T> {
    return this;
  }
  
  prepend(item: T): NonEmptyList<T> {
    return new NonEmptyList(item, this);
  }
  
  map<U>(): EmptyList<U> {
    return EmptyList.instance<U>();
  }
  
  filter(): EmptyList<T> {
    return this;
  }
  
  reduce<U>(_fn: (acc: U, item: T) => U, initial: U): U {
    return initial;
  }
  
  toArray(): T[] {
    return [];
  }
}

class NonEmptyList<T> extends ImmutableList<T> {
  readonly length: number;
  
  constructor(
    private readonly _head: T,
    private readonly _tail: ImmutableList<T>
  ) {
    super();
    this.length = 1 + _tail.length;
  }
  
  head(): T {
    return this._head;
  }
  
  tail(): ImmutableList<T> {
    return this._tail;
  }
  
  prepend(item: T): NonEmptyList<T> {
    return new NonEmptyList(item, this);
  }
  
  map<U>(fn: (item: T) => U): NonEmptyList<U> {
    return new NonEmptyList(fn(this._head), this._tail.map(fn));
  }
  
  filter(predicate: (item: T) => boolean): ImmutableList<T> {
    const filteredTail = this._tail.filter(predicate);
    return predicate(this._head) 
      ? new NonEmptyList(this._head, filteredTail)
      : filteredTail;
  }
  
  reduce<U>(fn: (acc: U, item: T) => U, initial: U): U {
    const headResult = fn(initial, this._head);
    return this._tail.reduce(fn, headResult);
  }
  
  toArray(): T[] {
    return [this._head, ...this._tail.toArray()];
  }
}

// ============================================================================
// DEPENDENCY INJECTION CONTAINER
// ============================================================================

/**
 * Type-safe dependency injection container
 */
type Constructor<T = {}> = new (...args: any[]) => T;

type ServiceKey<T> = string | symbol | Constructor<T>;

interface ServiceDescriptor<T> {
  factory: () => T;
  singleton?: boolean;
}

class DIContainer {
  private services = new Map<ServiceKey<any>, ServiceDescriptor<any>>();
  private singletonInstances = new Map<ServiceKey<any>, any>();
  
  register<T>(
    key: ServiceKey<T>,
    factory: () => T,
    options: { singleton?: boolean } = {}
  ): this {
    this.services.set(key, {
      factory,
      singleton: options.singleton ?? false
    });
    return this;
  }
  
  registerClass<T>(
    constructor: Constructor<T>,
    dependencies: ServiceKey<any>[] = [],
    options: { singleton?: boolean } = {}
  ): this {
    return this.register(
      constructor,
      () => {
        const deps = dependencies.map(dep => this.resolve(dep));
        return new constructor(...deps);
      },
      options
    );
  }
  
  resolve<T>(key: ServiceKey<T>): T {
    const descriptor = this.services.get(key);
    if (!descriptor) {
      throw new Error(`Service not registered: ${String(key)}`);
    }
    
    if (descriptor.singleton) {
      let instance = this.singletonInstances.get(key);
      if (!instance) {
        instance = descriptor.factory();
        this.singletonInstances.set(key, instance);
      }
      return instance;
    }
    
    return descriptor.factory();
  }
  
  // Type-safe resolution with compile-time checking
  get<T>(key: ServiceKey<T>): T {
    return this.resolve(key);
  }
}

// ============================================================================
// REACTIVE PROGRAMMING PATTERNS
// ============================================================================

/**
 * Observable implementation with type safety
 */
type Observer<T> = (value: T) => void;
type Unsubscribe = () => void;

class Observable<T> {
  private observers: Observer<T>[] = [];
  
  constructor(private producer?: (observer: Observer<T>) => void) {}
  
  subscribe(observer: Observer<T>): Unsubscribe {
    this.observers.push(observer);
    
    if (this.producer) {
      this.producer(observer);
    }
    
    return () => {
      const index = this.observers.indexOf(observer);
      if (index > -1) {
        this.observers.splice(index, 1);
      }
    };
  }
  
  next(value: T): void {
    this.observers.forEach(observer => observer(value));
  }
  
  map<U>(fn: (value: T) => U): Observable<U> {
    return new Observable<U>(observer => {
      return this.subscribe(value => observer(fn(value)));
    });
  }
  
  filter(predicate: (value: T) => boolean): Observable<T> {
    return new Observable<T>(observer => {
      return this.subscribe(value => {
        if (predicate(value)) {
          observer(value);
        }
      });
    });
  }
  
  debounce(delay: number): Observable<T> {
    return new Observable<T>(observer => {
      let timeoutId: NodeJS.Timeout;
      
      return this.subscribe(value => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => observer(value), delay);
      });
    });
  }
  
  // Static factory methods
  static of<T>(...values: T[]): Observable<T> {
    return new Observable<T>(observer => {
      values.forEach(value => observer(value));
    });
  }
  
  static fromEvent<T extends Event>(
    target: EventTarget,
    eventName: string
  ): Observable<T> {
    return new Observable<T>(observer => {
      const handler = (event: Event) => observer(event as T);
      target.addEventListener(eventName, handler);
      
      return () => target.removeEventListener(eventName, handler);
    });
  }
  
  static interval(delay: number): Observable<number> {
    return new Observable<number>(observer => {
      let count = 0;
      const intervalId = setInterval(() => {
        observer(count++);
      }, delay);
      
      return () => clearInterval(intervalId);
    });
  }
}

// ============================================================================
// EXAMPLE USAGE AND DEMONSTRATIONS
// ============================================================================

/**
 * Demonstration of advanced TypeScript features
 */
function demonstrateAdvancedTypeScript(): void {
  console.log('=== Advanced TypeScript Examples ===\n');
  
  // 1. Immutable List Operations
  console.log('1. Immutable List Operations');
  console.log('-'.repeat(30));
  
  const list = EmptyList.instance<number>()
    .prepend(3)
    .prepend(2)
    .prepend(1);
  
  const doubled = list.map(x => x * 2);
  const evens = list.filter(x => x % 2 === 0);
  
  console.log('Original list:', list.toArray());
  console.log('Doubled:', doubled.toArray());
  console.log('Evens only:', evens.toArray());
  console.log('Sum:', list.reduce((acc, x) => acc + x, 0));
  
  // 2. Result Type for Error Handling
  console.log('\n2. Result Type for Error Handling');
  console.log('-'.repeat(35));
  
  const safeDivide = (a: number, b: number): Result<number> => {
    return b === 0 
      ? ResultUtils.err(new Error('Division by zero'))
      : ResultUtils.ok(a / b);
  };
  
  const result1 = safeDivide(10, 2);
  const result2 = safeDivide(10, 0);
  
  console.log('10 / 2 =', ResultUtils.unwrapOr(result1, 0));
  console.log('10 / 0 =', ResultUtils.unwrapOr(result2, 0));
  
  // 3. Functional Pipeline
  console.log('\n3. Functional Pipeline');
  console.log('-'.repeat(25));
  
  const processNumbers = pipe(
    [1, 2, 3, 4, 5],
    (nums: number[]) => nums.filter(x => x % 2 === 0),
    (nums: number[]) => nums.map(x => x * x),
    (nums: number[]) => nums.reduce((acc, x) => acc + x, 0)
  );
  
  console.log('Process [1,2,3,4,5] -> filter evens -> square -> sum:', processNumbers);
  
  // 4. Dependency Injection
  console.log('\n4. Dependency Injection');
  console.log('-'.repeat(25));
  
  interface Logger {
    log(message: string): void;
  }
  
  class ConsoleLogger implements Logger {
    log(message: string): void {
      console.log(`[LOG] ${message}`);
    }
  }
  
  interface ApiService {
    getData(): string;
  }
  
  class ApiServiceImpl implements ApiService {
    constructor(private logger: Logger) {}
    
    getData(): string {
      this.logger.log('Fetching data...');
      return 'Sample data';
    }
  }
  
  const container = new DIContainer()
    .register<Logger>('Logger', () => new ConsoleLogger(), { singleton: true })
    .register<ApiService>('ApiService', () => 
      new ApiServiceImpl(container.get<Logger>('Logger'))
    );
  
  const apiService = container.get<ApiService>('ApiService');
  console.log('API Result:', apiService.getData());
  
  // 5. Observable Pattern
  console.log('\n5. Observable Pattern');
  console.log('-'.repeat(20));
  
  const numbers$ = Observable.of(1, 2, 3, 4, 5);
  const doubled$ = numbers$.map(x => x * 2);
  const evens$ = doubled$.filter(x => x % 4 === 0);
  
  evens$.subscribe(value => {
    console.log('Even doubled number:', value);
  });
  
  console.log('\n=== Advanced TypeScript demonstration complete ===');
}

// Export types and utilities for use in other modules
export {
  DeepReadonly,
  DeepPartial,
  PathsToStringProps,
  CamelCase,
  KebabCase,
  Result,
  Option,
  ResultUtils,
  pipe,
  curry,
  ImmutableList,
  EmptyList,
  NonEmptyList,
  DIContainer,
  Observable,
  demonstrateAdvancedTypeScript
};

// Type-level tests (these will be checked at compile time)
type TestCamelCase = CamelCase<'hello_world_test'>; // Should be 'helloWorldTest'
type TestKebabCase = KebabCase<'HelloWorldTest'>; // Should be 'hello-world-test'

// Run demonstration if this module is executed directly
if (typeof require !== 'undefined' && require.main === module) {
  demonstrateAdvancedTypeScript();
}