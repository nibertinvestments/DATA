/**
 * Advanced Generic Types in TypeScript
 * Demonstrates generic constraints, mapped types, and conditional types
 */

// Generic constraints
interface Lengthwise {
    length: number;
}

function logLength<T extends Lengthwise>(arg: T): T {
    console.log(arg.length);
    return arg;
}

// Generic factory pattern
interface Constructor<T> {
    new(...args: any[]): T;
}

function create<T>(ctor: Constructor<T>, ...args: any[]): T {
    return new ctor(...args);
}

// Custom mapped types
type MyReadonly<T> = {
    readonly [P in keyof T]: T[P];
};

type MyPartial<T> = {
    [P in keyof T]?: T[P];
};

type Nullable<T> = {
    [P in keyof T]: T[P] | null;
};

// Conditional types
type Flatten<T> = T extends Array<infer U> ? U : T;

type ExtractPromise<T> = T extends Promise<infer U> ? U : T;

// Advanced generics
interface Repository<T> {
    getById(id: string): T | null;
    getAll(): T[];
    create(item: Omit<T, 'id'>): T;
    update(id: string, item: MyPartial<T>): T;
    delete(id: string): boolean;
}

class InMemoryRepository<T extends { id: string }> implements Repository<T> {
    private items: { [key: string]: T } = {};
    
    getById(id: string): T | null {
        return this.items[id] || null;
    }
    
    getAll(): T[] {
        const result: T[] = [];
        for (const key in this.items) {
            if (this.items.hasOwnProperty(key)) {
                result.push(this.items[key]);
            }
        }
        return result;
    }
    
    create(item: Omit<T, 'id'>): T {
        const id = Math.random().toString(36).substr(2, 9);
        const newItem = { ...item as any, id } as T;
        this.items[id] = newItem;
        return newItem;
    }
    
    update(id: string, item: MyPartial<T>): T {
        const existing = this.items[id];
        if (!existing) throw new Error('Not found');
        const updated = { ...existing as any, ...item as any };
        this.items[id] = updated;
        return updated;
    }
    
    delete(id: string): boolean {
        if (this.items[id]) {
            delete this.items[id];
            return true;
        }
        return false;
    }
}

// Type guards
function isString(value: unknown): value is string {
    return typeof value === 'string';
}

function isArray<T>(value: unknown): value is T[] {
    return Array.isArray(value);
}

// Utility types
type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// Example usage
interface User {
    id: string;
    name: string;
    email: string;
    age: number;
}

const userRepo = new InMemoryRepository<User>();

console.log("Advanced TypeScript Generics");
console.log("===========================");
