/**
 * Comprehensive TypeScript Examples
 * Demonstrates advanced types, interfaces, generics, decorators, and modern patterns
 */

// ========== Type System Demonstrations ==========

// Basic Types and Interfaces
interface User {
    readonly id: number;
    name: string;
    email: string;
    age?: number;
    roles: string[];
    metadata: Record<string, any>;
}

// Generic Interfaces
interface Repository<T, K = number> {
    findById(id: K): Promise<T | null>;
    findAll(): Promise<T[]>;
    create(entity: Omit<T, 'id'>): Promise<T>;
    update(id: K, entity: Partial<T>): Promise<T | null>;
    delete(id: K): Promise<boolean>;
}

// Union and Intersection Types
type Status = 'pending' | 'approved' | 'rejected';
type Theme = 'light' | 'dark' | 'auto';

interface BaseEntity {
    id: number;
    createdAt: Date;
    updatedAt: Date;
}

interface Timestamped {
    timestamp: number;
}

type EntityWithTimestamp = BaseEntity & Timestamped;

// Conditional Types
type NonNullable<T> = T extends null | undefined ? never : T;
type ReturnTypeOf<T> = T extends (...args: any[]) => infer R ? R : any;
type PromiseValue<T> = T extends Promise<infer U> ? U : T;

// Mapped Types
type Optional<T> = {
    [P in keyof T]?: T[P];
};

type ReadOnly<T> = {
    readonly [P in keyof T]: T[P];
};

type StringKeys<T> = {
    [K in keyof T]: T[K] extends string ? K : never;
}[keyof T];

// Template Literal Types
type EventName<T extends string> = `on${Capitalize<T>}`;
type ApiEndpoint<T extends string> = `/api/v1/${T}`;

// ========== Advanced Generic Patterns ==========

// Generic Factory Pattern
interface Factory<T> {
    create(...args: any[]): T;
}

class UserFactory implements Factory<User> {
    private static idCounter = 1;

    create(name: string, email: string, roles: string[] = ['user']): User {
        return {
            id: UserFactory.idCounter++,
            name,
            email,
            roles,
            metadata: {}
        };
    }
}

// Generic Repository Implementation
class InMemoryRepository<T extends BaseEntity> implements Repository<T> {
    private items: Map<number, T> = new Map();
    private nextId = 1;

    async findById(id: number): Promise<T | null> {
        return this.items.get(id) || null;
    }

    async findAll(): Promise<T[]> {
        return Array.from(this.items.values());
    }

    async create(entity: Omit<T, 'id'>): Promise<T> {
        const now = new Date();
        const newEntity = {
            ...entity,
            id: this.nextId++,
            createdAt: now,
            updatedAt: now
        } as T;

        this.items.set(newEntity.id, newEntity);
        return newEntity;
    }

    async update(id: number, entity: Partial<T>): Promise<T | null> {
        const existing = this.items.get(id);
        if (!existing) return null;

        const updated = {
            ...existing,
            ...entity,
            updatedAt: new Date()
        };

        this.items.set(id, updated);
        return updated;
    }

    async delete(id: number): Promise<boolean> {
        return this.items.delete(id);
    }

    // Additional query methods
    async findBy<K extends keyof T>(
        field: K,
        value: T[K]
    ): Promise<T[]> {
        return Array.from(this.items.values()).filter(
            item => item[field] === value
        );
    }

    async count(): Promise<number> {
        return this.items.size;
    }
}

// ========== Decorators ==========

// Class Decorator
function Entity(tableName: string) {
    return function <T extends { new (...args: any[]): {} }>(constructor: T) {
        return class extends constructor {
            tableName = tableName;
        };
    };
}

// Method Decorator
function LogExecution(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = function (...args: any[]) {
        console.log(`Executing ${propertyName} with args:`, args);
        const start = performance.now();
        const result = method.apply(this, args);
        const end = performance.now();
        console.log(`${propertyName} completed in ${end - start}ms`);
        return result;
    };
}

// Property Decorator
function Validate(validationFn: (value: any) => boolean, errorMessage: string) {
    return function (target: any, propertyName: string) {
        let value: any;

        const getter = () => value;
        const setter = (newValue: any) => {
            if (!validationFn(newValue)) {
                throw new Error(`${errorMessage}: ${newValue}`);
            }
            value = newValue;
        };

        Object.defineProperty(target, propertyName, {
            get: getter,
            set: setter,
            enumerable: true,
            configurable: true
        });
    };
}

// ========== Advanced Class Patterns ==========

// Abstract Base Class
abstract class Service<T extends BaseEntity> {
    constructor(protected repository: Repository<T>) {}

    abstract validate(entity: Partial<T>): Promise<boolean>;

    async getById(id: number): Promise<T | null> {
        return this.repository.findById(id);
    }

    async getAll(): Promise<T[]> {
        return this.repository.findAll();
    }

    @LogExecution
    async create(entity: Omit<T, 'id' | 'createdAt' | 'updatedAt'>): Promise<T> {
        const isValid = await this.validate(entity);
        if (!isValid) {
            throw new Error('Validation failed');
        }
        return this.repository.create(entity);
    }

    async update(id: number, updates: Partial<T>): Promise<T | null> {
        const isValid = await this.validate(updates);
        if (!isValid) {
            throw new Error('Validation failed');
        }
        return this.repository.update(id, updates);
    }

    async delete(id: number): Promise<boolean> {
        return this.repository.delete(id);
    }
}

// Concrete Implementation
interface Product extends BaseEntity {
    name: string;
    price: number;
    category: string;
    inStock: boolean;
}

@Entity('products')
class ProductService extends Service<Product> {
    @Validate(
        (value: any) => typeof value === 'string' && value.length >= 2,
        'Product name must be at least 2 characters'
    )
    private _name!: string;

    async validate(product: Partial<Product>): Promise<boolean> {
        if (product.name && product.name.length < 2) return false;
        if (product.price && product.price < 0) return false;
        return true;
    }

    async findByCategory(category: string): Promise<Product[]> {
        const products = await this.repository.findAll();
        return products.filter(p => p.category === category);
    }

    async findInStock(): Promise<Product[]> {
        const products = await this.repository.findAll();
        return products.filter(p => p.inStock);
    }

    async updatePrice(id: number, newPrice: number): Promise<Product | null> {
        if (newPrice < 0) {
            throw new Error('Price cannot be negative');
        }
        return this.update(id, { price: newPrice });
    }
}

// ========== Async Patterns ==========

// Promise-based HTTP Client
interface HttpResponse<T> {
    data: T;
    status: number;
    statusText: string;
    headers: Record<string, string>;
}

class HttpClient {
    constructor(private baseURL: string = '') {}

    async get<T>(url: string, headers?: Record<string, string>): Promise<HttpResponse<T>> {
        // Simulated HTTP request
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    data: {} as T,
                    status: 200,
                    statusText: 'OK',
                    headers: { 'content-type': 'application/json' }
                });
            }, 100);
        });
    }

    async post<T, U>(url: string, data: U, headers?: Record<string, string>): Promise<HttpResponse<T>> {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    data: data as unknown as T,
                    status: 201,
                    statusText: 'Created',
                    headers: { 'content-type': 'application/json' }
                });
            }, 150);
        });
    }

    async put<T, U>(url: string, data: U, headers?: Record<string, string>): Promise<HttpResponse<T>> {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    data: data as unknown as T,
                    status: 200,
                    statusText: 'OK',
                    headers: { 'content-type': 'application/json' }
                });
            }, 120);
        });
    }

    async delete(url: string, headers?: Record<string, string>): Promise<HttpResponse<void>> {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    data: undefined as void,
                    status: 204,
                    statusText: 'No Content',
                    headers: {}
                });
            }, 100);
        });
    }
}

// Async Iterator Pattern
class AsyncDataStream<T> {
    constructor(private data: T[], private batchSize: number = 5) {}

    async *[Symbol.asyncIterator](): AsyncIterableIterator<T[]> {
        for (let i = 0; i < this.data.length; i += this.batchSize) {
            // Simulate async processing
            await new Promise(resolve => setTimeout(resolve, 100));
            yield this.data.slice(i, i + this.batchSize);
        }
    }

    async collect(): Promise<T[]> {
        const result: T[] = [];
        for await (const batch of this) {
            result.push(...batch);
        }
        return result;
    }

    async forEach(callback: (batch: T[]) => void | Promise<void>): Promise<void> {
        for await (const batch of this) {
            await callback(batch);
        }
    }
}

// ========== Utility Types and Functions ==========

// Type Guards
function isUser(obj: any): obj is User {
    return obj && 
           typeof obj.id === 'number' &&
           typeof obj.name === 'string' &&
           typeof obj.email === 'string' &&
           Array.isArray(obj.roles);
}

function isProduct(obj: any): obj is Product {
    return obj &&
           typeof obj.id === 'number' &&
           typeof obj.name === 'string' &&
           typeof obj.price === 'number' &&
           typeof obj.category === 'string' &&
           typeof obj.inStock === 'boolean';
}

// Generic Utility Functions
function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
    const result = {} as Pick<T, K>;
    keys.forEach(key => {
        if (key in obj) {
            result[key] = obj[key];
        }
    });
    return result;
}

function omit<T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> {
    const result = { ...obj };
    keys.forEach(key => {
        delete result[key];
    });
    return result;
}

function deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (obj instanceof Date) {
        return new Date(obj.getTime()) as T;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item)) as T;
    }

    const cloned = {} as T;
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}

// ========== Event System ==========

interface EventMap {
    'user:created': { user: User };
    'user:updated': { user: User; changes: Partial<User> };
    'user:deleted': { userId: number };
    'product:created': { product: Product };
    'product:updated': { product: Product; changes: Partial<Product> };
}

type EventCallback<T> = (data: T) => void | Promise<void>;

class EventEmitter<T extends Record<string, any> = EventMap> {
    private listeners: Map<keyof T, Set<EventCallback<any>>> = new Map();

    on<K extends keyof T>(event: K, callback: EventCallback<T[K]>): () => void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(callback);

        // Return unsubscribe function
        return () => {
            this.listeners.get(event)?.delete(callback);
        };
    }

    emit<K extends keyof T>(event: K, data: T[K]): void {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event callback for ${String(event)}:`, error);
                }
            });
        }
    }

    async emitAsync<K extends keyof T>(event: K, data: T[K]): Promise<void> {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            const promises = Array.from(callbacks).map(callback => {
                try {
                    return Promise.resolve(callback(data));
                } catch (error) {
                    console.error(`Error in async event callback for ${String(event)}:`, error);
                    return Promise.resolve();
                }
            });
            await Promise.all(promises);
        }
    }

    off<K extends keyof T>(event: K, callback?: EventCallback<T[K]>): void {
        if (callback) {
            this.listeners.get(event)?.delete(callback);
        } else {
            this.listeners.delete(event);
        }
    }

    removeAllListeners(): void {
        this.listeners.clear();
    }
}

// ========== State Management Pattern ==========

interface State {
    users: User[];
    products: Product[];
    currentUser: User | null;
    loading: boolean;
    error: string | null;
}

type Action =
    | { type: 'SET_LOADING'; payload: boolean }
    | { type: 'SET_ERROR'; payload: string | null }
    | { type: 'SET_CURRENT_USER'; payload: User | null }
    | { type: 'ADD_USER'; payload: User }
    | { type: 'UPDATE_USER'; payload: { id: number; updates: Partial<User> } }
    | { type: 'REMOVE_USER'; payload: number }
    | { type: 'ADD_PRODUCT'; payload: Product }
    | { type: 'UPDATE_PRODUCT'; payload: { id: number; updates: Partial<Product> } }
    | { type: 'REMOVE_PRODUCT'; payload: number };

type Reducer<S, A> = (state: S, action: A) => S;

const initialState: State = {
    users: [],
    products: [],
    currentUser: null,
    loading: false,
    error: null
};

const reducer: Reducer<State, Action> = (state, action) => {
    switch (action.type) {
        case 'SET_LOADING':
            return { ...state, loading: action.payload };

        case 'SET_ERROR':
            return { ...state, error: action.payload };

        case 'SET_CURRENT_USER':
            return { ...state, currentUser: action.payload };

        case 'ADD_USER':
            return { ...state, users: [...state.users, action.payload] };

        case 'UPDATE_USER':
            return {
                ...state,
                users: state.users.map(user =>
                    user.id === action.payload.id
                        ? { ...user, ...action.payload.updates }
                        : user
                )
            };

        case 'REMOVE_USER':
            return {
                ...state,
                users: state.users.filter(user => user.id !== action.payload)
            };

        case 'ADD_PRODUCT':
            return { ...state, products: [...state.products, action.payload] };

        case 'UPDATE_PRODUCT':
            return {
                ...state,
                products: state.products.map(product =>
                    product.id === action.payload.id
                        ? { ...product, ...action.payload.updates }
                        : product
                )
            };

        case 'REMOVE_PRODUCT':
            return {
                ...state,
                products: state.products.filter(product => product.id !== action.payload)
            };

        default:
            return state;
    }
};

class Store<S, A> {
    private state: S;
    private listeners: Set<(state: S) => void> = new Set();

    constructor(
        private reducer: Reducer<S, A>,
        initialState: S
    ) {
        this.state = initialState;
    }

    getState(): S {
        return this.state;
    }

    dispatch(action: A): void {
        this.state = this.reducer(this.state, action);
        this.listeners.forEach(listener => listener(this.state));
    }

    subscribe(listener: (state: S) => void): () => void {
        this.listeners.add(listener);
        return () => {
            this.listeners.delete(listener);
        };
    }
}

// ========== Main Demo Functions ==========

async function demonstrateBasicTypes(): Promise<void> {
    console.log('=== Basic TypeScript Types Demo ===');

    const userFactory = new UserFactory();
    const user = userFactory.create('John Doe', 'john@example.com', ['admin']);

    console.log('Created user:', user);
    console.log('Is valid user:', isUser(user));

    // Type narrowing
    function processEntity(entity: User | Product) {
        if (isUser(entity)) {
            console.log(`Processing user: ${entity.name} (${entity.email})`);
        } else if (isProduct(entity)) {
            console.log(`Processing product: ${entity.name} ($${entity.price})`);
        }
    }

    processEntity(user);
}

async function demonstrateGenerics(): Promise<void> {
    console.log('\n=== Generics Demo ===');

    const userRepo = new InMemoryRepository<User & BaseEntity>();
    const productRepo = new InMemoryRepository<Product>();

    // Create users
    const user1 = await userRepo.create({
        name: 'Alice Smith',
        email: 'alice@example.com',
        roles: ['user'],
        metadata: {}
    });

    const user2 = await userRepo.create({
        name: 'Bob Johnson',
        email: 'bob@example.com',
        roles: ['admin'],
        metadata: { department: 'IT' }
    });

    console.log('Created users:', await userRepo.findAll());

    // Create products
    const product1 = await productRepo.create({
        name: 'Laptop',
        price: 999.99,
        category: 'Electronics',
        inStock: true
    });

    console.log('Created product:', product1);
    console.log('Total products:', await productRepo.count());
}

async function demonstrateAsyncPatterns(): Promise<void> {
    console.log('\n=== Async Patterns Demo ===');

    const httpClient = new HttpClient('https://api.example.com');

    // Simulate API calls
    const response = await httpClient.get<User>('/users/1');
    console.log('API Response status:', response.status);

    // Async iteration
    const data = Array.from({ length: 20 }, (_, i) => ({ id: i + 1, value: `Item ${i + 1}` }));
    const stream = new AsyncDataStream(data, 5);

    console.log('Processing async stream:');
    let batchNumber = 1;
    await stream.forEach(async (batch) => {
        console.log(`  Batch ${batchNumber++}: ${batch.length} items`);
    });
}

async function demonstrateEventSystem(): Promise<void> {
    console.log('\n=== Event System Demo ===');

    const eventEmitter = new EventEmitter<EventMap>();

    // Set up listeners
    const unsubscribeUser = eventEmitter.on('user:created', ({ user }) => {
        console.log(`User created: ${user.name}`);
    });

    eventEmitter.on('user:updated', ({ user, changes }) => {
        console.log(`User updated: ${user.name}`, changes);
    });

    eventEmitter.on('product:created', ({ product }) => {
        console.log(`Product created: ${product.name} ($${product.price})`);
    });

    // Emit events
    const user = new UserFactory().create('Jane Doe', 'jane@example.com');
    eventEmitter.emit('user:created', { user });

    eventEmitter.emit('user:updated', {
        user: { ...user, name: 'Jane Smith' },
        changes: { name: 'Jane Smith' }
    });

    eventEmitter.emit('product:created', {
        product: {
            id: 1,
            name: 'Smartphone',
            price: 699.99,
            category: 'Electronics',
            inStock: true,
            createdAt: new Date(),
            updatedAt: new Date()
        }
    });

    // Clean up
    unsubscribeUser();
}

async function demonstrateStateManagement(): Promise<void> {
    console.log('\n=== State Management Demo ===');

    const store = new Store(reducer, initialState);

    // Subscribe to state changes
    const unsubscribe = store.subscribe((state) => {
        console.log('State updated. Users:', state.users.length, 'Products:', state.products.length);
    });

    // Dispatch actions
    store.dispatch({ type: 'SET_LOADING', payload: true });

    const user = new UserFactory().create('Admin User', 'admin@example.com', ['admin']);
    store.dispatch({ type: 'ADD_USER', payload: user });

    store.dispatch({
        type: 'ADD_PRODUCT',
        payload: {
            id: 1,
            name: 'Tablet',
            price: 299.99,
            category: 'Electronics',
            inStock: true,
            createdAt: new Date(),
            updatedAt: new Date()
        }
    });

    store.dispatch({ type: 'SET_LOADING', payload: false });

    console.log('Final state:', store.getState());

    // Clean up
    unsubscribe();
}

async function demonstrateUtilities(): Promise<void> {
    console.log('\n=== Utility Functions Demo ===');

    const user = new UserFactory().create('Test User', 'test@example.com', ['user', 'tester']);

    // Pick specific properties
    const userSummary = pick(user, ['id', 'name', 'email']);
    console.log('User summary:', userSummary);

    // Omit sensitive properties
    const publicUser = omit(user, ['metadata']);
    console.log('Public user:', publicUser);

    // Deep clone
    const clonedUser = deepClone(user);
    clonedUser.name = 'Cloned User';
    console.log('Original user name:', user.name);
    console.log('Cloned user name:', clonedUser.name);
}

// ========== Main Execution ==========

async function main(): Promise<void> {
    console.log('=== Comprehensive TypeScript Examples ===\n');

    await demonstrateBasicTypes();
    await demonstrateGenerics();
    await demonstrateAsyncPatterns();
    await demonstrateEventSystem();
    await demonstrateStateManagement();
    await demonstrateUtilities();

    console.log('\n=== TypeScript Features Demonstrated ===');
    console.log('- Advanced type system (unions, intersections, conditional types)');
    console.log('- Generic programming patterns');
    console.log('- Decorators for cross-cutting concerns');
    console.log('- Abstract classes and interfaces');
    console.log('- Type guards and type narrowing');
    console.log('- Async/await and Promise patterns');
    console.log('- Event-driven architecture');
    console.log('- State management patterns');
    console.log('- Utility types and functions');
    console.log('- Repository and service patterns');
}

// Run if this is the main module
if (typeof require !== 'undefined' && require.main === module) {
    main().catch(console.error);
}

export {
    User,
    Product,
    Repository,
    InMemoryRepository,
    Service,
    ProductService,
    HttpClient,
    AsyncDataStream,
    EventEmitter,
    Store,
    isUser,
    isProduct,
    pick,
    omit,
    deepClone
};