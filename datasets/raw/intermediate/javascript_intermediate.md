# Intermediate JavaScript Dataset - Modern ES6+ and Advanced Patterns

## Dataset 1: Advanced ES6+ Features and Modules
```javascript
// ES6+ Classes with advanced features
class EventEmitter {
    #listeners = new Map(); // Private field
    
    constructor(maxListeners = 10) {
        this.maxListeners = maxListeners;
    }
    
    // Method with parameter destructuring
    on(event, listener, { once = false, priority = 0 } = {}) {
        if (!this.#listeners.has(event)) {
            this.#listeners.set(event, []);
        }
        
        const listeners = this.#listeners.get(event);
        if (listeners.length >= this.maxListeners) {
            throw new Error(`Too many listeners for event: ${event}`);
        }
        
        listeners.push({ listener, once, priority });
        listeners.sort((a, b) => b.priority - a.priority);
        
        return this; // Method chaining
    }
    
    emit(event, ...args) {
        const listeners = this.#listeners.get(event);
        if (!listeners) return false;
        
        // Use for...of with destructuring
        for (const { listener, once } of [...listeners]) {
            try {
                listener.apply(this, args);
            } catch (error) {
                console.error(`Error in event listener for ${event}:`, error);
            }
            
            if (once) {
                this.off(event, listener);
            }
        }
        
        return true;
    }
    
    off(event, listenerToRemove) {
        const listeners = this.#listeners.get(event);
        if (!listeners) return this;
        
        const index = listeners.findIndex(({ listener }) => listener === listenerToRemove);
        if (index !== -1) {
            listeners.splice(index, 1);
        }
        
        return this;
    }
    
    // Getter with computed property
    get eventNames() {
        return [...this.#listeners.keys()];
    }
    
    // Static method
    static create(options = {}) {
        return new EventEmitter(options.maxListeners);
    }
}

// Advanced destructuring and spread operators
class DataProcessor {
    constructor(config = {}) {
        // Destructuring with defaults and renaming
        const {
            apiUrl: baseUrl = 'https://api.example.com',
            timeout = 5000,
            headers: defaultHeaders = {},
            ...otherConfig
        } = config;
        
        this.config = {
            baseUrl,
            timeout,
            headers: { 'Content-Type': 'application/json', ...defaultHeaders },
            ...otherConfig
        };
    }
    
    // Method with advanced parameter handling
    async processData(data, {
        transform = x => x,
        filter = () => true,
        sort = null,
        limit = null
    } = {}) {
        let result = [...data]; // Shallow copy using spread
        
        // Pipeline processing using method chaining
        result = result
            .filter(filter)
            .map(transform);
        
        if (sort) {
            result.sort(sort);
        }
        
        if (limit) {
            result = result.slice(0, limit);
        }
        
        return result;
    }
    
    // Generator method
    *batchProcess(data, batchSize = 100) {
        for (let i = 0; i < data.length; i += batchSize) {
            yield data.slice(i, i + batchSize);
        }
    }
    
    // Async generator
    async *fetchDataStream(urls) {
        for (const url of urls) {
            try {
                const response = await fetch(`${this.config.baseUrl}${url}`);
                const data = await response.json();
                yield data;
            } catch (error) {
                yield { error: error.message, url };
            }
        }
    }
}

// Module pattern with advanced exports
export class APIClient extends DataProcessor {
    #authToken = null; // Private field
    
    constructor(config) {
        super(config);
        this.interceptors = {
            request: [],
            response: []
        };
    }
    
    // Fluent interface
    setAuth(token) {
        this.#authToken = token;
        return this;
    }
    
    addRequestInterceptor(interceptor) {
        this.interceptors.request.push(interceptor);
        return this;
    }
    
    addResponseInterceptor(interceptor) {
        this.interceptors.response.push(interceptor);
        return this;
    }
    
    // Advanced async/await with error handling
    async request(endpoint, options = {}) {
        const {
            method = 'GET',
            body = null,
            headers = {},
            ...otherOptions
        } = options;
        
        // Apply request interceptors
        let requestConfig = {
            method,
            headers: {
                ...this.config.headers,
                ...(this.#authToken && { Authorization: `Bearer ${this.#authToken}` }),
                ...headers
            },
            body: body ? JSON.stringify(body) : null,
            ...otherOptions
        };
        
        for (const interceptor of this.interceptors.request) {
            requestConfig = await interceptor(requestConfig);
        }
        
        try {
            let response = await fetch(`${this.config.baseUrl}${endpoint}`, requestConfig);
            
            // Apply response interceptors
            for (const interceptor of this.interceptors.response) {
                response = await interceptor(response);
            }
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }
    
    // Method using template literals and optional chaining
    buildQuery(params = {}) {
        const queryString = Object.entries(params)
            .filter(([_, value]) => value != null)
            .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
            .join('&');
        
        return queryString ? `?${queryString}` : '';
    }
}

// Usage examples
export function demonstrateAdvancedFeatures() {
    console.log('=== Advanced ES6+ Features Demo ===');
    
    // EventEmitter usage
    const emitter = EventEmitter.create({ maxListeners: 5 });
    
    emitter
        .on('data', data => console.log('Handler 1:', data), { priority: 1 })
        .on('data', data => console.log('Handler 2:', data), { priority: 2 })
        .on('error', error => console.error('Error handler:', error), { once: true });
    
    emitter.emit('data', { message: 'Hello World' });
    emitter.emit('error', new Error('Test error'));
    emitter.emit('error', new Error('This won\'t be handled')); // Won't be handled due to 'once'
    
    // DataProcessor usage
    const processor = new DataProcessor({
        apiUrl: 'https://jsonplaceholder.typicode.com',
        timeout: 10000
    });
    
    const sampleData = [
        { id: 1, name: 'Alice', age: 25, department: 'Engineering' },
        { id: 2, name: 'Bob', age: 30, department: 'Marketing' },
        { id: 3, name: 'Carol', age: 28, department: 'Engineering' },
        { id: 4, name: 'David', age: 35, department: 'Sales' }
    ];
    
    // Advanced data processing
    const processedData = processor.processData(sampleData, {
        filter: person => person.department === 'Engineering',
        transform: person => ({ ...person, experience: person.age - 22 }),
        sort: (a, b) => b.age - a.age,
        limit: 10
    });
    
    console.log('Processed data:', processedData);
    
    // Generator usage
    console.log('Batch processing:');
    for (const batch of processor.batchProcess(sampleData, 2)) {
        console.log('Batch:', batch.map(p => p.name));
    }
    
    // API Client usage
    const client = new APIClient({ apiUrl: 'https://jsonplaceholder.typicode.com' })
        .setAuth('sample-token')
        .addRequestInterceptor(async config => {
            console.log('Request interceptor:', config.method, config.url);
            return config;
        })
        .addResponseInterceptor(async response => {
            console.log('Response interceptor:', response.status);
            return response;
        });
    
    // Async demonstration
    client.request('/posts/1')
        .then(data => console.log('API Response:', data))
        .catch(error => console.error('API Error:', error));
}
```

## Dataset 2: Advanced Promises and Async Patterns
```javascript
// Promise utilities and advanced patterns
class PromiseUtils {
    // Promise with timeout
    static timeout(promise, ms, message = 'Operation timed out') {
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error(message)), ms);
        });
        
        return Promise.race([promise, timeoutPromise]);
    }
    
    // Retry with exponential backoff
    static async retry(fn, options = {}) {
        const {
            maxAttempts = 3,
            baseDelay = 1000,
            maxDelay = 10000,
            backoffFactor = 2,
            onRetry = () => {}
        } = options;
        
        let lastError;
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                if (attempt === maxAttempts) {
                    throw error;
                }
                
                const delay = Math.min(
                    baseDelay * Math.pow(backoffFactor, attempt - 1),
                    maxDelay
                );
                
                onRetry(error, attempt, delay);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        
        throw lastError;
    }
    
    // Parallel execution with concurrency limit
    static async parallel(tasks, concurrency = 5) {
        const results = [];
        const executing = [];
        
        for (const [index, task] of tasks.entries()) {
            const promise = Promise.resolve().then(() => task()).then(
                result => ({ status: 'fulfilled', value: result, index }),
                reason => ({ status: 'rejected', reason, index })
            );
            
            results.push(promise);
            
            if (tasks.length >= concurrency) {
                executing.push(promise);
                
                if (executing.length >= concurrency) {
                    await Promise.race(executing);
                    executing.splice(
                        executing.findIndex(p => p === promise), 1
                    );
                }
            }
        }
        
        const settled = await Promise.allSettled(results);
        return settled.map(result => result.value).sort((a, b) => a.index - b.index);
    }
    
    // Circuit breaker pattern
    static createCircuitBreaker(fn, options = {}) {
        const {
            failureThreshold = 5,
            resetTimeout = 60000,
            monitor = () => {}
        } = options;
        
        let state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        let failures = 0;
        let lastFailureTime = null;
        
        return async function circuitBreakerWrapper(...args) {
            monitor(state, failures);
            
            if (state === 'OPEN') {
                if (Date.now() - lastFailureTime >= resetTimeout) {
                    state = 'HALF_OPEN';
                } else {
                    throw new Error('Circuit breaker is OPEN');
                }
            }
            
            try {
                const result = await fn.apply(this, args);
                
                // Reset on success
                if (state === 'HALF_OPEN') {
                    state = 'CLOSED';
                    failures = 0;
                }
                
                return result;
            } catch (error) {
                failures++;
                lastFailureTime = Date.now();
                
                if (failures >= failureThreshold) {
                    state = 'OPEN';
                }
                
                throw error;
            }
        };
    }
    
    // Promise pool for resource management
    static createPromisePool(createResource, destroyResource, poolSize = 5) {
        const pool = [];
        const waiting = [];
        let created = 0;
        
        async function acquire() {
            if (pool.length > 0) {
                return pool.pop();
            }
            
            if (created < poolSize) {
                created++;
                return await createResource();
            }
            
            // Wait for a resource to become available
            return new Promise(resolve => {
                waiting.push(resolve);
            });
        }
        
        function release(resource) {
            if (waiting.length > 0) {
                const resolve = waiting.shift();
                resolve(resource);
            } else {
                pool.push(resource);
            }
        }
        
        async function destroy() {
            while (pool.length > 0) {
                const resource = pool.pop();
                await destroyResource(resource);
                created--;
            }
        }
        
        return { acquire, release, destroy };
    }
}

// Advanced async patterns
class AsyncDataManager {
    constructor() {
        this.cache = new Map();
        this.loading = new Map();
    }
    
    // Async memoization with cache invalidation
    async memoizedFetch(key, fetcher, ttl = 60000) {
        const cached = this.cache.get(key);
        
        if (cached && Date.now() - cached.timestamp < ttl) {
            return cached.data;
        }
        
        // Prevent duplicate requests
        if (this.loading.has(key)) {
            return this.loading.get(key);
        }
        
        const promise = fetcher().then(data => {
            this.cache.set(key, { data, timestamp: Date.now() });
            this.loading.delete(key);
            return data;
        }).catch(error => {
            this.loading.delete(key);
            throw error;
        });
        
        this.loading.set(key, promise);
        return promise;
    }
    
    // Async iterator for paginated data
    async *paginatedFetch(baseUrl, pageSize = 10) {
        let page = 1;
        let hasMore = true;
        
        while (hasMore) {
            try {
                const response = await fetch(`${baseUrl}?page=${page}&limit=${pageSize}`);
                const data = await response.json();
                
                if (data.items && data.items.length > 0) {
                    yield* data.items; // Yield each item
                    page++;
                    hasMore = data.hasMore || data.items.length === pageSize;
                } else {
                    hasMore = false;
                }
            } catch (error) {
                console.error('Pagination fetch error:', error);
                hasMore = false;
            }
        }
    }
    
    // Debounced async function
    createDebouncedAsync(fn, delay = 300) {
        let timeoutId;
        let latestPromise;
        
        return function debouncedAsync(...args) {
            return new Promise((resolve, reject) => {
                clearTimeout(timeoutId);
                
                timeoutId = setTimeout(async () => {
                    try {
                        latestPromise = fn.apply(this, args);
                        const result = await latestPromise;
                        resolve(result);
                    } catch (error) {
                        reject(error);
                    }
                }, delay);
            });
        };
    }
    
    // Queue with priority and async processing
    createAsyncQueue(processor, concurrency = 3) {
        const queue = [];
        let running = 0;
        
        async function processNext() {
            if (queue.length === 0 || running >= concurrency) {
                return;
            }
            
            running++;
            const { task, resolve, reject, priority } = queue.shift();
            
            try {
                const result = await processor(task);
                resolve(result);
            } catch (error) {
                reject(error);
            } finally {
                running--;
                processNext(); // Process next item
            }
        }
        
        function add(task, priority = 0) {
            return new Promise((resolve, reject) => {
                const item = { task, resolve, reject, priority };
                
                // Insert based on priority
                const insertIndex = queue.findIndex(item => item.priority < priority);
                if (insertIndex === -1) {
                    queue.push(item);
                } else {
                    queue.splice(insertIndex, 0, item);
                }
                
                processNext();
            });
        }
        
        return { add };
    }
}

// WebSocket with reconnection and message queuing
class RobustWebSocket extends EventTarget {
    constructor(url, options = {}) {
        super();
        this.url = url;
        this.options = {
            maxReconnectAttempts: 5,
            reconnectInterval: 1000,
            heartbeatInterval: 30000,
            ...options
        };
        
        this.ws = null;
        this.messageQueue = [];
        this.reconnectAttempts = 0;
        this.isConnected = false;
        this.heartbeatTimer = null;
        
        this.connect();
    }
    
    async connect() {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                
                // Send queued messages
                while (this.messageQueue.length > 0) {
                    const message = this.messageQueue.shift();
                    this.ws.send(JSON.stringify(message));
                }
                
                this.startHeartbeat();
                this.dispatchEvent(new CustomEvent('open'));
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.dispatchEvent(new CustomEvent('message', { detail: data }));
                } catch (error) {
                    console.error('Invalid JSON received:', event.data);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                this.stopHeartbeat();
                
                if (!event.wasClean && this.reconnectAttempts < this.options.maxReconnectAttempts) {
                    this.scheduleReconnect();
                }
                
                this.dispatchEvent(new CustomEvent('close', { detail: event }));
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.dispatchEvent(new CustomEvent('error', { detail: error }));
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.scheduleReconnect();
        }
    }
    
    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            this.connect();
        }, delay);
    }
    
    send(message) {
        if (this.isConnected) {
            this.ws.send(JSON.stringify(message));
        } else {
            this.messageQueue.push(message);
        }
    }
    
    startHeartbeat() {
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                this.send({ type: 'ping', timestamp: Date.now() });
            }
        }, this.options.heartbeatInterval);
    }
    
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    close() {
        this.stopHeartbeat();
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Demonstration function
export async function demonstrateAsyncPatterns() {
    console.log('=== Advanced Async Patterns Demo ===');
    
    // Promise utilities demo
    const unreliableTask = () => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                Math.random() > 0.7 ? resolve('Success!') : reject(new Error('Random failure'));
            }, Math.random() * 1000);
        });
    };
    
    // Retry with backoff
    try {
        const result = await PromiseUtils.retry(unreliableTask, {
            maxAttempts: 5,
            baseDelay: 500,
            onRetry: (error, attempt, delay) => {
                console.log(`Retry attempt ${attempt} after ${delay}ms: ${error.message}`);
            }
        });
        console.log('Retry result:', result);
    } catch (error) {
        console.log('Final retry error:', error.message);
    }
    
    // Parallel execution with concurrency limit
    const tasks = Array.from({ length: 10 }, (_, i) => 
        () => new Promise(resolve => setTimeout(() => resolve(`Task ${i}`), Math.random() * 1000))
    );
    
    const results = await PromiseUtils.parallel(tasks, 3);
    console.log('Parallel results:', results.map(r => r.value));
    
    // Async data manager demo
    const dataManager = new AsyncDataManager();
    
    // Memoized fetch
    const fetchUser = (id) => 
        new Promise(resolve => 
            setTimeout(() => resolve({ id, name: `User ${id}` }), 500)
        );
    
    console.time('First fetch');
    const user1 = await dataManager.memoizedFetch('user:1', () => fetchUser(1));
    console.timeEnd('First fetch');
    
    console.time('Cached fetch');
    const user2 = await dataManager.memoizedFetch('user:1', () => fetchUser(1));
    console.timeEnd('Cached fetch');
    
    console.log('Users:', user1, user2);
    
    // Debounced async function
    const debouncedSearch = dataManager.createDebouncedAsync(
        async (query) => {
            console.log(`Searching for: ${query}`);
            return `Results for ${query}`;
        },
        300
    );
    
    // These calls will be debounced
    debouncedSearch('a');
    debouncedSearch('ab');
    const searchResult = await debouncedSearch('abc');
    console.log('Search result:', searchResult);
}
```