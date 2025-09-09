// Comprehensive JavaScript Modern Web Development Examples
// Demonstrates ES2023+ features, async patterns, and full-stack concepts

// ============ Modern JavaScript Features ============

/**
 * Using ES2022+ class fields and private methods
 */
class User {
    // Private fields
    #id;
    #email;
    #createdAt;
    
    // Public fields with default values
    firstName = '';
    lastName = '';
    role = 'user';
    isActive = true;
    preferences = {
        theme: 'auto',
        language: 'en',
        notifications: true
    };
    
    constructor(firstName, lastName, email, role = 'user') {
        this.#id = crypto.randomUUID();
        this.firstName = firstName;
        this.lastName = lastName;
        this.#email = this.#validateEmail(email);
        this.role = role;
        this.#createdAt = new Date();
    }
    
    // Getter/setter for private fields
    get id() { return this.#id; }
    get email() { return this.#email; }
    get createdAt() { return this.#createdAt; }
    
    set email(value) {
        this.#email = this.#validateEmail(value);
    }
    
    // Computed properties
    get fullName() {
        return `${this.firstName} ${this.lastName}`.trim();
    }
    
    get age() {
        if (!this.dateOfBirth) return null;
        const today = new Date();
        const birthDate = new Date(this.dateOfBirth);
        let age = today.getFullYear() - birthDate.getFullYear();
        const monthDiff = today.getMonth() - birthDate.getMonth();
        
        if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
            age--;
        }
        
        return age;
    }
    
    get isAdult() {
        return this.age !== null && this.age >= 18;
    }
    
    // Private validation method
    #validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            throw new Error('Invalid email format');
        }
        return email.toLowerCase();
    }
    
    // Public methods
    updateProfile({ firstName, lastName, dateOfBirth, preferences }) {
        if (firstName !== undefined) this.firstName = firstName;
        if (lastName !== undefined) this.lastName = lastName;
        if (dateOfBirth !== undefined) this.dateOfBirth = dateOfBirth;
        if (preferences !== undefined) {
            this.preferences = { ...this.preferences, ...preferences };
        }
        return this;
    }
    
    hasPermission(permission) {
        const rolePermissions = {
            guest: ['read'],
            user: ['read', 'write:own'],
            moderator: ['read', 'write:own', 'moderate'],
            admin: ['read', 'write:all', 'delete:all', 'manage:users']
        };
        
        return this.isActive && (rolePermissions[this.role] || []).includes(permission);
    }
    
    toJSON() {
        return {
            id: this.#id,
            firstName: this.firstName,
            lastName: this.lastName,
            fullName: this.fullName,
            email: this.#email,
            role: this.role,
            isActive: this.isActive,
            age: this.age,
            isAdult: this.isAdult,
            preferences: this.preferences,
            createdAt: this.#createdAt.toISOString()
        };
    }
    
    // Static factory methods
    static fromJSON(data) {
        const user = new User(data.firstName, data.lastName, data.email, data.role);
        user.isActive = data.isActive ?? true;
        user.preferences = { ...user.preferences, ...data.preferences };
        if (data.dateOfBirth) user.dateOfBirth = data.dateOfBirth;
        return user;
    }
    
    static createGuest() {
        return new User('Guest', 'User', 'guest@example.com', 'guest');
    }
}

/**
 * Task class with modern features
 */
class Task {
    #id;
    #createdAt;
    #updatedAt;
    
    title = '';
    description = '';
    status = 'draft';
    priority = 'medium';
    assigneeId = null;
    creatorId = null;
    dueDate = null;
    completedAt = null;
    tags = [];
    
    constructor(title, description, priority = 'medium', assigneeId, creatorId) {
        this.#id = crypto.randomUUID();
        this.title = title;
        this.description = description;
        this.priority = priority;
        this.assigneeId = assigneeId;
        this.creatorId = creatorId;
        this.#createdAt = new Date();
        this.#updatedAt = new Date();
        
        this.#validate();
    }
    
    get id() { return this.#id; }
    get createdAt() { return this.#createdAt; }
    get updatedAt() { return this.#updatedAt; }
    
    get isOverdue() {
        return this.dueDate && 
               new Date() > new Date(this.dueDate) && 
               this.status !== 'completed' && 
               this.status !== 'cancelled';
    }
    
    get isCompleted() {
        return this.status === 'completed';
    }
    
    get progressPercentage() {
        const statusProgress = {
            draft: 0,
            active: 10,
            'in-progress': 50,
            review: 80,
            completed: 100,
            cancelled: 0
        };
        return statusProgress[this.status] || 0;
    }
    
    get timeToCompletion() {
        if (!this.completedAt) return null;
        return new Date(this.completedAt) - this.#createdAt;
    }
    
    #validate() {
        if (!this.title?.trim()) {
            throw new Error('Task title is required');
        }
        if (this.title.length > 200) {
            throw new Error('Title cannot exceed 200 characters');
        }
        if (this.description?.length > 2000) {
            throw new Error('Description cannot exceed 2000 characters');
        }
        if (!['low', 'medium', 'high', 'urgent'].includes(this.priority)) {
            throw new Error('Invalid priority level');
        }
    }
    
    #touch() {
        this.#updatedAt = new Date();
    }
    
    updateDetails({ title, description, priority, dueDate }) {
        if (title !== undefined) this.title = title;
        if (description !== undefined) this.description = description;
        if (priority !== undefined) this.priority = priority;
        if (dueDate !== undefined) this.dueDate = dueDate;
        
        this.#validate();
        this.#touch();
        return this;
    }
    
    changeStatus(newStatus) {
        const validTransitions = {
            draft: ['active', 'cancelled'],
            active: ['in-progress', 'cancelled'],
            'in-progress': ['review', 'active', 'cancelled'],
            review: ['completed', 'in-progress', 'cancelled'],
            completed: ['active'],
            cancelled: ['active']
        };
        
        if (!validTransitions[this.status]?.includes(newStatus)) {
            throw new Error(`Cannot transition from ${this.status} to ${newStatus}`);
        }
        
        this.status = newStatus;
        
        if (newStatus === 'completed') {
            this.completedAt = new Date();
        } else if (this.status === 'completed') {
            this.completedAt = null;
        }
        
        this.#touch();
        return this;
    }
    
    addTag(tag) {
        const normalizedTag = tag.trim().toLowerCase();
        if (normalizedTag && !this.tags.includes(normalizedTag)) {
            this.tags.push(normalizedTag);
            this.#touch();
        }
        return this;
    }
    
    removeTag(tag) {
        const normalizedTag = tag.trim().toLowerCase();
        const index = this.tags.indexOf(normalizedTag);
        if (index > -1) {
            this.tags.splice(index, 1);
            this.#touch();
        }
        return this;
    }
    
    toJSON() {
        return {
            id: this.#id,
            title: this.title,
            description: this.description,
            status: this.status,
            priority: this.priority,
            assigneeId: this.assigneeId,
            creatorId: this.creatorId,
            dueDate: this.dueDate,
            completedAt: this.completedAt,
            tags: [...this.tags],
            isOverdue: this.isOverdue,
            isCompleted: this.isCompleted,
            progressPercentage: this.progressPercentage,
            timeToCompletion: this.timeToCompletion,
            createdAt: this.#createdAt.toISOString(),
            updatedAt: this.#updatedAt.toISOString()
        };
    }
}

// ============ Async Patterns and Promises ============

/**
 * Advanced Promise utilities
 */
class PromiseUtils {
    static async retry(fn, maxAttempts = 3, delay = 1000) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                if (attempt < maxAttempts) {
                    await this.delay(delay * Math.pow(2, attempt - 1)); // Exponential backoff
                }
            }
        }
        
        throw new Error(`Failed after ${maxAttempts} attempts: ${lastError.message}`);
    }
    
    static delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    static timeout(promise, ms) {
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`Operation timed out after ${ms}ms`)), ms);
        });
        
        return Promise.race([promise, timeoutPromise]);
    }
    
    static async batch(items, processor, batchSize = 5) {
        const results = [];
        
        for (let i = 0; i < items.length; i += batchSize) {
            const batch = items.slice(i, i + batchSize);
            const batchResults = await Promise.all(batch.map(processor));
            results.push(...batchResults);
        }
        
        return results;
    }
    
    static async sequence(items, processor) {
        const results = [];
        
        for (const item of items) {
            const result = await processor(item);
            results.push(result);
        }
        
        return results;
    }
    
    static allSettled(promises) {
        return Promise.allSettled(promises);
    }
    
    static async waterfall(...functions) {
        let result;
        
        for (const fn of functions) {
            result = await fn(result);
        }
        
        return result;
    }
}

/**
 * Observable implementation for reactive programming
 */
class Observable {
    constructor(subscriber) {
        this.subscriber = subscriber;
    }
    
    static create(subscriber) {
        return new Observable(subscriber);
    }
    
    static fromArray(array) {
        return new Observable(observer => {
            array.forEach(item => observer.next(item));
            observer.complete();
        });
    }
    
    static fromPromise(promise) {
        return new Observable(observer => {
            promise
                .then(value => {
                    observer.next(value);
                    observer.complete();
                })
                .catch(error => observer.error(error));
        });
    }
    
    static interval(ms) {
        return new Observable(observer => {
            let count = 0;
            const intervalId = setInterval(() => {
                observer.next(count++);
            }, ms);
            
            return () => clearInterval(intervalId);
        });
    }
    
    subscribe(observer) {
        const unsubscribe = this.subscriber(observer);
        return { unsubscribe: unsubscribe || (() => {}) };
    }
    
    map(transform) {
        return new Observable(observer => {
            return this.subscribe({
                next: value => observer.next(transform(value)),
                error: err => observer.error(err),
                complete: () => observer.complete()
            });
        });
    }
    
    filter(predicate) {
        return new Observable(observer => {
            return this.subscribe({
                next: value => {
                    if (predicate(value)) {
                        observer.next(value);
                    }
                },
                error: err => observer.error(err),
                complete: () => observer.complete()
            });
        });
    }
    
    take(count) {
        return new Observable(observer => {
            let taken = 0;
            
            return this.subscribe({
                next: value => {
                    if (taken < count) {
                        observer.next(value);
                        taken++;
                        
                        if (taken === count) {
                            observer.complete();
                        }
                    }
                },
                error: err => observer.error(err),
                complete: () => observer.complete()
            });
        });
    }
    
    debounce(ms) {
        return new Observable(observer => {
            let timeoutId;
            
            return this.subscribe({
                next: value => {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => observer.next(value), ms);
                },
                error: err => observer.error(err),
                complete: () => observer.complete()
            });
        });
    }
}

// ============ State Management ============

/**
 * Simple state manager with reactive updates
 */
class StateManager {
    #state = {};
    #listeners = new Map();
    
    constructor(initialState = {}) {
        this.#state = { ...initialState };
    }
    
    getState() {
        return { ...this.#state };
    }
    
    setState(updates) {
        const prevState = { ...this.#state };
        this.#state = { ...this.#state, ...updates };
        
        // Notify listeners
        for (const [key, listeners] of this.#listeners) {
            if (key in updates) {
                listeners.forEach(listener => {
                    listener(this.#state[key], prevState[key]);
                });
            }
        }
        
        // Notify global listeners
        const globalListeners = this.#listeners.get('*') || [];
        globalListeners.forEach(listener => {
            listener(this.#state, prevState);
        });
    }
    
    subscribe(key, listener) {
        if (!this.#listeners.has(key)) {
            this.#listeners.set(key, []);
        }
        
        this.#listeners.get(key).push(listener);
        
        // Return unsubscribe function
        return () => {
            const listeners = this.#listeners.get(key);
            const index = listeners.indexOf(listener);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        };
    }
    
    // Helper method for reactive state updates
    createSelector(selector) {
        let previousValue = selector(this.#state);
        
        return {
            getValue: () => selector(this.#state),
            subscribe: (listener) => {
                return this.subscribe('*', (newState) => {
                    const newValue = selector(newState);
                    if (newValue !== previousValue) {
                        const oldValue = previousValue;
                        previousValue = newValue;
                        listener(newValue, oldValue);
                    }
                });
            }
        };
    }
}

// ============ HTTP Client with Advanced Features ============

class HttpClient {
    constructor(baseURL = '', defaultHeaders = {}) {
        this.baseURL = baseURL;
        this.defaultHeaders = defaultHeaders;
        this.interceptors = {
            request: [],
            response: []
        };
    }
    
    addRequestInterceptor(interceptor) {
        this.interceptors.request.push(interceptor);
    }
    
    addResponseInterceptor(interceptor) {
        this.interceptors.response.push(interceptor);
    }
    
    async request(url, options = {}) {
        let config = {
            method: 'GET',
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };
        
        const fullURL = url.startsWith('http') ? url : `${this.baseURL}${url}`;
        
        // Apply request interceptors
        for (const interceptor of this.interceptors.request) {
            config = await interceptor(config);
        }
        
        try {
            let response = await fetch(fullURL, config);
            
            // Apply response interceptors
            for (const interceptor of this.interceptors.response) {
                response = await interceptor(response);
            }
            
            if (!response.ok) {
                throw new HttpError(response.status, response.statusText, response);
            }
            
            return response;
        } catch (error) {
            if (error instanceof HttpError) {
                throw error;
            }
            throw new HttpError(0, 'Network Error', null, error);
        }
    }
    
    async get(url, options = {}) {
        return this.request(url, { ...options, method: 'GET' });
    }
    
    async post(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'POST',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
    }
    
    async put(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'PUT',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
    }
    
    async delete(url, options = {}) {
        return this.request(url, { ...options, method: 'DELETE' });
    }
    
    // Helper methods for common response formats
    async json(url, options = {}) {
        const response = await this.get(url, options);
        return response.json();
    }
    
    async text(url, options = {}) {
        const response = await this.get(url, options);
        return response.text();
    }
}

class HttpError extends Error {
    constructor(status, statusText, response, originalError = null) {
        super(`HTTP ${status}: ${statusText}`);
        this.name = 'HttpError';
        this.status = status;
        this.statusText = statusText;
        this.response = response;
        this.originalError = originalError;
    }
}

// ============ Repository Pattern ============

class Repository {
    constructor(httpClient, basePath) {
        this.http = httpClient;
        this.basePath = basePath;
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }
    
    async findAll(options = {}) {
        const cacheKey = `findAll:${JSON.stringify(options)}`;
        
        if (!options.skipCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        const queryString = new URLSearchParams(options).toString();
        const url = queryString ? `${this.basePath}?${queryString}` : this.basePath;
        
        const response = await this.http.get(url);
        const data = await response.json();
        
        this.cache.set(cacheKey, { data, timestamp: Date.now() });
        return data;
    }
    
    async findById(id, options = {}) {
        const cacheKey = `findById:${id}`;
        
        if (!options.skipCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        const response = await this.http.get(`${this.basePath}/${id}`);
        const data = await response.json();
        
        this.cache.set(cacheKey, { data, timestamp: Date.now() });
        return data;
    }
    
    async create(item) {
        const response = await this.http.post(this.basePath, item);
        const data = await response.json();
        
        // Invalidate relevant cache entries
        this.#invalidateCache();
        
        return data;
    }
    
    async update(id, updates) {
        const response = await this.http.put(`${this.basePath}/${id}`, updates);
        const data = await response.json();
        
        // Update cache
        this.cache.set(`findById:${id}`, { data, timestamp: Date.now() });
        this.#invalidateListCache();
        
        return data;
    }
    
    async delete(id) {
        await this.http.delete(`${this.basePath}/${id}`);
        
        // Remove from cache
        this.cache.delete(`findById:${id}`);
        this.#invalidateListCache();
        
        return true;
    }
    
    #invalidateCache() {
        this.cache.clear();
    }
    
    #invalidateListCache() {
        for (const key of this.cache.keys()) {
            if (key.startsWith('findAll:')) {
                this.cache.delete(key);
            }
        }
    }
}

// ============ Service Layer ============

class UserService {
    constructor(userRepository) {
        this.repository = userRepository;
        this.stateManager = new StateManager({
            users: [],
            currentUser: null,
            loading: false,
            error: null
        });
    }
    
    async createUser(userData) {
        try {
            this.stateManager.setState({ loading: true, error: null });
            
            const user = await this.repository.create(userData);
            
            const currentUsers = this.stateManager.getState().users;
            this.stateManager.setState({
                users: [...currentUsers, user],
                loading: false
            });
            
            return user;
        } catch (error) {
            this.stateManager.setState({
                loading: false,
                error: error.message
            });
            throw error;
        }
    }
    
    async getUsers(options = {}) {
        try {
            this.stateManager.setState({ loading: true, error: null });
            
            const users = await this.repository.findAll(options);
            
            this.stateManager.setState({
                users,
                loading: false
            });
            
            return users;
        } catch (error) {
            this.stateManager.setState({
                loading: false,
                error: error.message
            });
            throw error;
        }
    }
    
    async getUserById(id) {
        try {
            const user = await this.repository.findById(id);
            return user;
        } catch (error) {
            this.stateManager.setState({ error: error.message });
            throw error;
        }
    }
    
    async updateUser(id, updates) {
        try {
            this.stateManager.setState({ loading: true, error: null });
            
            const updatedUser = await this.repository.update(id, updates);
            
            const currentUsers = this.stateManager.getState().users;
            const userIndex = currentUsers.findIndex(user => user.id === id);
            
            if (userIndex > -1) {
                const newUsers = [...currentUsers];
                newUsers[userIndex] = updatedUser;
                
                this.stateManager.setState({
                    users: newUsers,
                    loading: false
                });
            }
            
            return updatedUser;
        } catch (error) {
            this.stateManager.setState({
                loading: false,
                error: error.message
            });
            throw error;
        }
    }
    
    async deleteUser(id) {
        try {
            this.stateManager.setState({ loading: true, error: null });
            
            await this.repository.delete(id);
            
            const currentUsers = this.stateManager.getState().users;
            const filteredUsers = currentUsers.filter(user => user.id !== id);
            
            this.stateManager.setState({
                users: filteredUsers,
                loading: false
            });
            
            return true;
        } catch (error) {
            this.stateManager.setState({
                loading: false,
                error: error.message
            });
            throw error;
        }
    }
    
    // Reactive state access
    subscribeToUsers(callback) {
        return this.stateManager.subscribe('users', callback);
    }
    
    subscribeToLoading(callback) {
        return this.stateManager.subscribe('loading', callback);
    }
    
    subscribeToError(callback) {
        return this.stateManager.subscribe('error', callback);
    }
    
    getState() {
        return this.stateManager.getState();
    }
}

// ============ Functional Programming Utilities ============

const FP = {
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
    
    // Composition
    compose: (...fns) => (value) => fns.reduceRight((acc, fn) => fn(acc), value),
    
    pipe: (...fns) => (value) => fns.reduce((acc, fn) => fn(acc), value),
    
    // Partial application
    partial: (fn, ...args1) => (...args2) => fn(...args1, ...args2),
    
    // Higher-order functions
    map: FP.curry((fn, array) => array.map(fn)),
    
    filter: FP.curry((predicate, array) => array.filter(predicate)),
    
    reduce: FP.curry((reducer, initial, array) => array.reduce(reducer, initial)),
    
    // Utility functions
    identity: (x) => x,
    
    constant: (value) => () => value,
    
    not: (fn) => (...args) => !fn(...args),
    
    // Array utilities
    head: (array) => array[0],
    
    tail: (array) => array.slice(1),
    
    take: FP.curry((n, array) => array.slice(0, n)),
    
    drop: FP.curry((n, array) => array.slice(n)),
    
    // Object utilities
    prop: FP.curry((key, object) => object[key]),
    
    pluck: FP.curry((key, array) => array.map(obj => obj[key])),
    
    groupBy: FP.curry((keyFn, array) => {
        return array.reduce((groups, item) => {
            const key = keyFn(item);
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(item);
            return groups;
        }, {});
    }),
    
    // Maybe monad implementation
    Maybe: {
        of: (value) => ({
            map: (fn) => value != null ? FP.Maybe.of(fn(value)) : FP.Maybe.nothing(),
            flatMap: (fn) => value != null ? fn(value) : FP.Maybe.nothing(),
            filter: (predicate) => value != null && predicate(value) ? FP.Maybe.of(value) : FP.Maybe.nothing(),
            getOrElse: (defaultValue) => value != null ? value : defaultValue,
            isNothing: () => value == null,
            isSomething: () => value != null
        }),
        
        nothing: () => ({
            map: () => FP.Maybe.nothing(),
            flatMap: () => FP.Maybe.nothing(),
            filter: () => FP.Maybe.nothing(),
            getOrElse: (defaultValue) => defaultValue,
            isNothing: () => true,
            isSomething: () => false
        })
    }
};

// ============ Event System ============

class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
        
        // Return unsubscribe function
        return () => this.off(event, listener);
    }
    
    once(event, listener) {
        const unsubscribe = this.on(event, (...args) => {
            listener(...args);
            unsubscribe();
        });
        return unsubscribe;
    }
    
    off(event, listener) {
        if (!this.events[event]) return;
        
        const index = this.events[event].indexOf(listener);
        if (index > -1) {
            this.events[event].splice(index, 1);
        }
    }
    
    emit(event, ...args) {
        if (!this.events[event]) return false;
        
        this.events[event].forEach(listener => {
            try {
                listener(...args);
            } catch (error) {
                console.error('Error in event listener:', error);
            }
        });
        
        return true;
    }
    
    removeAllListeners(event) {
        if (event) {
            delete this.events[event];
        } else {
            this.events = {};
        }
    }
    
    listenerCount(event) {
        return this.events[event] ? this.events[event].length : 0;
    }
    
    eventNames() {
        return Object.keys(this.events);
    }
}

// ============ Validation ============

const Validators = {
    required: (value) => {
        if (value == null || value === '') {
            return 'This field is required';
        }
        return null;
    },
    
    email: (value) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (value && !emailRegex.test(value)) {
            return 'Invalid email format';
        }
        return null;
    },
    
    minLength: (min) => (value) => {
        if (value && value.length < min) {
            return `Must be at least ${min} characters long`;
        }
        return null;
    },
    
    maxLength: (max) => (value) => {
        if (value && value.length > max) {
            return `Cannot exceed ${max} characters`;
        }
        return null;
    },
    
    pattern: (regex, message) => (value) => {
        if (value && !regex.test(value)) {
            return message || 'Invalid format';
        }
        return null;
    },
    
    number: (value) => {
        if (value && isNaN(Number(value))) {
            return 'Must be a valid number';
        }
        return null;
    },
    
    min: (minimum) => (value) => {
        const num = Number(value);
        if (!isNaN(num) && num < minimum) {
            return `Must be at least ${minimum}`;
        }
        return null;
    },
    
    max: (maximum) => (value) => {
        const num = Number(value);
        if (!isNaN(num) && num > maximum) {
            return `Cannot exceed ${maximum}`;
        }
        return null;
    }
};

function createValidator(rules) {
    return (data) => {
        const errors = {};
        
        for (const [field, fieldRules] of Object.entries(rules)) {
            const value = data[field];
            const fieldErrors = [];
            
            for (const rule of fieldRules) {
                const error = rule(value);
                if (error) {
                    fieldErrors.push(error);
                }
            }
            
            if (fieldErrors.length > 0) {
                errors[field] = fieldErrors;
            }
        }
        
        return {
            isValid: Object.keys(errors).length === 0,
            errors
        };
    };
}

// ============ Demo Application ============

async function runDemo() {
    console.log('=== JavaScript Modern Development Examples Demo ===\n');
    
    // Demo 1: Modern Classes
    console.log('=== Modern Classes Demo ===');
    try {
        const user = new User('Alice', 'Johnson', 'alice@example.com', 'user');
        console.log('Created user:', user.fullName);
        console.log('User JSON:', JSON.stringify(user.toJSON(), null, 2));
        
        user.updateProfile({
            dateOfBirth: '1990-05-15',
            preferences: { theme: 'dark', notifications: false }
        });
        
        console.log('Updated user age:', user.age);
        console.log('Is adult:', user.isAdult);
        console.log('Has permission to write:', user.hasPermission('write:own'));
        
        const task = new Task('Complete project', 'Finish the web application', 'high', user.id, user.id);
        task.addTag('frontend').addTag('urgent');
        task.changeStatus('active');
        task.changeStatus('in-progress');
        
        console.log('Created task:', task.title);
        console.log('Task progress:', task.progressPercentage + '%');
        console.log('Task tags:', task.tags);
    } catch (error) {
        console.error('Error in classes demo:', error.message);
    }
    
    console.log('\n=== Async Utilities Demo ===');
    
    // Demo 2: Promise Utilities
    const simulateApiCall = (delay = 1000, shouldFail = false) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (shouldFail) {
                    reject(new Error('API call failed'));
                } else {
                    resolve({ data: 'Success!', timestamp: Date.now() });
                }
            }, delay);
        });
    };
    
    try {
        // Retry example
        const retryResult = await PromiseUtils.retry(
            () => simulateApiCall(100, Math.random() > 0.7),
            3,
            200
        );
        console.log('Retry successful:', retryResult.data);
        
        // Batch processing
        const items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        const batchResults = await PromiseUtils.batch(
            items,
            async (item) => {
                await PromiseUtils.delay(50);
                return item * 2;
            },
            3
        );
        console.log('Batch results:', batchResults);
        
        // Timeout example
        try {
            await PromiseUtils.timeout(simulateApiCall(2000), 1000);
        } catch (error) {
            console.log('Timeout caught:', error.message);
        }
        
    } catch (error) {
        console.error('Error in async demo:', error.message);
    }
    
    console.log('\n=== Observable Demo ===');
    
    // Demo 3: Observables
    const numberStream = Observable.fromArray([1, 2, 3, 4, 5])
        .map(x => x * 2)
        .filter(x => x > 4);
    
    numberStream.subscribe({
        next: value => console.log('Observable value:', value),
        complete: () => console.log('Observable completed')
    });
    
    // Interval observable (commented out to avoid infinite loop in demo)
    // const intervalSub = Observable.interval(1000)
    //     .take(3)
    //     .subscribe({
    //         next: value => console.log('Interval:', value),
    //         complete: () => console.log('Interval completed')
    //     });
    
    console.log('\n=== State Management Demo ===');
    
    // Demo 4: State Management
    const appState = new StateManager({
        counter: 0,
        user: null,
        theme: 'light'
    });
    
    // Subscribe to counter changes
    const unsubscribeCounter = appState.subscribe('counter', (newValue, oldValue) => {
        console.log(`Counter changed from ${oldValue} to ${newValue}`);
    });
    
    // Create a selector for derived state
    const counterSelector = appState.createSelector(state => state.counter * 10);
    counterSelector.subscribe((newValue, oldValue) => {
        console.log(`Counter x10 changed from ${oldValue} to ${newValue}`);
    });
    
    appState.setState({ counter: 1 });
    appState.setState({ counter: 2 });
    appState.setState({ theme: 'dark' }); // Won't trigger counter listener
    
    unsubscribeCounter();
    appState.setState({ counter: 3 }); // Won't trigger the unsubscribed listener
    
    console.log('\n=== Functional Programming Demo ===');
    
    // Demo 5: Functional Programming
    const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Using curried functions
    const isEven = x => x % 2 === 0;
    const double = x => x * 2;
    const sum = (a, b) => a + b;
    
    const filterEvens = FP.filter(isEven);
    const mapDouble = FP.map(double);
    const sumAll = FP.reduce(sum, 0);
    
    // Function composition
    const processNumbers = FP.pipe(
        filterEvens,
        mapDouble,
        sumAll
    );
    
    const result = processNumbers(numbers);
    console.log('Functional pipeline result:', result); // Even numbers doubled and summed
    
    // Maybe monad example
    const maybeValue = FP.Maybe.of(10)
        .map(x => x * 2)
        .filter(x => x > 15)
        .map(x => x + 5);
    
    console.log('Maybe result:', maybeValue.getOrElse('No value'));
    
    const users = [
        { name: 'Alice', age: 25, department: 'Engineering' },
        { name: 'Bob', age: 30, department: 'Engineering' },
        { name: 'Charlie', age: 35, department: 'Marketing' }
    ];
    
    const groupedByDept = FP.groupBy(FP.prop('department'), users);
    console.log('Grouped by department:', groupedByDept);
    
    console.log('\n=== Validation Demo ===');
    
    // Demo 6: Validation
    const userValidator = createValidator({
        firstName: [Validators.required, Validators.minLength(2)],
        lastName: [Validators.required, Validators.minLength(2)],
        email: [Validators.required, Validators.email],
        age: [Validators.required, Validators.number, Validators.min(0), Validators.max(150)]
    });
    
    const validData = {
        firstName: 'John',
        lastName: 'Doe',
        email: 'john@example.com',
        age: '25'
    };
    
    const invalidData = {
        firstName: '',
        lastName: 'D',
        email: 'invalid-email',
        age: 'not-a-number'
    };
    
    console.log('Valid data validation:', userValidator(validData));
    console.log('Invalid data validation:', userValidator(invalidData));
    
    console.log('\n=== Event System Demo ===');
    
    // Demo 7: Event System
    const eventBus = new EventEmitter();
    
    const unsubscribe1 = eventBus.on('user:created', (user) => {
        console.log('User created event received:', user.name);
    });
    
    const unsubscribe2 = eventBus.once('user:login', (user) => {
        console.log('User login event received (once):', user.name);
    });
    
    eventBus.emit('user:created', { name: 'Alice', id: 1 });
    eventBus.emit('user:login', { name: 'Alice', id: 1 });
    eventBus.emit('user:login', { name: 'Alice', id: 1 }); // Won't trigger the 'once' listener
    
    console.log('Event listeners count for user:created:', eventBus.listenerCount('user:created'));
    
    unsubscribe1();
    unsubscribe2();
    
    console.log('\n=== Features Demonstrated ===');
    console.log('🚀 ES2022+ class fields and private methods');
    console.log('⚡ Advanced Promise utilities and async patterns');
    console.log('🌊 Reactive programming with Observables');
    console.log('🏪 State management with reactive updates');
    console.log('🌐 HTTP client with interceptors and caching');
    console.log('🏗️  Repository pattern with caching');
    console.log('🎯 Service layer with state management');
    console.log('🧮 Functional programming utilities');
    console.log('📡 Event-driven architecture');
    console.log('✅ Comprehensive validation system');
    console.log('🔧 Modern JavaScript features and patterns');
    console.log('📦 Modular and reusable code architecture');
    console.log('🎨 Clean code principles and SOLID design');
    console.log('⚙️  Error handling and resilience patterns');
}

// Run the demo
runDemo().catch(console.error);