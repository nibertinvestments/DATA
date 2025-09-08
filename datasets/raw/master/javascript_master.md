# Master JavaScript Dataset - Enterprise-Level Architecture and Performance

## Dataset 1: Advanced Design Patterns and Architecture
```javascript
// Observer Pattern with Advanced Features
class AdvancedEventEmitter {
    constructor(options = {}) {
        this.events = new Map();
        this.maxListeners = options.maxListeners || 10;
        this.onceEvents = new WeakSet();
        this.asyncListeners = new Set();
        this.middleware = [];
        this.errorHandler = options.errorHandler || this.defaultErrorHandler;
    }
    
    // Middleware system for event processing
    use(middleware) {
        if (typeof middleware !== 'function') {
            throw new TypeError('Middleware must be a function');
        }
        this.middleware.push(middleware);
        return this;
    }
    
    async emit(eventName, ...args) {
        const event = {
            name: eventName,
            data: args,
            timestamp: Date.now(),
            preventDefault: false,
            stopPropagation: false
        };
        
        // Apply middleware
        for (const middleware of this.middleware) {
            try {
                await middleware(event);
                if (event.preventDefault) return false;
                if (event.stopPropagation) break;
            } catch (error) {
                this.errorHandler(error, event);
                return false;
            }
        }
        
        const listeners = this.events.get(eventName);
        if (!listeners || listeners.length === 0) return false;
        
        const promises = [];
        
        for (const listener of [...listeners]) {
            try {
                if (this.asyncListeners.has(listener)) {
                    promises.push(listener.call(this, ...event.data));
                } else {
                    listener.call(this, ...event.data);
                }
                
                // Remove once listeners
                if (this.onceEvents.has(listener)) {
                    this.off(eventName, listener);
                    this.onceEvents.delete(listener);
                }
            } catch (error) {
                this.errorHandler(error, { eventName, listener, args: event.data });
            }
        }
        
        // Wait for async listeners
        if (promises.length > 0) {
            try {
                await Promise.allSettled(promises);
            } catch (error) {
                this.errorHandler(error, { eventName, args: event.data });
            }
        }
        
        return true;
    }
    
    on(eventName, listener, options = {}) {
        this.validateListener(listener);
        
        if (!this.events.has(eventName)) {
            this.events.set(eventName, []);
        }
        
        const listeners = this.events.get(eventName);
        
        if (listeners.length >= this.maxListeners) {
            console.warn(`MaxListenersExceeded: ${eventName} has ${listeners.length} listeners`);
        }
        
        // Handle priority insertion
        if (options.priority !== undefined) {
            const insertIndex = listeners.findIndex(l => 
                (l.priority || 0) < options.priority
            );
            listener.priority = options.priority;
            listeners.splice(insertIndex === -1 ? listeners.length : insertIndex, 0, listener);
        } else {
            listeners.push(listener);
        }
        
        if (options.async) {
            this.asyncListeners.add(listener);
        }
        
        return this;
    }
    
    once(eventName, listener, options = {}) {
        this.on(eventName, listener, options);
        this.onceEvents.add(listener);
        return this;
    }
    
    off(eventName, listener) {
        const listeners = this.events.get(eventName);
        if (!listeners) return this;
        
        const index = listeners.indexOf(listener);
        if (index !== -1) {
            listeners.splice(index, 1);
            this.asyncListeners.delete(listener);
            this.onceEvents.delete(listener);
        }
        
        if (listeners.length === 0) {
            this.events.delete(eventName);
        }
        
        return this;
    }
    
    validateListener(listener) {
        if (typeof listener !== 'function') {
            throw new TypeError('Listener must be a function');
        }
    }
    
    defaultErrorHandler(error, context) {
        console.error('EventEmitter Error:', error, context);
    }
    
    // Namespace support
    namespace(ns) {
        return new Proxy(this, {
            get(target, prop) {
                if (prop === 'emit' || prop === 'on' || prop === 'once' || prop === 'off') {
                    return function(eventName, ...args) {
                        return target[prop](`${ns}:${eventName}`, ...args);
                    };
                }
                return target[prop];
            }
        });
    }
}

// Command Pattern with Undo/Redo
class Command {
    constructor(execute, undo, context = null) {
        this.execute = execute;
        this.undo = undo;
        this.context = context;
        this.timestamp = Date.now();
        this.id = this.generateId();
    }
    
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
}

class CommandManager {
    constructor(options = {}) {
        this.history = [];
        this.currentIndex = -1;
        this.maxHistorySize = options.maxHistorySize || 100;
        this.eventEmitter = new AdvancedEventEmitter();
        this.macros = new Map();
        this.transactionStack = [];
    }
    
    execute(command) {
        if (!(command instanceof Command)) {
            throw new TypeError('Expected Command instance');
        }
        
        try {
            const result = command.execute.call(command.context);
            
            // Remove any commands after current index (branch pruning)
            this.history = this.history.slice(0, this.currentIndex + 1);
            
            // Add new command
            this.history.push(command);
            this.currentIndex++;
            
            // Limit history size
            if (this.history.length > this.maxHistorySize) {
                this.history.shift();
                this.currentIndex--;
            }
            
            this.eventEmitter.emit('commandExecuted', command, result);
            return result;
        } catch (error) {
            this.eventEmitter.emit('commandError', command, error);
            throw error;
        }
    }
    
    undo() {
        if (this.currentIndex < 0) return false;
        
        const command = this.history[this.currentIndex];
        
        try {
            const result = command.undo.call(command.context);
            this.currentIndex--;
            this.eventEmitter.emit('commandUndone', command, result);
            return result;
        } catch (error) {
            this.eventEmitter.emit('undoError', command, error);
            throw error;
        }
    }
    
    redo() {
        if (this.currentIndex >= this.history.length - 1) return false;
        
        this.currentIndex++;
        const command = this.history[this.currentIndex];
        
        try {
            const result = command.execute.call(command.context);
            this.eventEmitter.emit('commandRedone', command, result);
            return result;
        } catch (error) {
            this.currentIndex--;
            this.eventEmitter.emit('redoError', command, error);
            throw error;
        }
    }
    
    // Macro recording and playback
    startMacro(name) {
        this.macros.set(name, []);
        this.currentMacro = name;
    }
    
    stopMacro() {
        this.currentMacro = null;
    }
    
    playMacro(name) {
        const macro = this.macros.get(name);
        if (!macro) throw new Error(`Macro '${name}' not found`);
        
        for (const command of macro) {
            this.execute(command);
        }
    }
    
    // Transaction support
    beginTransaction() {
        this.transactionStack.push([]);
    }
    
    commitTransaction() {
        const commands = this.transactionStack.pop();
        if (!commands) return;
        
        // Create a composite command
        const compositeCommand = new Command(
            () => commands.forEach(cmd => cmd.execute.call(cmd.context)),
            () => commands.reverse().forEach(cmd => cmd.undo.call(cmd.context))
        );
        
        this.execute(compositeCommand);
    }
    
    rollbackTransaction() {
        const commands = this.transactionStack.pop();
        if (!commands) return;
        
        commands.reverse().forEach(cmd => {
            try {
                cmd.undo.call(cmd.context);
            } catch (error) {
                console.error('Rollback error:', error);
            }
        });
    }
}

// Strategy Pattern with Dynamic Loading
class StrategyManager {
    constructor() {
        this.strategies = new Map();
        this.middleware = [];
        this.cache = new Map();
        this.loadingPromises = new Map();
    }
    
    register(name, strategy) {
        if (typeof strategy !== 'function' && typeof strategy !== 'object') {
            throw new TypeError('Strategy must be a function or object');
        }
        
        this.strategies.set(name, strategy);
        this.cache.delete(name); // Clear cache when strategy is updated
        return this;
    }
    
    async registerAsync(name, importFunction) {
        if (this.loadingPromises.has(name)) {
            return this.loadingPromises.get(name);
        }
        
        const loadPromise = (async () => {
            try {
                const module = await importFunction();
                const strategy = module.default || module;
                this.register(name, strategy);
                return strategy;
            } catch (error) {
                this.loadingPromises.delete(name);
                throw error;
            }
        })();
        
        this.loadingPromises.set(name, loadPromise);
        return loadPromise;
    }
    
    async execute(strategyName, context, ...args) {
        let strategy = this.strategies.get(strategyName);
        
        // Try to load strategy if not found
        if (!strategy && this.loadingPromises.has(strategyName)) {
            strategy = await this.loadingPromises.get(strategyName);
        }
        
        if (!strategy) {
            throw new Error(`Strategy '${strategyName}' not found`);
        }
        
        // Apply middleware
        for (const middleware of this.middleware) {
            context = await middleware(context, strategyName);
        }
        
        // Execute strategy
        const result = typeof strategy === 'function' 
            ? await strategy(context, ...args)
            : await strategy.execute(context, ...args);
        
        return result;
    }
    
    use(middleware) {
        this.middleware.push(middleware);
        return this;
    }
    
    has(name) {
        return this.strategies.has(name) || this.loadingPromises.has(name);
    }
    
    remove(name) {
        this.strategies.delete(name);
        this.cache.delete(name);
        this.loadingPromises.delete(name);
        return this;
    }
}

// Facade Pattern for Complex Subsystem
class ApplicationFacade {
    constructor() {
        this.eventEmitter = new AdvancedEventEmitter();
        this.commandManager = new CommandManager();
        this.strategyManager = new StrategyManager();
        this.services = new Map();
        this.modules = new Map();
        this.config = new Map();
        
        this.setupDefaultStrategies();
        this.setupEventHandlers();
    }
    
    setupDefaultStrategies() {
        // Default validation strategies
        this.strategyManager.register('email', (value) => {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(value);
        });
        
        this.strategyManager.register('required', (value) => {
            return value != null && value !== '';
        });
        
        // Default data processing strategies
        this.strategyManager.register('uppercase', (value) => {
            return typeof value === 'string' ? value.toUpperCase() : value;
        });
        
        this.strategyManager.register('trim', (value) => {
            return typeof value === 'string' ? value.trim() : value;
        });
    }
    
    setupEventHandlers() {
        this.eventEmitter.on('service:registered', (serviceName) => {
            console.log(`Service '${serviceName}' registered`);
        });
        
        this.eventEmitter.on('module:loaded', (moduleName) => {
            console.log(`Module '${moduleName}' loaded`);
        });
    }
    
    // Service management
    registerService(name, service) {
        if (this.services.has(name)) {
            throw new Error(`Service '${name}' already registered`);
        }
        
        this.services.set(name, service);
        this.eventEmitter.emit('service:registered', name);
        return this;
    }
    
    getService(name) {
        const service = this.services.get(name);
        if (!service) {
            throw new Error(`Service '${name}' not found`);
        }
        return service;
    }
    
    // Module management with lazy loading
    async loadModule(name, importFunction) {
        if (this.modules.has(name)) {
            return this.modules.get(name);
        }
        
        try {
            const module = await importFunction();
            this.modules.set(name, module);
            
            // Auto-register services if module exports them
            if (module.services) {
                Object.entries(module.services).forEach(([serviceName, service]) => {
                    this.registerService(`${name}.${serviceName}`, service);
                });
            }
            
            // Auto-register strategies if module exports them
            if (module.strategies) {
                Object.entries(module.strategies).forEach(([strategyName, strategy]) => {
                    this.strategyManager.register(`${name}.${strategyName}`, strategy);
                });
            }
            
            this.eventEmitter.emit('module:loaded', name);
            return module;
        } catch (error) {
            this.eventEmitter.emit('module:loadError', name, error);
            throw error;
        }
    }
    
    // Configuration management
    setConfig(key, value) {
        this.config.set(key, value);
        this.eventEmitter.emit('config:changed', key, value);
        return this;
    }
    
    getConfig(key, defaultValue = null) {
        return this.config.get(key) ?? defaultValue;
    }
    
    // High-level operations
    async processData(data, processors = []) {
        let result = data;
        
        for (const processor of processors) {
            if (typeof processor === 'string') {
                result = await this.strategyManager.execute(processor, result);
            } else if (typeof processor === 'object') {
                result = await this.strategyManager.execute(
                    processor.strategy, 
                    result, 
                    ...processor.args || []
                );
            }
        }
        
        return result;
    }
    
    async validate(data, rules) {
        const errors = [];
        
        for (const [field, fieldRules] of Object.entries(rules)) {
            const value = data[field];
            
            for (const rule of fieldRules) {
                const isValid = await this.strategyManager.execute(rule, value);
                if (!isValid) {
                    errors.push({ field, rule, value });
                }
            }
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }
    
    // Command execution with history
    executeCommand(command) {
        return this.commandManager.execute(command);
    }
    
    undo() {
        return this.commandManager.undo();
    }
    
    redo() {
        return this.commandManager.redo();
    }
    
    // Event handling
    on(event, listener, options) {
        return this.eventEmitter.on(event, listener, options);
    }
    
    emit(event, ...args) {
        return this.eventEmitter.emit(event, ...args);
    }
    
    // Cleanup
    destroy() {
        this.services.clear();
        this.modules.clear();
        this.config.clear();
        this.eventEmitter.events.clear();
        this.commandManager.history = [];
        this.strategyManager.strategies.clear();
    }
}

// Usage Examples
async function demonstrateAdvancedPatterns() {
    const app = new ApplicationFacade();
    
    // Register custom strategies
    await app.strategyManager.registerAsync('hash', () => 
        import('crypto').then(crypto => (value) => 
            crypto.createHash('sha256').update(value).digest('hex')
        )
    );
    
    // Register a service
    app.registerService('logger', {
        log: (message, level = 'info') => {
            console.log(`[${level.toUpperCase()}] ${new Date().toISOString()}: ${message}`);
        }
    });
    
    // Process data with multiple strategies
    const processedData = await app.processData('  Hello World  ', [
        'trim',
        'uppercase',
        { strategy: 'hash', args: [] }
    ]);
    
    console.log('Processed data:', processedData);
    
    // Validate data
    const validationResult = await app.validate(
        { email: 'user@example.com', name: 'John' },
        {
            email: ['required', 'email'],
            name: ['required']
        }
    );
    
    console.log('Validation result:', validationResult);
    
    // Command pattern example
    let counter = 0;
    
    const incrementCommand = new Command(
        () => ++counter,
        () => --counter
    );
    
    app.executeCommand(incrementCommand);
    console.log('Counter after increment:', counter);
    
    app.undo();
    console.log('Counter after undo:', counter);
    
    app.redo();
    console.log('Counter after redo:', counter);
    
    return app;
}

// Export for module usage
export {
    AdvancedEventEmitter,
    Command,
    CommandManager,
    StrategyManager,
    ApplicationFacade,
    demonstrateAdvancedPatterns
};
```

## Dataset 2: Performance Optimization and Memory Management
```javascript
// Advanced Memory Management and Performance Monitoring
class MemoryManager {
    constructor(options = {}) {
        this.pools = new Map();
        this.metrics = {
            allocations: 0,
            deallocations: 0,
            poolHits: 0,
            poolMisses: 0,
            memoryUsage: []
        };
        this.monitoringInterval = options.monitoringInterval || 5000;
        this.maxPoolSize = options.maxPoolSize || 100;
        this.garbageCollectionThreshold = options.gcThreshold || 1000;
        
        this.startMonitoring();
    }
    
    // Object pooling for frequently created objects
    createPool(type, factory, resetFunction) {
        this.pools.set(type, {
            objects: [],
            factory,
            reset: resetFunction || (() => {}),
            created: 0,
            reused: 0
        });
        return this;
    }
    
    acquire(type, ...args) {
        const pool = this.pools.get(type);
        if (!pool) {
            throw new Error(`Pool for type '${type}' not found`);
        }
        
        let object;
        
        if (pool.objects.length > 0) {
            object = pool.objects.pop();
            pool.reset(object, ...args);
            pool.reused++;
            this.metrics.poolHits++;
        } else {
            object = pool.factory(...args);
            pool.created++;
            this.metrics.poolMisses++;
        }
        
        this.metrics.allocations++;
        return object;
    }
    
    release(type, object) {
        const pool = this.pools.get(type);
        if (!pool) return false;
        
        if (pool.objects.length < this.maxPoolSize) {
            pool.objects.push(object);
            this.metrics.deallocations++;
            return true;
        }
        
        return false;
    }
    
    // Memory monitoring
    startMonitoring() {
        if (typeof window !== 'undefined' && window.performance && window.performance.memory) {
            this.monitoringTimer = setInterval(() => {
                const memory = window.performance.memory;
                this.metrics.memoryUsage.push({
                    timestamp: Date.now(),
                    used: memory.usedJSHeapSize,
                    total: memory.totalJSHeapSize,
                    limit: memory.jsHeapSizeLimit
                });
                
                // Keep only last 100 measurements
                if (this.metrics.memoryUsage.length > 100) {
                    this.metrics.memoryUsage = this.metrics.memoryUsage.slice(-100);
                }
            }, this.monitoringInterval);
        }
    }
    
    stopMonitoring() {
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
            this.monitoringTimer = null;
        }
    }
    
    getMemoryReport() {
        const poolStats = Array.from(this.pools.entries()).map(([type, pool]) => ({
            type,
            available: pool.objects.length,
            created: pool.created,
            reused: pool.reused,
            reuseRatio: pool.reused / (pool.created + pool.reused) || 0
        }));
        
        return {
            pools: poolStats,
            metrics: this.metrics,
            efficiency: {
                poolHitRatio: this.metrics.poolHits / (this.metrics.poolHits + this.metrics.poolMisses) || 0,
                allocationEfficiency: this.metrics.deallocations / this.metrics.allocations || 0
            }
        };
    }
    
    forceGarbageCollection() {
        // Clear pools if they're too large
        this.pools.forEach(pool => {
            if (pool.objects.length > this.garbageCollectionThreshold) {
                pool.objects = pool.objects.slice(0, Math.floor(this.maxPoolSize / 2));
            }
        });
        
        // Force garbage collection if available (Chrome DevTools)
        if (window.gc && typeof window.gc === 'function') {
            window.gc();
        }
    }
}

// Performance monitoring and profiling
class PerformanceProfiler {
    constructor() {
        this.profiles = new Map();
        this.activeTimers = new Map();
        this.marks = new Map();
        this.measurements = [];
        this.hooks = new Map();
    }
    
    // Method profiling
    profile(target, methodName, options = {}) {
        const originalMethod = target[methodName];
        if (typeof originalMethod !== 'function') {
            throw new Error(`Method '${methodName}' is not a function`);
        }
        
        const profileKey = `${target.constructor.name}.${methodName}`;
        this.profiles.set(profileKey, {
            calls: 0,
            totalTime: 0,
            minTime: Infinity,
            maxTime: 0,
            errors: 0,
            avgTime: 0
        });
        
        target[methodName] = (...args) => {
            const start = performance.now();
            const profile = this.profiles.get(profileKey);
            
            try {
                const result = originalMethod.apply(target, args);
                
                // Handle promises
                if (result && typeof result.then === 'function') {
                    return result
                        .then(value => {
                            this.recordMethodCall(profileKey, start);
                            return value;
                        })
                        .catch(error => {
                            this.recordMethodCall(profileKey, start, error);
                            throw error;
                        });
                }
                
                this.recordMethodCall(profileKey, start);
                return result;
            } catch (error) {
                this.recordMethodCall(profileKey, start, error);
                throw error;
            }
        };
        
        // Preserve original method properties
        Object.defineProperty(target[methodName], 'originalMethod', {
            value: originalMethod,
            writable: false
        });
        
        return this;
    }
    
    recordMethodCall(profileKey, startTime, error = null) {
        const endTime = performance.now();
        const duration = endTime - startTime;
        const profile = this.profiles.get(profileKey);
        
        profile.calls++;
        profile.totalTime += duration;
        profile.minTime = Math.min(profile.minTime, duration);
        profile.maxTime = Math.max(profile.maxTime, duration);
        profile.avgTime = profile.totalTime / profile.calls;
        
        if (error) {
            profile.errors++;
        }
    }
    
    // Function timing
    time(label) {
        this.activeTimers.set(label, performance.now());
        return this;
    }
    
    timeEnd(label) {
        const startTime = this.activeTimers.get(label);
        if (!startTime) {
            console.warn(`Timer '${label}' not found`);
            return null;
        }
        
        const duration = performance.now() - startTime;
        this.activeTimers.delete(label);
        
        console.log(`${label}: ${duration.toFixed(3)}ms`);
        return duration;
    }
    
    // Performance marks and measurements
    mark(name) {
        performance.mark(name);
        this.marks.set(name, performance.now());
        return this;
    }
    
    measure(name, startMark, endMark) {
        performance.measure(name, startMark, endMark);
        
        const startTime = this.marks.get(startMark);
        const endTime = this.marks.get(endMark);
        
        if (startTime && endTime) {
            const measurement = {
                name,
                duration: endTime - startTime,
                timestamp: Date.now()
            };
            
            this.measurements.push(measurement);
            return measurement;
        }
        
        return null;
    }
    
    // Performance observer for detailed metrics
    observePerformance(callback) {
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                callback(entries);
            });
            
            observer.observe({ entryTypes: ['measure', 'navigation', 'resource', 'paint'] });
            return observer;
        }
        
        return null;
    }
    
    // Hook into function calls for monitoring
    hook(target, methodName, beforeHook, afterHook) {
        const originalMethod = target[methodName];
        const hookKey = `${target.constructor.name}.${methodName}`;
        
        target[methodName] = function(...args) {
            if (beforeHook) {
                const beforeResult = beforeHook.call(this, methodName, args);
                if (beforeResult === false) return; // Cancel execution
            }
            
            const result = originalMethod.apply(this, args);
            
            if (afterHook) {
                afterHook.call(this, methodName, args, result);
            }
            
            return result;
        };
        
        this.hooks.set(hookKey, { before: beforeHook, after: afterHook });
        return this;
    }
    
    // Generate performance report
    getReport() {
        const profileReport = Array.from(this.profiles.entries()).map(([key, profile]) => ({
            method: key,
            ...profile,
            efficiency: profile.errors / profile.calls || 0
        }));
        
        const measurementReport = this.measurements.slice(-20); // Last 20 measurements
        
        return {
            profiles: profileReport,
            measurements: measurementReport,
            activeTimers: Array.from(this.activeTimers.keys()),
            totalMethods: this.profiles.size,
            totalMeasurements: this.measurements.length
        };
    }
    
    // Clear all profiling data
    clear() {
        this.profiles.clear();
        this.activeTimers.clear();
        this.marks.clear();
        this.measurements = [];
        this.hooks.clear();
        
        // Clear performance marks
        performance.clearMarks();
        performance.clearMeasures();
    }
}

// Advanced caching with TTL and size limits
class AdvancedCache {
    constructor(options = {}) {
        this.cache = new Map();
        this.timers = new Map();
        this.accessTimes = new Map();
        this.maxSize = options.maxSize || 1000;
        this.defaultTTL = options.defaultTTL || 300000; // 5 minutes
        this.cleanupInterval = options.cleanupInterval || 60000; // 1 minute
        this.strategy = options.strategy || 'lru'; // lru, lfu, fifo
        this.metrics = {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            evictions: 0
        };
        
        this.startCleanupTimer();
    }
    
    set(key, value, ttl = this.defaultTTL) {
        // Remove existing entry
        if (this.cache.has(key)) {
            this.clearTimer(key);
        } else if (this.cache.size >= this.maxSize) {
            this.evict();
        }
        
        const entry = {
            value,
            created: Date.now(),
            accessed: Date.now(),
            accessCount: 0,
            ttl
        };
        
        this.cache.set(key, entry);
        this.accessTimes.set(key, Date.now());
        this.metrics.sets++;
        
        // Set expiration timer
        if (ttl > 0) {
            const timer = setTimeout(() => {
                this.delete(key);
            }, ttl);
            
            this.timers.set(key, timer);
        }
        
        return this;
    }
    
    get(key) {
        const entry = this.cache.get(key);
        
        if (!entry) {
            this.metrics.misses++;
            return undefined;
        }
        
        // Check if expired
        if (this.isExpired(entry)) {
            this.delete(key);
            this.metrics.misses++;
            return undefined;
        }
        
        // Update access information
        entry.accessed = Date.now();
        entry.accessCount++;
        this.accessTimes.set(key, Date.now());
        this.metrics.hits++;
        
        return entry.value;
    }
    
    has(key) {
        const entry = this.cache.get(key);
        return entry && !this.isExpired(entry);
    }
    
    delete(key) {
        const deleted = this.cache.delete(key);
        this.accessTimes.delete(key);
        this.clearTimer(key);
        
        if (deleted) {
            this.metrics.deletes++;
        }
        
        return deleted;
    }
    
    clear() {
        this.cache.clear();
        this.accessTimes.clear();
        this.timers.forEach(timer => clearTimeout(timer));
        this.timers.clear();
    }
    
    // Eviction strategies
    evict() {
        const keyToEvict = this.selectKeyForEviction();
        if (keyToEvict) {
            this.delete(keyToEvict);
            this.metrics.evictions++;
        }
    }
    
    selectKeyForEviction() {
        if (this.cache.size === 0) return null;
        
        switch (this.strategy) {
            case 'lru': // Least Recently Used
                return this.findLRUKey();
            case 'lfu': // Least Frequently Used
                return this.findLFUKey();
            case 'fifo': // First In, First Out
                return this.findFIFOKey();
            default:
                return this.cache.keys().next().value;
        }
    }
    
    findLRUKey() {
        let oldestKey = null;
        let oldestTime = Infinity;
        
        for (const [key, time] of this.accessTimes) {
            if (time < oldestTime) {
                oldestTime = time;
                oldestKey = key;
            }
        }
        
        return oldestKey;
    }
    
    findLFUKey() {
        let leastUsedKey = null;
        let leastUsedCount = Infinity;
        
        for (const [key, entry] of this.cache) {
            if (entry.accessCount < leastUsedCount) {
                leastUsedCount = entry.accessCount;
                leastUsedKey = key;
            }
        }
        
        return leastUsedKey;
    }
    
    findFIFOKey() {
        let oldestKey = null;
        let oldestCreated = Infinity;
        
        for (const [key, entry] of this.cache) {
            if (entry.created < oldestCreated) {
                oldestCreated = entry.created;
                oldestKey = key;
            }
        }
        
        return oldestKey;
    }
    
    isExpired(entry) {
        return entry.ttl > 0 && (Date.now() - entry.created) > entry.ttl;
    }
    
    clearTimer(key) {
        const timer = this.timers.get(key);
        if (timer) {
            clearTimeout(timer);
            this.timers.delete(key);
        }
    }
    
    startCleanupTimer() {
        this.cleanupTimer = setInterval(() => {
            this.cleanup();
        }, this.cleanupInterval);
    }
    
    cleanup() {
        const now = Date.now();
        const keysToDelete = [];
        
        for (const [key, entry] of this.cache) {
            if (this.isExpired(entry)) {
                keysToDelete.push(key);
            }
        }
        
        keysToDelete.forEach(key => this.delete(key));
    }
    
    getStats() {
        const hitRate = this.metrics.hits / (this.metrics.hits + this.metrics.misses) || 0;
        
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            hitRate: hitRate.toFixed(3),
            metrics: this.metrics,
            averageAge: this.getAverageAge(),
            strategy: this.strategy
        };
    }
    
    getAverageAge() {
        if (this.cache.size === 0) return 0;
        
        const now = Date.now();
        let totalAge = 0;
        
        for (const entry of this.cache.values()) {
            totalAge += now - entry.created;
        }
        
        return totalAge / this.cache.size;
    }
    
    destroy() {
        this.clear();
        if (this.cleanupTimer) {
            clearInterval(this.cleanupTimer);
        }
    }
}

// Example usage and benchmarking
class PerformanceBenchmark {
    constructor() {
        this.memoryManager = new MemoryManager();
        this.profiler = new PerformanceProfiler();
        this.cache = new AdvancedCache({ maxSize: 1000, defaultTTL: 60000 });
        
        this.setupObjectPools();
    }
    
    setupObjectPools() {
        // Point object pool
        this.memoryManager.createPool(
            'point',
            (x = 0, y = 0) => ({ x, y }),
            (point, x = 0, y = 0) => {
                point.x = x;
                point.y = y;
            }
        );
        
        // Array buffer pool
        this.memoryManager.createPool(
            'buffer',
            (size = 1024) => new ArrayBuffer(size),
            (buffer) => {
                // Reset buffer if needed
                new Uint8Array(buffer).fill(0);
            }
        );
    }
    
    // Benchmark object creation vs pooling
    benchmarkObjectPooling(iterations = 100000) {
        console.log('Benchmarking object pooling...');
        
        // Without pooling
        this.profiler.time('withoutPooling');
        for (let i = 0; i < iterations; i++) {
            const point = { x: i, y: i };
            // Simulate some work
            point.distance = Math.sqrt(point.x * point.x + point.y * point.y);
        }
        this.profiler.timeEnd('withoutPooling');
        
        // With pooling
        this.profiler.time('withPooling');
        for (let i = 0; i < iterations; i++) {
            const point = this.memoryManager.acquire('point', i, i);
            // Simulate some work
            point.distance = Math.sqrt(point.x * point.x + point.y * point.y);
            this.memoryManager.release('point', point);
        }
        this.profiler.timeEnd('withPooling');
        
        console.log('Memory manager report:', this.memoryManager.getMemoryReport());
    }
    
    // Benchmark cache performance
    benchmarkCache(operations = 50000) {
        console.log('Benchmarking cache performance...');
        
        const keys = Array.from({ length: 1000 }, (_, i) => `key_${i}`);
        const values = Array.from({ length: 1000 }, (_, i) => ({ data: `value_${i}`, timestamp: Date.now() }));
        
        this.profiler.time('cacheOperations');
        
        // Fill cache
        for (let i = 0; i < keys.length; i++) {
            this.cache.set(keys[i], values[i]);
        }
        
        // Random access
        for (let i = 0; i < operations; i++) {
            const randomKey = keys[Math.floor(Math.random() * keys.length)];
            this.cache.get(randomKey);
        }
        
        this.profiler.timeEnd('cacheOperations');
        
        console.log('Cache stats:', this.cache.getStats());
    }
    
    // Benchmark function profiling
    benchmarkMethodProfiling() {
        console.log('Benchmarking method profiling...');
        
        class TestClass {
            slowMethod(n) {
                let result = 0;
                for (let i = 0; i < n; i++) {
                    result += Math.random();
                }
                return result;
            }
            
            async asyncMethod(delay) {
                return new Promise(resolve => {
                    setTimeout(() => resolve('completed'), delay);
                });
            }
        }
        
        const testInstance = new TestClass();
        
        // Profile methods
        this.profiler.profile(testInstance, 'slowMethod');
        this.profiler.profile(testInstance, 'asyncMethod');
        
        // Run tests
        for (let i = 0; i < 100; i++) {
            testInstance.slowMethod(1000);
        }
        
        // Run async tests
        Promise.all([
            testInstance.asyncMethod(10),
            testInstance.asyncMethod(20),
            testInstance.asyncMethod(30)
        ]).then(() => {
            console.log('Profiler report:', this.profiler.getReport());
        });
    }
    
    runAllBenchmarks() {
        this.benchmarkObjectPooling();
        this.benchmarkCache();
        this.benchmarkMethodProfiling();
    }
}

// Usage example
export function demonstratePerformanceOptimization() {
    const benchmark = new PerformanceBenchmark();
    benchmark.runAllBenchmarks();
    
    return {
        memoryManager: benchmark.memoryManager,
        profiler: benchmark.profiler,
        cache: benchmark.cache
    };
}

export {
    MemoryManager,
    PerformanceProfiler,
    AdvancedCache,
    PerformanceBenchmark
};
```