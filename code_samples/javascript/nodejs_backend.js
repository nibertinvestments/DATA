// Node.js backend development patterns and practices

const http = require('http');
const url = require('url');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { EventEmitter } = require('events');

console.log("=== Node.js Backend Development ===");

// HTTP Server with routing
class SimpleRouter {
    constructor() {
        this.routes = new Map();
        this.middleware = [];
    }
    
    use(middleware) {
        this.middleware.push(middleware);
        return this;
    }
    
    get(path, handler) {
        this.addRoute('GET', path, handler);
        return this;
    }
    
    post(path, handler) {
        this.addRoute('POST', path, handler);
        return this;
    }
    
    put(path, handler) {
        this.addRoute('PUT', path, handler);
        return this;
    }
    
    delete(path, handler) {
        this.addRoute('DELETE', path, handler);
        return this;
    }
    
    addRoute(method, path, handler) {
        const key = `${method}:${path}`;
        this.routes.set(key, handler);
    }
    
    async handle(req, res) {
        // Parse URL and set up request object
        const parsedUrl = url.parse(req.url, true);
        req.path = parsedUrl.pathname;
        req.query = parsedUrl.query;
        req.params = {};
        
        // Add response helpers
        res.json = (data) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(data));
        };
        
        res.status = (code) => {
            res.statusCode = code;
            return res;
        };
        
        res.send = (data) => {
            res.writeHead(res.statusCode || 200, { 'Content-Type': 'text/plain' });
            res.end(data.toString());
        };
        
        try {
            // Execute middleware
            for (const middleware of this.middleware) {
                await new Promise((resolve, reject) => {
                    middleware(req, res, (error) => {
                        if (error) reject(error);
                        else resolve();
                    });
                });
            }
            
            // Find and execute route handler
            const routeKey = `${req.method}:${req.path}`;
            const handler = this.routes.get(routeKey);
            
            if (handler) {
                await handler(req, res);
            } else {
                res.status(404).send('Not Found');
            }
        } catch (error) {
            console.error('Request error:', error);
            res.status(500).send('Internal Server Error');
        }
    }
}

// Middleware examples
const loggingMiddleware = (req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`${timestamp} ${req.method} ${req.url}`);
    next();
};

const authMiddleware = (req, res, next) => {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        res.status(401).json({ error: 'Authentication required' });
        return;
    }
    
    const token = authHeader.substring(7);
    
    // Simple token validation (in real app, verify JWT or session)
    if (token === 'valid-token') {
        req.user = { id: 1, username: 'testuser' };
        next();
    } else {
        res.status(401).json({ error: 'Invalid token' });
    }
};

const corsMiddleware = (req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }
    
    next();
};

// Database simulation with in-memory storage
class InMemoryDatabase extends EventEmitter {
    constructor() {
        super();
        this.collections = new Map();
        this.indexes = new Map();
    }
    
    collection(name) {
        if (!this.collections.has(name)) {
            this.collections.set(name, new Map());
            this.indexes.set(name, new Map());
        }
        return new Collection(this, name);
    }
    
    getCollection(name) {
        return this.collections.get(name) || new Map();
    }
    
    getIndexes(name) {
        return this.indexes.get(name) || new Map();
    }
}

class Collection {
    constructor(db, name) {
        this.db = db;
        this.name = name;
        this.nextId = 1;
    }
    
    insert(document) {
        const collection = this.db.getCollection(this.name);
        const id = document.id || this.nextId++;
        const doc = { ...document, id, createdAt: new Date(), updatedAt: new Date() };
        
        collection.set(id, doc);
        this.updateIndexes(doc);
        this.db.emit('insert', { collection: this.name, document: doc });
        
        return doc;
    }
    
    findById(id) {
        const collection = this.db.getCollection(this.name);
        return collection.get(id);
    }
    
    find(query = {}) {
        const collection = this.db.getCollection(this.name);
        const results = [];
        
        for (const doc of collection.values()) {
            if (this.matchesQuery(doc, query)) {
                results.push(doc);
            }
        }
        
        return results;
    }
    
    findOne(query = {}) {
        const results = this.find(query);
        return results[0] || null;
    }
    
    update(id, updates) {
        const collection = this.db.getCollection(this.name);
        const doc = collection.get(id);
        
        if (!doc) {
            throw new Error('Document not found');
        }
        
        const updatedDoc = { ...doc, ...updates, updatedAt: new Date() };
        collection.set(id, updatedDoc);
        this.updateIndexes(updatedDoc);
        this.db.emit('update', { collection: this.name, document: updatedDoc });
        
        return updatedDoc;
    }
    
    delete(id) {
        const collection = this.db.getCollection(this.name);
        const doc = collection.get(id);
        
        if (doc) {
            collection.delete(id);
            this.removeFromIndexes(doc);
            this.db.emit('delete', { collection: this.name, document: doc });
            return true;
        }
        
        return false;
    }
    
    matchesQuery(doc, query) {
        for (const [key, value] of Object.entries(query)) {
            if (doc[key] !== value) {
                return false;
            }
        }
        return true;
    }
    
    updateIndexes(doc) {
        // Simple indexing by field values
        const indexes = this.db.getIndexes(this.name);
        
        for (const [key, value] of Object.entries(doc)) {
            if (!indexes.has(key)) {
                indexes.set(key, new Map());
            }
            
            const fieldIndex = indexes.get(key);
            if (!fieldIndex.has(value)) {
                fieldIndex.set(value, new Set());
            }
            
            fieldIndex.get(value).add(doc.id);
        }
    }
    
    removeFromIndexes(doc) {
        const indexes = this.db.getIndexes(this.name);
        
        for (const [key, value] of Object.entries(doc)) {
            const fieldIndex = indexes.get(key);
            if (fieldIndex && fieldIndex.has(value)) {
                fieldIndex.get(value).delete(doc.id);
            }
        }
    }
}

// Service layer patterns
class UserService {
    constructor(database) {
        this.db = database;
        this.users = database.collection('users');
    }
    
    async createUser(userData) {
        // Validate input
        if (!userData.email || !userData.username) {
            throw new Error('Email and username are required');
        }
        
        // Check if user already exists
        const existing = this.users.findOne({ email: userData.email });
        if (existing) {
            throw new Error('User already exists');
        }
        
        // Hash password (simplified)
        const hashedPassword = crypto
            .createHash('sha256')
            .update(userData.password || 'defaultpassword')
            .digest('hex');
        
        const user = this.users.insert({
            ...userData,
            password: hashedPassword,
            status: 'active'
        });
        
        // Remove password from response
        const { password, ...userResponse } = user;
        return userResponse;
    }
    
    async getUserById(id) {
        const user = this.users.findById(parseInt(id));
        if (!user) {
            throw new Error('User not found');
        }
        
        const { password, ...userResponse } = user;
        return userResponse;
    }
    
    async getAllUsers() {
        const users = this.users.find({ status: 'active' });
        return users.map(({ password, ...user }) => user);
    }
    
    async updateUser(id, updates) {
        const existingUser = this.users.findById(parseInt(id));
        if (!existingUser) {
            throw new Error('User not found');
        }
        
        // Don't allow updating certain fields
        const { password, id: userId, createdAt, ...allowedUpdates } = updates;
        
        const updatedUser = this.users.update(parseInt(id), allowedUpdates);
        const { password: pwd, ...userResponse } = updatedUser;
        return userResponse;
    }
    
    async deleteUser(id) {
        const deleted = this.users.delete(parseInt(id));
        if (!deleted) {
            throw new Error('User not found');
        }
        return { message: 'User deleted successfully' };
    }
    
    async searchUsers(query) {
        const users = this.users.find();
        return users
            .filter(user => 
                user.username.toLowerCase().includes(query.toLowerCase()) ||
                user.email.toLowerCase().includes(query.toLowerCase())
            )
            .map(({ password, ...user }) => user);
    }
}

// File operations and utilities
class FileService {
    constructor(uploadDir = './uploads') {
        this.uploadDir = uploadDir;
        this.ensureDirectoryExists();
    }
    
    async ensureDirectoryExists() {
        try {
            await fs.access(this.uploadDir);
        } catch {
            await fs.mkdir(this.uploadDir, { recursive: true });
        }
    }
    
    async saveFile(filename, data) {
        const filePath = path.join(this.uploadDir, filename);
        await fs.writeFile(filePath, data);
        return {
            filename,
            path: filePath,
            size: data.length,
            uploadedAt: new Date()
        };
    }
    
    async getFile(filename) {
        const filePath = path.join(this.uploadDir, filename);
        try {
            const data = await fs.readFile(filePath);
            const stats = await fs.stat(filePath);
            return {
                filename,
                data,
                size: stats.size,
                modifiedAt: stats.mtime
            };
        } catch (error) {
            throw new Error('File not found');
        }
    }
    
    async deleteFile(filename) {
        const filePath = path.join(this.uploadDir, filename);
        try {
            await fs.unlink(filePath);
            return true;
        } catch (error) {
            throw new Error('File not found');
        }
    }
    
    async listFiles() {
        try {
            const files = await fs.readdir(this.uploadDir);
            const fileDetails = await Promise.all(
                files.map(async (filename) => {
                    const filePath = path.join(this.uploadDir, filename);
                    const stats = await fs.stat(filePath);
                    return {
                        filename,
                        size: stats.size,
                        modifiedAt: stats.mtime,
                        isDirectory: stats.isDirectory()
                    };
                })
            );
            return fileDetails.filter(file => !file.isDirectory);
        } catch (error) {
            return [];
        }
    }
}

// WebSocket-like real-time communication simulation
class RealTimeService extends EventEmitter {
    constructor() {
        super();
        this.connections = new Map();
        this.rooms = new Map();
    }
    
    addConnection(connectionId, metadata = {}) {
        this.connections.set(connectionId, {
            id: connectionId,
            connectedAt: new Date(),
            ...metadata
        });
        
        this.emit('connection', connectionId);
    }
    
    removeConnection(connectionId) {
        // Remove from all rooms
        for (const [roomName, room] of this.rooms) {
            room.delete(connectionId);
            if (room.size === 0) {
                this.rooms.delete(roomName);
            }
        }
        
        this.connections.delete(connectionId);
        this.emit('disconnection', connectionId);
    }
    
    joinRoom(connectionId, roomName) {
        if (!this.rooms.has(roomName)) {
            this.rooms.set(roomName, new Set());
        }
        
        this.rooms.get(roomName).add(connectionId);
        this.emit('joinRoom', { connectionId, roomName });
    }
    
    leaveRoom(connectionId, roomName) {
        const room = this.rooms.get(roomName);
        if (room) {
            room.delete(connectionId);
            if (room.size === 0) {
                this.rooms.delete(roomName);
            }
        }
        
        this.emit('leaveRoom', { connectionId, roomName });
    }
    
    broadcast(message, roomName = null) {
        if (roomName) {
            const room = this.rooms.get(roomName);
            if (room) {
                for (const connectionId of room) {
                    this.sendToConnection(connectionId, message);
                }
            }
        } else {
            for (const connectionId of this.connections.keys()) {
                this.sendToConnection(connectionId, message);
            }
        }
    }
    
    sendToConnection(connectionId, message) {
        if (this.connections.has(connectionId)) {
            console.log(`📨 Sending to ${connectionId}:`, message);
            this.emit('message', { connectionId, message });
        }
    }
    
    getConnectionCount(roomName = null) {
        if (roomName) {
            const room = this.rooms.get(roomName);
            return room ? room.size : 0;
        }
        return this.connections.size;
    }
    
    getRooms() {
        return Array.from(this.rooms.keys()).map(roomName => ({
            name: roomName,
            connectionCount: this.rooms.get(roomName).size
        }));
    }
}

// Cache implementation
class CacheService {
    constructor(defaultTTL = 300000) { // 5 minutes default
        this.cache = new Map();
        this.timers = new Map();
        this.defaultTTL = defaultTTL;
    }
    
    set(key, value, ttl = this.defaultTTL) {
        // Clear existing timer if present
        if (this.timers.has(key)) {
            clearTimeout(this.timers.get(key));
        }
        
        this.cache.set(key, {
            value,
            createdAt: Date.now(),
            ttl
        });
        
        // Set expiration timer
        const timer = setTimeout(() => {
            this.delete(key);
        }, ttl);
        
        this.timers.set(key, timer);
    }
    
    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;
        
        // Check if expired
        if (Date.now() - item.createdAt > item.ttl) {
            this.delete(key);
            return null;
        }
        
        return item.value;
    }
    
    delete(key) {
        if (this.timers.has(key)) {
            clearTimeout(this.timers.get(key));
            this.timers.delete(key);
        }
        
        return this.cache.delete(key);
    }
    
    clear() {
        for (const timer of this.timers.values()) {
            clearTimeout(timer);
        }
        
        this.cache.clear();
        this.timers.clear();
    }
    
    size() {
        return this.cache.size;
    }
    
    keys() {
        return Array.from(this.cache.keys());
    }
    
    has(key) {
        return this.cache.has(key) && this.get(key) !== null;
    }
}

// Rate limiting
class RateLimiter {
    constructor(windowMs = 60000, maxRequests = 100) {
        this.windowMs = windowMs;
        this.maxRequests = maxRequests;
        this.requests = new Map();
    }
    
    isAllowed(identifier) {
        const now = Date.now();
        const windowStart = now - this.windowMs;
        
        if (!this.requests.has(identifier)) {
            this.requests.set(identifier, []);
        }
        
        const userRequests = this.requests.get(identifier);
        
        // Remove old requests outside the window
        const validRequests = userRequests.filter(timestamp => timestamp > windowStart);
        this.requests.set(identifier, validRequests);
        
        // Check if limit is exceeded
        if (validRequests.length >= this.maxRequests) {
            return false;
        }
        
        // Add current request
        validRequests.push(now);
        return true;
    }
    
    getRemainingRequests(identifier) {
        const now = Date.now();
        const windowStart = now - this.windowMs;
        
        if (!this.requests.has(identifier)) {
            return this.maxRequests;
        }
        
        const userRequests = this.requests.get(identifier);
        const validRequests = userRequests.filter(timestamp => timestamp > windowStart);
        
        return Math.max(0, this.maxRequests - validRequests.length);
    }
    
    getResetTime(identifier) {
        if (!this.requests.has(identifier)) {
            return Date.now();
        }
        
        const userRequests = this.requests.get(identifier);
        if (userRequests.length === 0) {
            return Date.now();
        }
        
        const oldestRequest = Math.min(...userRequests);
        return oldestRequest + this.windowMs;
    }
}

// Application setup and demonstration
function createApp() {
    const app = new SimpleRouter();
    const database = new InMemoryDatabase();
    const userService = new UserService(database);
    const fileService = new FileService();
    const realtimeService = new RealTimeService();
    const cache = new CacheService();
    const rateLimiter = new RateLimiter();
    
    // Middleware
    app.use(corsMiddleware);
    app.use(loggingMiddleware);
    
    // Rate limiting middleware
    app.use((req, res, next) => {
        const clientId = req.connection.remoteAddress || 'unknown';
        
        if (!rateLimiter.isAllowed(clientId)) {
            const resetTime = rateLimiter.getResetTime(clientId);
            res.setHeader('X-RateLimit-Reset', resetTime);
            res.status(429).json({ 
                error: 'Too Many Requests',
                resetTime: new Date(resetTime).toISOString()
            });
            return;
        }
        
        const remaining = rateLimiter.getRemainingRequests(clientId);
        res.setHeader('X-RateLimit-Remaining', remaining);
        next();
    });
    
    // Routes
    app.get('/', (req, res) => {
        res.json({ message: 'Node.js Backend API', version: '1.0.0' });
    });
    
    // User routes
    app.get('/users', async (req, res) => {
        try {
            const cacheKey = 'all_users';
            let users = cache.get(cacheKey);
            
            if (!users) {
                users = await userService.getAllUsers();
                cache.set(cacheKey, users, 60000); // Cache for 1 minute
            }
            
            res.json(users);
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
    
    app.get('/users/:id', async (req, res) => {
        try {
            const user = await userService.getUserById(req.params.id);
            res.json(user);
        } catch (error) {
            res.status(404).json({ error: error.message });
        }
    });
    
    app.post('/users', async (req, res) => {
        try {
            // Parse JSON body (simplified)
            let body = '';
            req.on('data', chunk => body += chunk);
            req.on('end', async () => {
                try {
                    const userData = JSON.parse(body);
                    const user = await userService.createUser(userData);
                    
                    // Invalidate cache
                    cache.delete('all_users');
                    
                    res.status(201).json(user);
                } catch (parseError) {
                    res.status(400).json({ error: 'Invalid JSON' });
                }
            });
        } catch (error) {
            res.status(400).json({ error: error.message });
        }
    });
    
    // Protected route example
    app.get('/protected', authMiddleware, (req, res) => {
        res.json({ 
            message: 'Access granted',
            user: req.user 
        });
    });
    
    // File upload simulation
    app.post('/upload', async (req, res) => {
        try {
            // Simulate file upload (in real app, use multer or similar)
            const filename = `file_${Date.now()}.txt`;
            const fileData = `Sample file content uploaded at ${new Date().toISOString()}`;
            
            const result = await fileService.saveFile(filename, Buffer.from(fileData));
            res.status(201).json(result);
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
    
    // WebSocket simulation
    app.get('/realtime/status', (req, res) => {
        res.json({
            connections: realtimeService.getConnectionCount(),
            rooms: realtimeService.getRooms()
        });
    });
    
    // Health check
    app.get('/health', (req, res) => {
        res.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            cache: {
                size: cache.size(),
                keys: cache.keys()
            }
        });
    });
    
    return app;
}

// Start server and demonstrate functionality
async function startServer() {
    const app = createApp();
    
    const server = http.createServer((req, res) => {
        app.handle(req, res);
    });
    
    const PORT = process.env.PORT || 3000;
    
    server.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
        console.log(`📝 API Documentation:`);
        console.log(`   GET    /           - API info`);
        console.log(`   GET    /users      - List users`);
        console.log(`   GET    /users/:id  - Get user by ID`);
        console.log(`   POST   /users      - Create user`);
        console.log(`   POST   /upload     - Upload file`);
        console.log(`   GET    /health     - Health check`);
        console.log(`   GET    /protected  - Protected route (requires auth)`);
    });
    
    return server;
}

// Export for module usage
module.exports = {
    SimpleRouter,
    InMemoryDatabase,
    UserService,
    FileService,
    RealTimeService,
    CacheService,
    RateLimiter,
    createApp,
    startServer
};

// Run if this file is executed directly
if (require.main === module) {
    startServer().catch(console.error);
}