/**
 * JavaScript Async Programming Examples
 * 
 * This module demonstrates various approaches to asynchronous programming
 * in JavaScript, including callbacks, promises, and async/await patterns.
 */

// Utility function to simulate async operations
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * 1. Callback Pattern Examples
 */

// Basic callback function
function fetchUserCallback(userId, callback) {
    // Simulate API call with setTimeout
    setTimeout(() => {
        if (userId <= 0) {
            callback(new Error('Invalid user ID'), null);
            return;
        }
        
        const user = {
            id: userId,
            name: `User${userId}`,
            email: `user${userId}@example.com`
        };
        
        callback(null, user);
    }, 100);
}

// Callback with error handling
function handleUserData() {
    console.log('=== Callback Pattern ===');
    
    fetchUserCallback(1, (error, user) => {
        if (error) {
            console.error('Error fetching user:', error.message);
            return;
        }
        
        console.log('User fetched:', user);
        
        // Nested callback (callback hell example)
        fetchUserCallback(user.id + 1, (error, nextUser) => {
            if (error) {
                console.error('Error fetching next user:', error.message);
                return;
            }
            
            console.log('Next user fetched:', nextUser);
        });
    });
}

/**
 * 2. Promise Pattern Examples
 */

// Promise-based function
function fetchUserPromise(userId) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (userId <= 0) {
                reject(new Error('Invalid user ID'));
                return;
            }
            
            const user = {
                id: userId,
                name: `User${userId}`,
                email: `user${userId}@example.com`,
                createdAt: new Date().toISOString()
            };
            
            resolve(user);
        }, 100);
    });
}

// Promise chaining
function handleUserDataPromises() {
    console.log('\n=== Promise Pattern ===');
    
    fetchUserPromise(1)
        .then(user => {
            console.log('User fetched:', user);
            return fetchUserPromise(user.id + 1);
        })
        .then(nextUser => {
            console.log('Next user fetched:', nextUser);
            return fetchUserPromise(nextUser.id + 1);
        })
        .then(thirdUser => {
            console.log('Third user fetched:', thirdUser);
        })
        .catch(error => {
            console.error('Error in promise chain:', error.message);
        });
}

// Promise.all for concurrent operations
function handleMultipleUsers() {
    console.log('\n=== Promise.all Pattern ===');
    
    const userIds = [1, 2, 3, 4, 5];
    const userPromises = userIds.map(id => fetchUserPromise(id));
    
    Promise.all(userPromises)
        .then(users => {
            console.log('All users fetched:', users.length);
            users.forEach(user => console.log(`- ${user.name}: ${user.email}`));
        })
        .catch(error => {
            console.error('Error fetching users:', error.message);
        });
}

// Promise.allSettled for handling partial failures
function handleUsersWithFailures() {
    console.log('\n=== Promise.allSettled Pattern ===');
    
    const userIds = [1, -1, 3, -2, 5]; // Some invalid IDs
    const userPromises = userIds.map(id => fetchUserPromise(id));
    
    Promise.allSettled(userPromises)
        .then(results => {
            console.log('All promises settled:');
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    console.log(`✓ User ${index + 1}:`, result.value.name);
                } else {
                    console.log(`✗ User ${index + 1}:`, result.reason.message);
                }
            });
        });
}

/**
 * 3. Async/Await Pattern Examples
 */

// Basic async/await function
async function fetchUserAsync(userId) {
    await delay(100); // Simulate network delay
    
    if (userId <= 0) {
        throw new Error('Invalid user ID');
    }
    
    return {
        id: userId,
        name: `User${userId}`,
        email: `user${userId}@example.com`,
        lastLogin: new Date().toISOString()
    };
}

// Async/await with error handling
async function handleUserDataAsync() {
    console.log('\n=== Async/Await Pattern ===');
    
    try {
        const user = await fetchUserAsync(1);
        console.log('User fetched:', user);
        
        const nextUser = await fetchUserAsync(user.id + 1);
        console.log('Next user fetched:', nextUser);
        
        const thirdUser = await fetchUserAsync(nextUser.id + 1);
        console.log('Third user fetched:', thirdUser);
        
    } catch (error) {
        console.error('Error in async function:', error.message);
    }
}

// Async/await with concurrent operations
async function handleMultipleUsersAsync() {
    console.log('\n=== Async/Await with Concurrency ===');
    
    try {
        const userIds = [1, 2, 3, 4, 5];
        
        // Sequential approach (slower)
        console.log('Sequential fetch:');
        const startTime = Date.now();
        const sequentialUsers = [];
        for (const id of userIds) {
            const user = await fetchUserAsync(id);
            sequentialUsers.push(user);
        }
        console.log(`Sequential time: ${Date.now() - startTime}ms`);
        
        // Concurrent approach (faster)
        console.log('Concurrent fetch:');
        const concurrentStart = Date.now();
        const userPromises = userIds.map(id => fetchUserAsync(id));
        const concurrentUsers = await Promise.all(userPromises);
        console.log(`Concurrent time: ${Date.now() - concurrentStart}ms`);
        
        console.log(`Fetched ${concurrentUsers.length} users concurrently`);
        
    } catch (error) {
        console.error('Error in concurrent fetch:', error.message);
    }
}

/**
 * 4. Advanced Async Patterns
 */

// Retry mechanism with exponential backoff
async function fetchWithRetry(userId, maxRetries = 3) {
    let retries = 0;
    
    while (retries < maxRetries) {
        try {
            // Simulate occasional failures
            if (Math.random() < 0.3 && retries < 2) {
                throw new Error('Network error');
            }
            
            return await fetchUserAsync(userId);
            
        } catch (error) {
            retries++;
            
            if (retries >= maxRetries) {
                throw new Error(`Failed after ${maxRetries} retries: ${error.message}`);
            }
            
            // Exponential backoff: 1s, 2s, 4s
            const delayTime = Math.pow(2, retries - 1) * 1000;
            console.log(`Retry ${retries} after ${delayTime}ms delay`);
            await delay(delayTime);
        }
    }
}

// Timeout wrapper for async operations
function withTimeout(promise, timeoutMs) {
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Operation timed out')), timeoutMs);
    });
    
    return Promise.race([promise, timeoutPromise]);
}

// Rate limiting with async queue
class AsyncQueue {
    constructor(concurrency = 1) {
        this.concurrency = concurrency;
        this.running = 0;
        this.queue = [];
    }
    
    async add(asyncFunction) {
        return new Promise((resolve, reject) => {
            this.queue.push({
                asyncFunction,
                resolve,
                reject
            });
            
            this.process();
        });
    }
    
    async process() {
        if (this.running >= this.concurrency || this.queue.length === 0) {
            return;
        }
        
        this.running++;
        const { asyncFunction, resolve, reject } = this.queue.shift();
        
        try {
            const result = await asyncFunction();
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running--;
            this.process(); // Process next item in queue
        }
    }
}

// Demonstration function
async function demonstrateAdvancedPatterns() {
    console.log('\n=== Advanced Async Patterns ===');
    
    // Retry pattern
    try {
        console.log('Testing retry pattern:');
        const user = await fetchWithRetry(1);
        console.log('User fetched with retry:', user.name);
    } catch (error) {
        console.error('Retry failed:', error.message);
    }
    
    // Timeout pattern
    try {
        console.log('\nTesting timeout pattern:');
        const slowOperation = delay(2000).then(() => ({ data: 'slow result' }));
        const result = await withTimeout(slowOperation, 1000);
        console.log('Result:', result);
    } catch (error) {
        console.log('Timeout caught:', error.message);
    }
    
    // Rate limiting with queue
    console.log('\nTesting async queue:');
    const queue = new AsyncQueue(2); // Max 2 concurrent operations
    
    const queuePromises = [1, 2, 3, 4, 5].map(id => 
        queue.add(() => fetchUserAsync(id))
    );
    
    const queueResults = await Promise.all(queuePromises);
    console.log(`Queue processed ${queueResults.length} users`);
}

/**
 * Main execution function
 */
async function main() {
    console.log('JavaScript Async Programming Examples');
    console.log('=====================================');
    
    // Run all examples
    handleUserData();
    
    // Wait a bit for callbacks to complete
    await delay(500);
    
    handleUserDataPromises();
    await delay(500);
    
    handleMultipleUsers();
    await delay(500);
    
    handleUsersWithFailures();
    await delay(500);
    
    await handleUserDataAsync();
    await handleMultipleUsersAsync();
    await demonstrateAdvancedPatterns();
    
    console.log('\n=== All Examples Complete ===');
}

// Export functions for use in other modules
module.exports = {
    fetchUserCallback,
    fetchUserPromise,
    fetchUserAsync,
    fetchWithRetry,
    withTimeout,
    AsyncQueue,
    delay
};

// Run examples if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}