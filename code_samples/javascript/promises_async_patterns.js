/**
 * Advanced Promise and Async Patterns in JavaScript
 * Demonstrates promise composition, error handling, and async utilities
 */

// Promise utility functions
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Retry with exponential backoff
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            
            const delayTime = baseDelay * Math.pow(2, i);
            console.log(`Retry ${i + 1} after ${delayTime}ms`);
            await delay(delayTime);
        }
    }
}

// Promise.all with concurrency limit
async function promiseAllLimit(promises, limit) {
    const results = [];
    const executing = [];
    
    for (const [index, promise] of promises.entries()) {
        const p = Promise.resolve(promise).then(
            result => {
                results[index] = result;
            }
        );
        
        results.push(undefined);
        
        if (limit <= promises.length) {
            const e = p.then(() => executing.splice(executing.indexOf(e), 1));
            executing.push(e);
            
            if (executing.length >= limit) {
                await Promise.race(executing);
            }
        }
    }
    
    await Promise.all(executing);
    return results;
}

// Promise timeout wrapper
function withTimeout(promise, timeoutMs) {
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Timeout')), timeoutMs)
        )
    ]);
}

// Async queue implementation
class AsyncQueue {
    constructor(concurrency = 1) {
        this.concurrency = concurrency;
        this.running = 0;
        this.queue = [];
    }
    
    async add(fn) {
        return new Promise((resolve, reject) => {
            this.queue.push({ fn, resolve, reject });
            this.process();
        });
    }
    
    async process() {
        if (this.running >= this.concurrency || this.queue.length === 0) {
            return;
        }
        
        this.running++;
        const { fn, resolve, reject } = this.queue.shift();
        
        try {
            const result = await fn();
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running--;
            this.process();
        }
    }
}

// Async generator example
async function* asyncGenerator(items) {
    for (const item of items) {
        await delay(100);
        yield item * 2;
    }
}

// Pipeline composition
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const pipeAsync = (...fns) => x => fns.reduce(async (v, f) => f(await v), x);

// Demonstration
async function main() {
    console.log("Advanced Promise Patterns");
    console.log("========================\n");
    
    // Test retry with backoff
    console.log("1. Retry with backoff:");
    let attempts = 0;
    const flaky = async () => {
        attempts++;
        if (attempts < 3) throw new Error("Fail");
        return "Success";
    };
    const result = await retryWithBackoff(flaky);
    console.log(`Result: ${result}\n`);
    
    // Test async queue
    console.log("2. Async queue:");
    const queue = new AsyncQueue(2);
    const tasks = [1, 2, 3, 4, 5].map(n =>
        () => delay(100).then(() => n * 2)
    );
    const queueResults = await Promise.all(tasks.map(task => queue.add(task)));
    console.log(`Queue results: ${queueResults}\n`);
    
    // Test async generator
    console.log("3. Async generator:");
    for await (const value of asyncGenerator([1, 2, 3])) {
        console.log(`  Generated: ${value}`);
    }
}

main().catch(console.error);
