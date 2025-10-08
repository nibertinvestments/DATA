"""
Advanced Multiprocessing Patterns in Python
Demonstrates process pools, shared memory, and IPC
"""

import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue, Process, Value, Array
import time
import os

def worker_function(x):
    """Simple worker function for pool"""
    return x * x

def parallel_map_reduce():
    """Demonstrate map-reduce with process pool"""
    with Pool(processes=4) as pool:
        # Map phase
        numbers = range(1, 100)
        squared = pool.map(worker_function, numbers)
        
        # Reduce phase
        total = sum(squared)
        return total

def shared_memory_example():
    """Demonstrate shared memory between processes"""
    # Shared value
    counter = Value('i', 0)
    
    # Shared array
    shared_arr = Array('d', [0.0] * 10)
    
    def increment_counter(val, arr, idx):
        with val.get_lock():
            val.value += 1
        arr[idx] = val.value * 2.0
    
    processes = []
    for i in range(5):
        p = Process(target=increment_counter, args=(counter, shared_arr, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f"Final counter value: {counter.value}")
    print(f"Shared array: {list(shared_arr[:5])}")

def producer_consumer_pattern():
    """Demonstrate producer-consumer with Queue"""
    def producer(queue, items):
        for item in items:
            queue.put(item)
            time.sleep(0.1)
        queue.put(None)  # Sentinel
    
    def consumer(queue):
        results = []
        while True:
            item = queue.get()
            if item is None:
                break
            results.append(item * 2)
        return results
    
    q = Queue()
    items = range(10)
    
    prod = Process(target=producer, args=(q, items))
    cons = Process(target=consumer, args=(q,))
    
    prod.start()
    cons.start()
    
    prod.join()
    cons.join()

def parallel_computation():
    """Compute intensive task with parallel execution"""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    numbers = range(1, 10000)
    
    # Serial execution
    start = time.time()
    serial_primes = [n for n in numbers if is_prime(n)]
    serial_time = time.time() - start
    
    # Parallel execution
    start = time.time()
    with Pool() as pool:
        results = pool.map(is_prime, numbers)
        parallel_primes = [n for n, is_p in zip(numbers, results) if is_p]
    parallel_time = time.time() - start
    
    print(f"Serial time: {serial_time:.4f}s")
    print(f"Parallel time: {parallel_time:.4f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
    print(f"Found {len(parallel_primes)} primes")

if __name__ == "__main__":
    print("Advanced Multiprocessing Patterns")
    print("=================================\n")
    
    print("1. Parallel Map-Reduce:")
    result = parallel_map_reduce()
    print(f"   Sum of squares: {result}\n")
    
    print("2. Shared Memory:")
    shared_memory_example()
    print()
    
    print("3. Producer-Consumer Pattern:")
    producer_consumer_pattern()
    print()
    
    print("4. Parallel Computation:")
    parallel_computation()
