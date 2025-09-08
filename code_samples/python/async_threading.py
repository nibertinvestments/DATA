"""
Multi-threading and Asynchronous Programming Examples in Python
Demonstrates threading, asyncio, concurrent.futures, and parallel processing.
"""

import asyncio
import threading
import concurrent.futures
import time
import random
import queue
import multiprocessing
from typing import List, Callable, Any, Optional
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import aiofiles  # Would require: pip install aiofiles
import requests
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a computational task."""
    id: int
    data: Any
    duration: float = 1.0
    
    def execute(self) -> Any:
        """Execute the task (simulate work)."""
        time.sleep(self.duration)
        return f"Task {self.id} completed with data: {self.data}"


class ThreadSafeCounter:
    """Thread-safe counter using locks."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter safely."""
        with self._lock:
            self._value += amount
            return self._value
    
    def get_value(self) -> int:
        """Get current value safely."""
        with self._lock:
            return self._value


class WorkerThread(threading.Thread):
    """Custom worker thread class."""
    
    def __init__(self, task_queue: queue.Queue, result_queue: queue.Queue, name: str):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.shutdown_flag = threading.Event()
    
    def run(self):
        """Main thread execution loop."""
        logger.info(f"Worker {self.name} started")
        
        while not self.shutdown_flag.is_set():
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=1.0)
                
                # Execute task
                result = task.execute()
                self.result_queue.put(result)
                
                # Mark task as done
                self.task_queue.task_done()
                
                logger.info(f"Worker {self.name} completed task {task.id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.name} error: {e}")
        
        logger.info(f"Worker {self.name} stopped")
    
    def shutdown(self):
        """Signal thread to shutdown."""
        self.shutdown_flag.set()


class ThreadPoolManager:
    """Custom thread pool manager."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers: List[WorkerThread] = []
        self.counter = ThreadSafeCounter()
    
    def start(self):
        """Start all worker threads."""
        for i in range(self.num_workers):
            worker = WorkerThread(
                self.task_queue, 
                self.result_queue, 
                f"Worker-{i+1}"
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} worker threads")
    
    def submit_task(self, task: Task):
        """Submit a task to the pool."""
        self.task_queue.put(task)
        self.counter.increment()
    
    def get_results(self, timeout: float = None) -> List[str]:
        """Get all available results."""
        results = []
        
        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def shutdown(self, wait: bool = True):
        """Shutdown all worker threads."""
        # Signal all workers to shutdown
        for worker in self.workers:
            worker.shutdown()
        
        if wait:
            # Wait for all workers to finish
            for worker in self.workers:
                worker.join()
        
        logger.info("All workers shut down")


# Asyncio Examples
class AsyncTaskProcessor:
    """Asynchronous task processor."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_task(self, task: Task) -> str:
        """Process a single task asynchronously."""
        async with self.semaphore:
            logger.info(f"Starting async task {task.id}")
            
            # Simulate async work
            await asyncio.sleep(task.duration)
            
            result = f"Async task {task.id} completed with data: {task.data}"
            logger.info(f"Completed async task {task.id}")
            
            return result
    
    async def process_tasks(self, tasks: List[Task]) -> List[str]:
        """Process multiple tasks concurrently."""
        # Create coroutines for all tasks
        coroutines = [self.process_task(task) for task in tasks]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        
        return successful_results


class AsyncWebClient:
    """Asynchronous web client for making HTTP requests."""
    
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        # In real implementation, would use aiohttp.ClientSession()
        self.session = "mock_session"  # Mock for demonstration
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # In real implementation, would close the session
        self.session = None
    
    async def fetch_url(self, url: str) -> dict:
        """Fetch a single URL asynchronously."""
        async with self.semaphore:
            logger.info(f"Fetching: {url}")
            
            # Simulate network request
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Mock response
            return {
                "url": url,
                "status": 200,
                "content_length": random.randint(1000, 10000),
                "response_time": random.uniform(0.5, 2.0)
            }
    
    async def fetch_multiple_urls(self, urls: List[str]) -> List[dict]:
        """Fetch multiple URLs concurrently."""
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        
        return successful_results


# Producer-Consumer Pattern
class ProducerConsumerExample:
    """Producer-Consumer pattern with threading."""
    
    def __init__(self, buffer_size: int = 10):
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.shutdown_event = threading.Event()
        self.produced_count = ThreadSafeCounter()
        self.consumed_count = ThreadSafeCounter()
    
    def producer(self, producer_id: int, num_items: int):
        """Producer function."""
        logger.info(f"Producer {producer_id} started")
        
        for i in range(num_items):
            if self.shutdown_event.is_set():
                break
            
            item = f"Producer-{producer_id}-Item-{i}"
            self.buffer.put(item)
            self.produced_count.increment()
            
            logger.info(f"Producer {producer_id} produced: {item}")
            time.sleep(random.uniform(0.1, 0.5))
        
        logger.info(f"Producer {producer_id} finished")
    
    def consumer(self, consumer_id: int):
        """Consumer function."""
        logger.info(f"Consumer {consumer_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                item = self.buffer.get(timeout=1.0)
                self.consumed_count.increment()
                
                # Simulate processing
                time.sleep(random.uniform(0.1, 0.3))
                
                logger.info(f"Consumer {consumer_id} consumed: {item}")
                self.buffer.task_done()
                
            except queue.Empty:
                continue
        
        logger.info(f"Consumer {consumer_id} finished")
    
    def run_simulation(self, num_producers: int = 2, num_consumers: int = 3, items_per_producer: int = 5):
        """Run producer-consumer simulation."""
        threads = []
        
        # Start producers
        for i in range(num_producers):
            producer_thread = threading.Thread(
                target=self.producer,
                args=(i, items_per_producer),
                name=f"Producer-{i}"
            )
            producer_thread.start()
            threads.append(producer_thread)
        
        # Start consumers
        for i in range(num_consumers):
            consumer_thread = threading.Thread(
                target=self.consumer,
                args=(i,),
                name=f"Consumer-{i}"
            )
            consumer_thread.start()
            threads.append(consumer_thread)
        
        # Wait for producers to finish
        for thread in threads[:num_producers]:
            thread.join()
        
        # Wait for all items to be consumed
        self.buffer.join()
        
        # Shutdown consumers
        self.shutdown_event.set()
        
        # Wait for consumers to finish
        for thread in threads[num_producers:]:
            thread.join()
        
        logger.info(f"Simulation complete. Produced: {self.produced_count.get_value()}, "
                   f"Consumed: {self.consumed_count.get_value()}")


# Multiprocessing Examples
def cpu_intensive_task(n: int) -> int:
    """CPU-intensive task for multiprocessing demonstration."""
    total = 0
    for i in range(n):
        total += i * i
    return total


class MultiprocessingManager:
    """Manager for multiprocessing tasks."""
    
    @staticmethod
    def run_with_process_pool(tasks: List[int], max_workers: int = None) -> List[int]:
        """Run CPU-intensive tasks with process pool."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(cpu_intensive_task, task): task for task in tasks}
            
            results = []
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Task {task} completed with result: {result}")
                except Exception as e:
                    logger.error(f"Task {task} generated exception: {e}")
            
            return results
    
    @staticmethod
    def run_with_thread_pool(tasks: List[Task], max_workers: int = None) -> List[str]:
        """Run I/O-bound tasks with thread pool."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(task.execute) for task in tasks]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task generated exception: {e}")
            
            return results


# Decorators for async and threading
def async_timeout(seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise
        return wrapper
    return decorator


def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


# Demo functions
def demonstrate_threading():
    """Demonstrate threading concepts."""
    print("=== Threading Demonstration ===")
    
    # Custom thread pool
    pool = ThreadPoolManager(num_workers=3)
    pool.start()
    
    # Submit tasks
    tasks = [Task(i, f"data_{i}", random.uniform(0.5, 1.5)) for i in range(5)]
    
    for task in tasks:
        pool.submit_task(task)
    
    # Wait a bit and get results
    time.sleep(3)
    results = pool.get_results()
    
    print(f"Completed {len(results)} tasks")
    for result in results:
        print(f"  {result}")
    
    pool.shutdown()


async def demonstrate_asyncio():
    """Demonstrate asyncio concepts."""
    print("\n=== Asyncio Demonstration ===")
    
    # Process tasks asynchronously
    processor = AsyncTaskProcessor(max_concurrent=3)
    tasks = [Task(i, f"async_data_{i}", random.uniform(0.5, 1.0)) for i in range(5)]
    
    start_time = time.time()
    results = await processor.process_tasks(tasks)
    end_time = time.time()
    
    print(f"Async processing completed in {end_time - start_time:.2f} seconds")
    print(f"Completed {len(results)} tasks")
    
    # Demonstrate async web client
    async with AsyncWebClient(max_concurrent=3) as client:
        urls = [
            "https://example.com/api/1",
            "https://example.com/api/2",
            "https://example.com/api/3",
            "https://example.com/api/4"
        ]
        
        web_results = await client.fetch_multiple_urls(urls)
        print(f"\nFetched {len(web_results)} URLs")
        for result in web_results:
            print(f"  {result['url']}: {result['status']} ({result['content_length']} bytes)")


def demonstrate_producer_consumer():
    """Demonstrate producer-consumer pattern."""
    print("\n=== Producer-Consumer Pattern ===")
    
    pc_example = ProducerConsumerExample(buffer_size=5)
    pc_example.run_simulation(
        num_producers=2,
        num_consumers=2,
        items_per_producer=3
    )


def demonstrate_multiprocessing():
    """Demonstrate multiprocessing concepts."""
    print("\n=== Multiprocessing Demonstration ===")
    
    # CPU-intensive tasks
    cpu_tasks = [100000, 200000, 150000, 175000]
    
    # Run with process pool (good for CPU-bound tasks)
    start_time = time.time()
    mp_manager = MultiprocessingManager()
    cpu_results = mp_manager.run_with_process_pool(cpu_tasks, max_workers=2)
    cpu_time = time.time() - start_time
    
    print(f"Multiprocessing completed in {cpu_time:.2f} seconds")
    print(f"Results: {cpu_results}")
    
    # Run with thread pool (good for I/O-bound tasks)
    io_tasks = [Task(i, f"io_data_{i}", 0.5) for i in range(4)]
    
    start_time = time.time()
    io_results = mp_manager.run_with_thread_pool(io_tasks, max_workers=4)
    io_time = time.time() - start_time
    
    print(f"Thread pool completed in {io_time:.2f} seconds")
    print(f"Completed {len(io_results)} I/O tasks")


if __name__ == "__main__":
    print("=== Multi-threading and Asynchronous Programming Examples ===\n")
    
    # Run demonstrations
    demonstrate_threading()
    
    # Run async demonstration
    asyncio.run(demonstrate_asyncio())
    
    demonstrate_producer_consumer()
    demonstrate_multiprocessing()
    
    print("\n=== Features Demonstrated ===")
    print("- Custom thread pools and worker threads")
    print("- Thread-safe data structures")
    print("- Asyncio for concurrent I/O operations")
    print("- Producer-consumer patterns")
    print("- Multiprocessing for CPU-bound tasks")
    print("- Context managers for resource management")
    print("- Error handling and timeouts")
    print("- Decorators for async and retry logic")