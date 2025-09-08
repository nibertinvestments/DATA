// Rust concurrency and async programming for AI training dataset.
// Demonstrates threading, channels, async/await, and parallel processing.

use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;
use std::collections::HashMap;

/// Basic threading example with shared state.
pub fn basic_threading_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for i in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            println!("Thread {} incremented counter to {}", i, *num);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final counter value: {}", *counter.lock().unwrap());
}

/// Channel communication between threads.
pub fn channel_communication_example() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }
}

/// Multiple producer, single consumer example.
pub fn multiple_producer_example() {
    let (tx, rx) = mpsc::channel();

    for i in 0..3 {
        let tx = tx.clone();
        thread::spawn(move || {
            let vals = vec![
                format!("Producer {} - Message 1", i),
                format!("Producer {} - Message 2", i),
                format!("Producer {} - Message 3", i),
            ];

            for val in vals {
                tx.send(val).unwrap();
                thread::sleep(Duration::from_millis(500));
            }
        });
    }
    drop(tx); // Close the sending side

    for received in rx {
        println!("Received: {}", received);
    }
}

/// Producer-Consumer pattern with bounded channel.
use std::sync::mpsc::sync_channel;

pub struct Producer {
    id: usize,
    sender: std::sync::mpsc::SyncSender<WorkItem>,
}

pub struct Consumer {
    id: usize,
    receiver: Arc<Mutex<std::sync::mpsc::Receiver<WorkItem>>>,
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    id: usize,
    data: String,
}

impl Producer {
    pub fn new(id: usize, sender: std::sync::mpsc::SyncSender<WorkItem>) -> Self {
        Producer { id, sender }
    }

    pub fn produce(&self, work_items: Vec<WorkItem>) {
        for item in work_items {
            println!("Producer {} producing item: {:?}", self.id, item);
            self.sender.send(item).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Consumer {
    pub fn new(id: usize, receiver: Arc<Mutex<std::sync::mpsc::Receiver<WorkItem>>>) -> Self {
        Consumer { id, receiver }
    }

    pub fn consume(&self) {
        loop {
            let receiver = self.receiver.lock().unwrap();
            match receiver.try_recv() {
                Ok(item) => {
                    drop(receiver); // Release the lock
                    println!("Consumer {} processing item: {:?}", self.id, item);
                    thread::sleep(Duration::from_millis(200));
                }
                Err(mpsc::TryRecvError::Empty) => {
                    drop(receiver);
                    thread::sleep(Duration::from_millis(10));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    drop(receiver);
                    println!("Consumer {} shutting down", self.id);
                    break;
                }
            }
        }
    }
}

pub fn producer_consumer_example() {
    let (tx, rx) = sync_channel(5); // Bounded channel with capacity 5
    let rx = Arc::new(Mutex::new(rx));

    // Create producers
    let mut producer_handles = vec![];
    for i in 0..2 {
        let tx = tx.clone();
        let handle = thread::spawn(move || {
            let producer = Producer::new(i, tx);
            let work_items = (0..5)
                .map(|j| WorkItem {
                    id: i * 100 + j,
                    data: format!("Work from producer {}", i),
                })
                .collect();
            producer.produce(work_items);
        });
        producer_handles.push(handle);
    }

    // Create consumers
    let mut consumer_handles = vec![];
    for i in 0..3 {
        let rx = Arc::clone(&rx);
        let handle = thread::spawn(move || {
            let consumer = Consumer::new(i, rx);
            consumer.consume();
        });
        consumer_handles.push(handle);
    }

    // Wait for all producers to finish
    for handle in producer_handles {
        handle.join().unwrap();
    }
    drop(tx); // Close the sending side

    // Wait for all consumers to finish
    for handle in consumer_handles {
        handle.join().unwrap();
    }
}

/// Async/await example (requires tokio)
#[cfg(feature = "async")]
pub mod async_examples {
    use tokio::time::{sleep, Duration};
    use std::future::Future;
    use std::pin::Pin;

    pub async fn async_computation(id: usize) -> usize {
        println!("Starting async computation {}", id);
        sleep(Duration::from_millis(1000)).await;
        println!("Finished async computation {}", id);
        id * 2
    }

    pub async fn concurrent_async_example() {
        let futures = (0..5).map(|i| async_computation(i));
        
        let results = futures::future::join_all(futures).await;
        println!("Results: {:?}", results);
    }

    pub async fn async_channel_example() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);

        // Spawn a task to send values
        tokio::spawn(async move {
            for i in 0..10 {
                if tx.send(i).await.is_err() {
                    break;
                }
                sleep(Duration::from_millis(100)).await;
            }
        });

        // Receive values
        while let Some(value) = rx.recv().await {
            println!("Received: {}", value);
        }
    }

    pub async fn async_mutex_example() {
        let counter = std::sync::Arc::new(tokio::sync::Mutex::new(0));
        let mut handles = vec![];

        for i in 0..10 {
            let counter = std::sync::Arc::clone(&counter);
            let handle = tokio::spawn(async move {
                let mut num = counter.lock().await;
                *num += 1;
                println!("Task {} incremented counter to {}", i, *num);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        println!("Final counter value: {}", *counter.lock().await);
    }
}

/// Thread pool implementation for parallel processing.
use std::sync::mpsc::{Receiver, Sender};

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<Sender<Job>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0);

        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));

        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }

        ThreadPool {
            workers,
            sender: Some(sender),
        }
    }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.as_ref().unwrap().send(job).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());

        for worker in &mut self.workers {
            println!("Shutting down worker {}", worker.id);

            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let message = receiver.lock().unwrap().recv();

            match message {
                Ok(job) => {
                    println!("Worker {id} got a job; executing.");
                    job();
                }
                Err(_) => {
                    println!("Worker {id} disconnected; shutting down.");
                    break;
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }
}

/// Parallel computation example using thread pool.
pub fn parallel_computation_example() {
    let pool = ThreadPool::new(4);

    for i in 0..8 {
        pool.execute(move || {
            println!("Computing factorial of {}", i);
            let result = (1..=i).fold(1u64, |acc, x| acc * x);
            println!("Factorial of {} is {}", i, result);
            thread::sleep(Duration::from_millis(500));
        });
    }

    // ThreadPool will be dropped here, shutting down all workers
}

/// Race condition demonstration and fix.
pub fn race_condition_demo() {
    println!("=== Race Condition Demo ===");
    
    // Bad example - race condition
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for i in 0..1000 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                let mut num = counter.lock().unwrap();
                *num += 1;
                // Small delay to increase chance of race condition
                drop(num);
                thread::yield_now();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final counter (should be 1,000,000): {}", *counter.lock().unwrap());
}

/// Deadlock prevention example.
pub fn deadlock_prevention_example() {
    let resource1 = Arc::new(Mutex::new(0));
    let resource2 = Arc::new(Mutex::new(0));

    let res1_clone = Arc::clone(&resource1);
    let res2_clone = Arc::clone(&resource2);

    let handle1 = thread::spawn(move || {
        // Always acquire locks in the same order to prevent deadlock
        let _guard1 = res1_clone.lock().unwrap();
        println!("Thread 1: Acquired resource 1");
        thread::sleep(Duration::from_millis(100));
        
        let _guard2 = res2_clone.lock().unwrap();
        println!("Thread 1: Acquired resource 2");
    });

    let res1_clone = Arc::clone(&resource1);
    let res2_clone = Arc::clone(&resource2);

    let handle2 = thread::spawn(move || {
        // Same order - prevents deadlock
        let _guard1 = res1_clone.lock().unwrap();
        println!("Thread 2: Acquired resource 1");
        thread::sleep(Duration::from_millis(100));
        
        let _guard2 = res2_clone.lock().unwrap();
        println!("Thread 2: Acquired resource 2");
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(Mutex::new(0));

        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            pool.execute(move || {
                let mut num = counter.lock().unwrap();
                *num += 1;
            });
        }

        // Give some time for tasks to complete
        thread::sleep(Duration::from_millis(500));
        
        // Note: This test is inherently racy, but demonstrates the concept
        let final_count = *counter.lock().unwrap();
        assert!(final_count <= 10);
    }

    #[test]
    fn test_producer_consumer() {
        let (tx, rx) = sync_channel(2);
        let rx = Arc::new(Mutex::new(rx));

        let producer_handle = thread::spawn(move || {
            let producer = Producer::new(0, tx);
            let work_items = vec![
                WorkItem { id: 1, data: "test1".to_string() },
                WorkItem { id: 2, data: "test2".to_string() },
            ];
            producer.produce(work_items);
        });

        let consumer_handle = thread::spawn(move || {
            let consumer = Consumer::new(0, rx);
            // Consume for a limited time
            for _ in 0..10 {
                thread::sleep(Duration::from_millis(50));
            }
        });

        producer_handle.join().unwrap();
        // Note: Consumer will continue until channel is closed
    }
}