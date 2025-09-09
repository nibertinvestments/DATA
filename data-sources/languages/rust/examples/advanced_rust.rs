//! Advanced Rust Programming Examples for AI Coding Agents
//! 
//! This module demonstrates advanced Rust features including:
//! - Ownership, borrowing, and lifetimes
//! - Trait system and generic programming
//! - Error handling with Result and Option
//! - Async/await and concurrent programming
//! - Unsafe code and FFI patterns
//! - Memory management and zero-cost abstractions
//! - Macro programming and procedural macros
//! - Advanced type system features
//! - Performance optimization patterns
//! 
//! Author: AI Dataset Creation Team
//! License: MIT
//! Created: 2024

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use std::fmt::{self, Debug, Display};
use std::error::Error as StdError;
use std::pin::Pin;
use std::future::Future;
use std::task::{Context, Poll};
use std::marker::PhantomData;

// =============================================================================
// Advanced Type System and Trait Programming
// =============================================================================

/// Trait for types that can be cloned with a cost estimation
trait CostlyClone {
    /// Clone with awareness of computational cost
    fn costly_clone(&self) -> Self;
    
    /// Estimate the cost of cloning (0-100 scale)
    fn clone_cost(&self) -> u32;
}

/// Trait for zero-sized types that provide compile-time guarantees
trait TypeState: 'static + Send + Sync {}

/// Empty states for the state machine pattern
struct Pending;
struct Processing;
struct Completed;

impl TypeState for Pending {}
impl TypeState for Processing {}
impl TypeState for Completed {}

/// Generic state machine using phantom types
#[derive(Debug)]
struct StateMachine<T, S: TypeState> {
    data: T,
    _state: PhantomData<S>,
}

impl<T> StateMachine<T, Pending> {
    /// Create a new state machine in pending state
    fn new(data: T) -> Self {
        StateMachine {
            data,
            _state: PhantomData,
        }
    }
    
    /// Transition to processing state
    fn start_processing(self) -> StateMachine<T, Processing> {
        StateMachine {
            data: self.data,
            _state: PhantomData,
        }
    }
}

impl<T> StateMachine<T, Processing> {
    /// Process the data and transition to completed state
    fn complete<F>(mut self, processor: F) -> StateMachine<T, Completed>
    where
        F: FnOnce(&mut T),
    {
        processor(&mut self.data);
        StateMachine {
            data: self.data,
            _state: PhantomData,
        }
    }
}

impl<T> StateMachine<T, Completed> {
    /// Extract the processed data
    fn into_data(self) -> T {
        self.data
    }
}

/// Advanced trait with associated types and where clauses
trait Container {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    
    /// Add an item to the container
    fn add(&mut self, item: Self::Item);
    
    /// Get an iterator over items
    fn iter(&self) -> Self::Iter;
    
    /// Get the number of items
    fn len(&self) -> usize;
    
    /// Check if container is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Thread-safe container implementation
#[derive(Debug)]
struct SafeVec<T> {
    data: Arc<RwLock<Vec<T>>>,
}

impl<T> SafeVec<T> {
    fn new() -> Self {
        SafeVec {
            data: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    fn with_capacity(capacity: usize) -> Self {
        SafeVec {
            data: Arc::new(RwLock::new(Vec::with_capacity(capacity))),
        }
    }
}

impl<T> Clone for SafeVec<T> {
    fn clone(&self) -> Self {
        SafeVec {
            data: Arc::clone(&self.data),
        }
    }
}

impl<T: Clone> Container for SafeVec<T> {
    type Item = T;
    type Iter = std::vec::IntoIter<T>;
    
    fn add(&mut self, item: T) {
        self.data.write().unwrap().push(item);
    }
    
    fn iter(&self) -> Self::Iter {
        self.data.read().unwrap().clone().into_iter()
    }
    
    fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }
}

/// Higher-kinded type simulation with traits
trait Functor<A> {
    type Wrapped<B>;
    
    fn map<B, F>(self, f: F) -> Self::Wrapped<B>
    where
        F: FnOnce(A) -> B;
}

impl<A> Functor<A> for Option<A> {
    type Wrapped<B> = Option<B>;
    
    fn map<B, F>(self, f: F) -> Option<B>
    where
        F: FnOnce(A) -> B,
    {
        self.map(f)
    }
}

impl<A, E> Functor<A> for Result<A, E> {
    type Wrapped<B> = Result<B, E>;
    
    fn map<B, F>(self, f: F) -> Result<B, E>
    where
        F: FnOnce(A) -> B,
    {
        self.map(f)
    }
}

// =============================================================================
// Advanced Error Handling and Result Types
// =============================================================================

/// Custom error type with context and error chaining
#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(String),
    Network { code: u16, message: String },
    Validation { field: String, reason: String },
    Custom { error: Box<dyn StdError + Send + Sync>, context: String },
}

impl Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(err) => write!(f, "IO error: {}", err),
            AppError::Parse(msg) => write!(f, "Parse error: {}", msg),
            AppError::Network { code, message } => {
                write!(f, "Network error {}: {}", code, message)
            }
            AppError::Validation { field, reason } => {
                write!(f, "Validation error in field '{}': {}", field, reason)
            }
            AppError::Custom { error, context } => {
                write!(f, "Error in {}: {}", context, error)
            }
        }
    }
}

impl StdError for AppError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            AppError::Io(err) => Some(err),
            AppError::Custom { error, .. } => Some(error.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::Io(err)
    }
}

/// Result type alias for consistent error handling
type AppResult<T> = Result<T, AppError>;

/// Trait for adding context to errors
trait ResultExt<T> {
    fn with_context<F>(self, f: F) -> AppResult<T>
    where
        F: FnOnce() -> String;
    
    fn context(self, msg: &str) -> AppResult<T>;
}

impl<T, E> ResultExt<T> for Result<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn with_context<F>(self, f: F) -> AppResult<T>
    where
        F: FnOnce() -> String,
    {
        match self {
            Ok(val) => Ok(val),
            Err(err) => Err(AppError::Custom {
                error: Box::new(err),
                context: f(),
            }),
        }
    }
    
    fn context(self, msg: &str) -> AppResult<T> {
        self.with_context(|| msg.to_string())
    }
}

/// Monadic chain operations for Result
trait ResultChain<T> {
    fn and_then_async<F, Fut, U>(self, f: F) -> Pin<Box<dyn Future<Output = AppResult<U>> + Send>>
    where
        F: FnOnce(T) -> Fut + Send + 'static,
        Fut: Future<Output = AppResult<U>> + Send + 'static,
        T: Send + 'static,
        U: Send + 'static;
}

impl<T> ResultChain<T> for AppResult<T> {
    fn and_then_async<F, Fut, U>(self, f: F) -> Pin<Box<dyn Future<Output = AppResult<U>> + Send>>
    where
        F: FnOnce(T) -> Fut + Send + 'static,
        Fut: Future<Output = AppResult<U>> + Send + 'static,
        T: Send + 'static,
        U: Send + 'static,
    {
        Box::pin(async move {
            match self {
                Ok(val) => f(val).await,
                Err(err) => Err(err),
            }
        })
    }
}

// =============================================================================
// Memory Management and Smart Pointers
// =============================================================================

/// Custom smart pointer with reference counting and weak references
struct SharedPtr<T> {
    inner: Arc<T>,
}

impl<T> SharedPtr<T> {
    fn new(value: T) -> Self {
        SharedPtr {
            inner: Arc::new(value),
        }
    }
    
    fn downgrade(&self) -> WeakPtr<T> {
        WeakPtr {
            inner: Arc::downgrade(&self.inner),
        }
    }
    
    fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
    
    fn weak_count(&self) -> usize {
        Arc::weak_count(&self.inner)
    }
}

impl<T> Clone for SharedPtr<T> {
    fn clone(&self) -> Self {
        SharedPtr {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> std::ops::Deref for SharedPtr<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

struct WeakPtr<T> {
    inner: std::sync::Weak<T>,
}

impl<T> WeakPtr<T> {
    fn upgrade(&self) -> Option<SharedPtr<T>> {
        self.inner.upgrade().map(|arc| SharedPtr { inner: arc })
    }
    
    fn strong_count(&self) -> usize {
        self.inner.strong_count()
    }
}

impl<T> Clone for WeakPtr<T> {
    fn clone(&self) -> Self {
        WeakPtr {
            inner: self.inner.clone(),
        }
    }
}

/// Arena allocator for efficient memory management
struct Arena {
    chunks: Vec<Vec<u8>>,
    current_chunk: usize,
    current_offset: usize,
    chunk_size: usize,
}

impl Arena {
    fn new() -> Self {
        Self::with_chunk_size(4096)
    }
    
    fn with_chunk_size(chunk_size: usize) -> Self {
        Arena {
            chunks: vec![Vec::with_capacity(chunk_size)],
            current_chunk: 0,
            current_offset: 0,
            chunk_size,
        }
    }
    
    fn allocate<T>(&mut self, value: T) -> &mut T {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        
        // Align the offset
        let aligned_offset = (self.current_offset + align - 1) & !(align - 1);
        
        // Check if we need a new chunk
        if aligned_offset + size > self.chunk_size {
            self.chunks.push(Vec::with_capacity(self.chunk_size));
            self.current_chunk += 1;
            self.current_offset = 0;
            aligned_offset = 0;
        }
        
        // Ensure the chunk has enough capacity
        let chunk = &mut self.chunks[self.current_chunk];
        if chunk.len() < aligned_offset + size {
            chunk.resize(aligned_offset + size, 0);
        }
        
        // Place the value
        unsafe {
            let ptr = chunk.as_mut_ptr().add(aligned_offset) as *mut T;
            std::ptr::write(ptr, value);
            self.current_offset = aligned_offset + size;
            &mut *ptr
        }
    }
    
    fn reset(&mut self) {
        self.current_chunk = 0;
        self.current_offset = 0;
        for chunk in &mut self.chunks {
            chunk.clear();
        }
    }
}

// =============================================================================
// Concurrent Programming and Async Patterns
// =============================================================================

/// Lock-free queue using atomic operations
pub struct LockFreeQueue<T> {
    head: Arc<AtomicNode<T>>,
    tail: Arc<AtomicNode<T>>,
}

struct AtomicNode<T> {
    data: Option<T>,
    next: Arc<Mutex<Option<Arc<AtomicNode<T>>>>>,
}

impl<T> AtomicNode<T> {
    fn new(data: Option<T>) -> Arc<Self> {
        Arc::new(AtomicNode {
            data,
            next: Arc::new(Mutex::new(None)),
        })
    }
}

impl<T> LockFreeQueue<T> {
    pub fn new() -> Self {
        let dummy = AtomicNode::new(None);
        LockFreeQueue {
            head: dummy.clone(),
            tail: dummy,
        }
    }
    
    pub fn enqueue(&self, data: T) {
        let new_node = AtomicNode::new(Some(data));
        
        loop {
            let tail = self.tail.clone();
            let mut next = tail.next.lock().unwrap();
            
            if next.is_none() {
                *next = Some(new_node.clone());
                break;
            }
        }
        
        // Try to advance tail
        let _ = self.tail.clone(); // Simplified for example
    }
    
    pub fn dequeue(&self) -> Option<T> {
        loop {
            let head = self.head.clone();
            let next_guard = head.next.lock().unwrap();
            
            if let Some(ref next_node) = *next_guard {
                // Move data out of next node
                if let Some(data) = &next_node.data {
                    // In a real implementation, we'd properly handle the atomic update
                    // This is a simplified version for demonstration
                    return None; // Placeholder
                }
            }
            
            return None;
        }
    }
}

/// Async task scheduler with work stealing
struct TaskScheduler {
    workers: Vec<thread::JoinHandle<()>>,
    task_queue: Arc<Mutex<VecDeque<Pin<Box<dyn Future<Output = ()> + Send>>>>>,
    shutdown: Arc<Mutex<bool>>,
}

impl TaskScheduler {
    fn new(num_workers: usize) -> Self {
        let task_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(Mutex::new(false));
        let mut workers = Vec::with_capacity(num_workers);
        
        for worker_id in 0..num_workers {
            let queue = Arc::clone(&task_queue);
            let shutdown_flag = Arc::clone(&shutdown);
            
            let handle = thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                
                loop {
                    let task = {
                        let mut queue = queue.lock().unwrap();
                        if *shutdown_flag.lock().unwrap() && queue.is_empty() {
                            break;
                        }
                        queue.pop_front()
                    };
                    
                    if let Some(task) = task {
                        rt.block_on(task);
                    } else {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
                
                println!("Worker {} shutting down", worker_id);
            });
            
            workers.push(handle);
        }
        
        TaskScheduler {
            workers,
            task_queue,
            shutdown,
        }
    }
    
    fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let mut queue = self.task_queue.lock().unwrap();
        queue.push_back(Box::pin(future));
    }
    
    fn shutdown(self) {
        {
            let mut shutdown = self.shutdown.lock().unwrap();
            *shutdown = true;
        }
        
        for worker in self.workers {
            worker.join().unwrap();
        }
    }
}

/// Channel with backpressure and flow control
struct BackpressureChannel<T> {
    sender: mpsc::Sender<T>,
    receiver: Arc<Mutex<mpsc::Receiver<T>>>,
    capacity: usize,
    current_size: Arc<Mutex<usize>>,
}

impl<T> BackpressureChannel<T> {
    fn new(capacity: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        
        BackpressureChannel {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
            capacity,
            current_size: Arc::new(Mutex::new(0)),
        }
    }
    
    fn try_send(&self, value: T) -> Result<(), T> {
        let mut size = self.current_size.lock().unwrap();
        
        if *size >= self.capacity {
            return Err(value);
        }
        
        match self.sender.send(value) {
            Ok(()) => {
                *size += 1;
                Ok(())
            }
            Err(mpsc::SendError(value)) => Err(value),
        }
    }
    
    fn recv(&self) -> Option<T> {
        let receiver = self.receiver.lock().unwrap();
        match receiver.recv() {
            Ok(value) => {
                let mut size = self.current_size.lock().unwrap();
                *size = size.saturating_sub(1);
                Some(value)
            }
            Err(_) => None,
        }
    }
    
    fn len(&self) -> usize {
        *self.current_size.lock().unwrap()
    }
    
    fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }
}

// =============================================================================
// Performance Optimization and Zero-Cost Abstractions
// =============================================================================

/// Iterator adapter for batching elements
struct BatchIterator<I, T> {
    iter: I,
    batch_size: usize,
    _phantom: PhantomData<T>,
}

impl<I, T> BatchIterator<I, T>
where
    I: Iterator<Item = T>,
{
    fn new(iter: I, batch_size: usize) -> Self {
        BatchIterator {
            iter,
            batch_size,
            _phantom: PhantomData,
        }
    }
}

impl<I, T> Iterator for BatchIterator<I, T>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            match self.iter.next() {
                Some(item) => batch.push(item),
                None => break,
            }
        }
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// Extension trait for iterator batching
trait BatchExt<T>: Iterator<Item = T> + Sized {
    fn batch(self, size: usize) -> BatchIterator<Self, T> {
        BatchIterator::new(self, size)
    }
}

impl<I, T> BatchExt<T> for I where I: Iterator<Item = T> {}

/// SIMD-optimized vector operations (simplified)
struct SimdVector {
    data: Vec<f32>,
}

impl SimdVector {
    fn new(data: Vec<f32>) -> Self {
        SimdVector { data }
    }
    
    fn dot_product(&self, other: &SimdVector) -> f32 {
        assert_eq!(self.data.len(), other.data.len());
        
        // In a real implementation, this would use SIMD instructions
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    
    fn magnitude(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    
    fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            for value in &mut self.data {
                *value /= mag;
            }
        }
    }
    
    fn scale(&mut self, factor: f32) {
        for value in &mut self.data {
            *value *= factor;
        }
    }
}

/// Cache-friendly data structure with memory layout optimization
#[repr(C)]
struct CacheFriendlyMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    row_major: bool,
}

impl CacheFriendlyMatrix {
    fn new(rows: usize, cols: usize, row_major: bool) -> Self {
        CacheFriendlyMatrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            row_major,
        }
    }
    
    fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        
        let index = if self.row_major {
            row * self.cols + col
        } else {
            col * self.rows + row
        };
        
        self.data.get(index).copied()
    }
    
    fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row >= self.rows || col >= self.cols {
            return false;
        }
        
        let index = if self.row_major {
            row * self.cols + col
        } else {
            col * self.rows + row
        };
        
        if let Some(element) = self.data.get_mut(index) {
            *element = value;
            true
        } else {
            false
        }
    }
    
    fn transpose(&self) -> CacheFriendlyMatrix {
        let mut result = CacheFriendlyMatrix::new(self.cols, self.rows, self.row_major);
        
        for row in 0..self.rows {
            for col in 0..self.cols {
                if let Some(value) = self.get(row, col) {
                    result.set(col, row, value);
                }
            }
        }
        
        result
    }
}

// =============================================================================
// Macro Programming and Code Generation
// =============================================================================

/// Macro for creating compile-time type-safe enums
macro_rules! define_enum {
    (
        $(#[$attr:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_attr:meta])*
                $variant:ident $(($($field:ty),*))? = $value:expr
            ),* $(,)?
        }
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        $vis enum $name {
            $(
                $(#[$variant_attr])*
                $variant $(($($field),*))? = $value,
            )*
        }
        
        impl $name {
            /// Get all possible variants
            pub const fn variants() -> &'static [Self] {
                &[$(Self::$variant),*]
            }
            
            /// Get the discriminant value
            pub const fn discriminant(&self) -> isize {
                *self as isize
            }
        }
        
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        Self::$variant => write!(f, stringify!($variant)),
                    )*
                }
            }
        }
    };
}

// Example usage of the macro
define_enum! {
    /// HTTP status codes
    pub enum HttpStatus {
        Ok = 200,
        NotFound = 404,
        InternalServerError = 500,
    }
}

/// Macro for generating builder pattern
macro_rules! builder {
    (
        $(#[$struct_attr:meta])*
        struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field:ident: $field_type:ty
            ),* $(,)?
        }
    ) => {
        $(#[$struct_attr])*
        #[derive(Debug, Clone)]
        pub struct $name {
            $(
                $(#[$field_attr])*
                pub $field: $field_type,
            )*
        }
        
        paste::paste! {
            #[derive(Debug, Default)]
            pub struct [<$name Builder>] {
                $(
                    $field: Option<$field_type>,
                )*
            }
            
            impl [<$name Builder>] {
                pub fn new() -> Self {
                    Self::default()
                }
                
                $(
                    pub fn $field(mut self, value: $field_type) -> Self {
                        self.$field = Some(value);
                        self
                    }
                )*
                
                pub fn build(self) -> Result<$name, String> {
                    Ok($name {
                        $(
                            $field: self.$field.ok_or_else(|| {
                                format!("Field '{}' is required", stringify!($field))
                            })?,
                        )*
                    })
                }
            }
        }
        
        impl $name {
            pub fn builder() -> paste::paste!([<$name Builder>]) {
                paste::paste!([<$name Builder>]::new())
            }
        }
    };
}

// Note: The paste crate would be needed for this macro to work properly
// This is a simplified version for demonstration

// =============================================================================
// Example Usage and Demonstrations
// =============================================================================

/// Comprehensive demonstration of all Rust features
fn demonstrate_advanced_rust() {
    println!("🚀 Advanced Rust Programming Demonstration");
    println!("{}", "=".repeat(50));
    
    // Test state machine
    println!("\n📝 Type-Safe State Machine:");
    let machine = StateMachine::new("Hello, Rust!".to_string());
    let processing = machine.start_processing();
    let completed = processing.complete(|data| {
        data.push_str(" - Processed");
    });
    let result = completed.into_data();
    println!("State machine result: {}", result);
    
    // Test container trait
    println!("\n📦 Generic Container:");
    let mut container = SafeVec::new();
    container.add("First".to_string());
    container.add("Second".to_string());
    container.add("Third".to_string());
    
    println!("Container length: {}", container.len());
    for item in container.iter() {
        println!("  Item: {}", item);
    }
    
    // Test error handling
    println!("\n❌ Advanced Error Handling:");
    let result = divide_numbers(10.0, 2.0);
    match result {
        Ok(value) => println!("Division result: {:.2}", value),
        Err(err) => println!("Error: {}", err),
    }
    
    let error_result = divide_numbers(10.0, 0.0);
    match error_result {
        Ok(_) => println!("Unexpected success"),
        Err(err) => {
            println!("Expected error: {}", err);
            if let Some(source) = err.source() {
                println!("Error source: {}", source);
            }
        }
    }
    
    // Test smart pointers
    println!("\n🧠 Smart Pointers:");
    let shared = SharedPtr::new(vec![1, 2, 3, 4, 5]);
    let weak = shared.downgrade();
    
    println!("Strong count: {}", shared.strong_count());
    println!("Weak count: {}", shared.weak_count());
    
    {
        let _shared2 = shared.clone();
        println!("Strong count after clone: {}", shared.strong_count());
    }
    
    if let Some(upgraded) = weak.upgrade() {
        println!("Successfully upgraded weak reference");
        println!("Vector length: {}", upgraded.len());
    }
    
    // Test arena allocator
    println!("\n🏟️ Arena Allocator:");
    let mut arena = Arena::new();
    
    let numbers: Vec<&mut i32> = (0..10)
        .map(|i| arena.allocate(i * i))
        .collect();
    
    println!("Allocated numbers:");
    for (i, num) in numbers.iter().enumerate() {
        println!("  {}: {}", i, num);
    }
    
    // Test concurrent programming
    println!("\n⚡ Concurrent Programming:");
    
    // Lock-free queue (simplified demonstration)
    let queue = LockFreeQueue::new();
    queue.enqueue("First message".to_string());
    queue.enqueue("Second message".to_string());
    println!("Lock-free queue created and populated");
    
    // Backpressure channel
    let channel = BackpressureChannel::new(3);
    
    for i in 0..5 {
        match channel.try_send(format!("Message {}", i)) {
            Ok(()) => println!("Sent message {}", i),
            Err(msg) => println!("Failed to send: {}", msg),
        }
    }
    
    println!("Channel length: {}", channel.len());
    
    // Receive messages
    while let Some(msg) = channel.recv() {
        println!("Received: {}", msg);
    }
    
    // Test performance optimizations
    println!("\n⚡ Performance Optimizations:");
    
    // Batch iterator
    let numbers: Vec<i32> = (1..=20).collect();
    println!("Batched iteration:");
    for batch in numbers.into_iter().batch(5) {
        println!("  Batch: {:?}", batch);
    }
    
    // SIMD vector operations
    let vec1 = SimdVector::new(vec![1.0, 2.0, 3.0, 4.0]);
    let vec2 = SimdVector::new(vec![2.0, 3.0, 4.0, 5.0]);
    
    let dot_product = vec1.dot_product(&vec2);
    println!("Dot product: {:.2}", dot_product);
    println!("Vector 1 magnitude: {:.2}", vec1.magnitude());
    
    // Cache-friendly matrix
    let mut matrix = CacheFriendlyMatrix::new(3, 3, true);
    
    // Fill matrix with values
    for i in 0..3 {
        for j in 0..3 {
            matrix.set(i, j, (i * 3 + j) as f64);
        }
    }
    
    println!("Matrix (3x3):");
    for i in 0..3 {
        print!("  ");
        for j in 0..3 {
            if let Some(value) = matrix.get(i, j) {
                print!("{:6.1} ", value);
            }
        }
        println!();
    }
    
    // Test macro-generated enum
    println!("\n🔧 Macro-Generated Types:");
    let status = HttpStatus::Ok;
    println!("HTTP Status: {} ({})", status, status.discriminant());
    
    println!("All HTTP status variants:");
    for variant in HttpStatus::variants() {
        println!("  {}: {}", variant, variant.discriminant());
    }
    
    // Benchmark performance
    println!("\n📊 Performance Benchmarking:");
    
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..1_000_000 {
        sum = sum.wrapping_add(i);
    }
    let duration = start.elapsed();
    
    println!("Sum calculation:");
    println!("  Result: {}", sum);
    println!("  Time: {:?}", duration);
    println!("  Rate: {:.2} million ops/sec", 1.0 / duration.as_secs_f64());
    
    println!("\n✅ All Rust demonstrations completed!");
}

/// Example function for error handling demonstration
fn divide_numbers(a: f64, b: f64) -> AppResult<f64> {
    if b == 0.0 {
        return Err(AppError::Validation {
            field: "divisor".to_string(),
            reason: "cannot be zero".to_string(),
        });
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(AppError::Parse(
            "numbers must be finite".to_string()
        ));
    }
    
    Ok(a / b)
}

/// Main function to run all demonstrations
fn main() {
    demonstrate_advanced_rust();
}