/**
 * Modern C++ Advanced Examples for AI Coding Agents
 * =================================================
 * 
 * This module demonstrates advanced C++ features including:
 * - Modern C++17/20 features and best practices
 * - Template metaprogramming and SFINAE
 * - Memory management with smart pointers
 * - Concurrent programming with std::thread and std::async
 * - Custom allocators and memory pools
 * - RAII patterns and exception safety
 * - STL algorithms and functional programming
 * - Performance optimization techniques
 * 
 * Author: AI Dataset Creation Team
 * License: MIT
 * Created: 2024
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <optional>
#include <variant>
#include <type_traits>
#include <concepts>
#include <ranges>
#include <span>
#include <queue>
#include <numeric>
#include <execution>

// =============================================================================
// Modern C++20 Concepts and Template Metaprogramming
// =============================================================================

namespace ModernCpp {
    
    /**
     * C++20 Concepts for type constraints and better error messages.
     */
    template<typename T>
    concept Numeric = std::is_arithmetic_v<T>;
    
    template<typename T>
    concept Container = requires(T t) {
        t.begin();
        t.end();
        t.size();
    };
    
    template<typename T>
    concept Hashable = requires(T t) {
        std::hash<T>{}(t);
    };
    
    /**
     * Advanced template metaprogramming with SFINAE and concepts.
     */
    template<Numeric T>
    class Matrix {
    private:
        std::vector<T> data_;
        size_t rows_, cols_;
        
    public:
        /**
         * Constructor with perfect forwarding and concepts.
         */
        template<Container C>
        requires std::convertible_to<typename C::value_type, T>
        Matrix(size_t rows, size_t cols, const C& initial_data) 
            : rows_(rows), cols_(cols) {
            data_.reserve(rows * cols);
            
            auto it = initial_data.begin();
            for (size_t i = 0; i < rows * cols && it != initial_data.end(); ++i, ++it) {
                data_.emplace_back(static_cast<T>(*it));
            }
            
            // Fill remaining with default values if needed
            data_.resize(rows * cols, T{});
        }
        
        /**
         * Default constructor with zero initialization.
         */
        Matrix(size_t rows, size_t cols, T default_value = T{}) 
            : data_(rows * cols, default_value), rows_(rows), cols_(cols) {}
        
        /**
         * Element access with bounds checking.
         */
        T& operator()(size_t row, size_t col) {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_[row * cols_ + col];
        }
        
        const T& operator()(size_t row, size_t col) const {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_[row * cols_ + col];
        }
        
        /**
         * Matrix multiplication with compile-time optimization.
         */
        template<Numeric U>
        requires std::convertible_to<U, T>
        auto operator*(const Matrix<U>& other) const {
            if (cols_ != other.rows_) {
                throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
            }
            
            using ResultType = decltype(T{} * U{});
            Matrix<ResultType> result(rows_, other.cols_);
            
            // Cache-friendly matrix multiplication
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t k = 0; k < cols_; ++k) {
                    for (size_t j = 0; j < other.cols_; ++j) {
                        result(i, j) += (*this)(i, k) * other(k, j);
                    }
                }
            }
            
            return result;
        }
        
        /**
         * Apply function to all elements with perfect forwarding.
         */
        template<typename F>
        requires std::invocable<F, T&>
        void transform(F&& func) {
            std::for_each(data_.begin(), data_.end(), std::forward<F>(func));
        }
        
        /**
         * Parallel transform using execution policies (C++17).
         */
        template<typename F>
        requires std::invocable<F, T&>
        void parallel_transform(F&& func) {
            // Note: std::execution requires linking with TBB on some systems
            #ifdef __cpp_lib_execution
            std::for_each(std::execution::par_unseq, 
                         data_.begin(), data_.end(), 
                         std::forward<F>(func));
            #else
            // Fallback to sequential execution
            std::for_each(data_.begin(), data_.end(), std::forward<F>(func));
            #endif
        }
        
        // Getters
        size_t rows() const noexcept { return rows_; }
        size_t cols() const noexcept { return cols_; }
        
        /**
         * Range-based iteration support.
         */
        auto begin() noexcept { return data_.begin(); }
        auto end() noexcept { return data_.end(); }
        auto begin() const noexcept { return data_.begin(); }
        auto end() const noexcept { return data_.end(); }
    };
    
    /**
     * Variadic template function with fold expressions (C++17).
     */
    template<Numeric... Args>
    constexpr auto sum(Args... args) {
        return (args + ...);  // C++17 fold expression
    }
    
    /**
     * Compile-time factorial using constexpr.
     */
    constexpr uint64_t factorial(int n) {
        return n <= 1 ? 1 : n * factorial(n - 1);
    }
    
    /**
     * Type trait to detect if type has a specific member function.
     */
    template<typename T>
    concept HasToString = requires(T t) {
        { t.to_string() } -> std::convertible_to<std::string>;
    };
    
    /**
     * Generic print function using concepts.
     */
    template<typename T>
    void print(const T& value) {
        if constexpr (HasToString<T>) {
            std::cout << value.to_string() << std::endl;
        } else if constexpr (std::is_arithmetic_v<T>) {
            std::cout << value << std::endl;
        } else {
            std::cout << "[Object of type " << typeid(T).name() << "]" << std::endl;
        }
    }
}

// =============================================================================
// Advanced Memory Management and RAII
// =============================================================================

namespace MemoryManagement {
    
    /**
     * Custom memory pool allocator for performance-critical applications.
     */
    template<typename T, size_t PoolSize = 1024>
    class PoolAllocator {
    private:
        alignas(T) char pool_[PoolSize * sizeof(T)];
        std::vector<bool> used_;
        mutable std::mutex mutex_;
        
    public:
        using value_type = T;
        
        PoolAllocator() : used_(PoolSize, false) {}
        
        /**
         * Thread-safe allocation from pool.
         */
        T* allocate(size_t n) {
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (n != 1) {
                throw std::bad_alloc{}; // Only single object allocation supported
            }
            
            for (size_t i = 0; i < PoolSize; ++i) {
                if (!used_[i]) {
                    used_[i] = true;
                    return reinterpret_cast<T*>(pool_ + i * sizeof(T));
                }
            }
            
            throw std::bad_alloc{}; // Pool exhausted
        }
        
        /**
         * Thread-safe deallocation back to pool.
         */
        void deallocate(T* p, size_t n) noexcept {
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (n != 1 || !p) return;
            
            char* char_p = reinterpret_cast<char*>(p);
            if (char_p >= pool_ && char_p < pool_ + sizeof(pool_)) {
                size_t index = (char_p - pool_) / sizeof(T);
                if (index < PoolSize) {
                    used_[index] = false;
                }
            }
        }
        
        /**
         * Get pool statistics.
         */
        std::pair<size_t, size_t> get_stats() const {
            std::lock_guard<std::mutex> lock(mutex_);
            size_t used_count = std::count(used_.begin(), used_.end(), true);
            return {used_count, PoolSize - used_count};
        }
    };
    
    /**
     * RAII wrapper for C-style resources with custom deleter.
     */
    template<typename T, typename Deleter = std::default_delete<T>>
    class RAIIWrapper {
    private:
        std::unique_ptr<T, Deleter> resource_;
        
    public:
        template<typename... Args>
        explicit RAIIWrapper(Args&&... args) 
            : resource_(std::make_unique<T>(std::forward<Args>(args)...)) {}
        
        RAIIWrapper(T* ptr, Deleter deleter = Deleter{})
            : resource_(ptr, deleter) {}
        
        // Move-only semantics
        RAIIWrapper(const RAIIWrapper&) = delete;
        RAIIWrapper& operator=(const RAIIWrapper&) = delete;
        
        RAIIWrapper(RAIIWrapper&&) = default;
        RAIIWrapper& operator=(RAIIWrapper&&) = default;
        
        T* get() const noexcept { return resource_.get(); }
        T& operator*() const { return *resource_; }
        T* operator->() const noexcept { return resource_.get(); }
        
        explicit operator bool() const noexcept { return resource_ != nullptr; }
        
        /**
         * Release ownership and return raw pointer.
         */
        T* release() noexcept { return resource_.release(); }
        
        /**
         * Reset with new resource.
         */
        void reset(T* ptr = nullptr) { resource_.reset(ptr); }
    };
    
    /**
     * Smart pointer with reference counting and thread safety.
     */
    template<typename T>
    class shared_ptr_ts {
    private:
        T* ptr_;
        std::atomic<size_t>* ref_count_;
        
        void release() {
            if (ref_count_ && ref_count_->fetch_sub(1) == 1) {
                delete ptr_;
                delete ref_count_;
            }
        }
        
    public:
        explicit shared_ptr_ts(T* ptr = nullptr) 
            : ptr_(ptr), ref_count_(ptr ? new std::atomic<size_t>(1) : nullptr) {}
        
        shared_ptr_ts(const shared_ptr_ts& other) 
            : ptr_(other.ptr_), ref_count_(other.ref_count_) {
            if (ref_count_) {
                ref_count_->fetch_add(1);
            }
        }
        
        shared_ptr_ts(shared_ptr_ts&& other) noexcept
            : ptr_(other.ptr_), ref_count_(other.ref_count_) {
            other.ptr_ = nullptr;
            other.ref_count_ = nullptr;
        }
        
        ~shared_ptr_ts() { release(); }
        
        shared_ptr_ts& operator=(const shared_ptr_ts& other) {
            if (this != &other) {
                release();
                ptr_ = other.ptr_;
                ref_count_ = other.ref_count_;
                if (ref_count_) {
                    ref_count_->fetch_add(1);
                }
            }
            return *this;
        }
        
        shared_ptr_ts& operator=(shared_ptr_ts&& other) noexcept {
            if (this != &other) {
                release();
                ptr_ = other.ptr_;
                ref_count_ = other.ref_count_;
                other.ptr_ = nullptr;
                other.ref_count_ = nullptr;
            }
            return *this;
        }
        
        T* get() const noexcept { return ptr_; }
        T& operator*() const { return *ptr_; }
        T* operator->() const noexcept { return ptr_; }
        
        size_t use_count() const noexcept {
            return ref_count_ ? ref_count_->load() : 0;
        }
        
        explicit operator bool() const noexcept { return ptr_ != nullptr; }
    };
}

// =============================================================================
// Concurrent Programming and Thread Safety
// =============================================================================

namespace ConcurrentProgramming {
    
    /**
     * Thread-safe queue with blocking operations.
     */
    template<typename T>
    class ThreadSafeQueue {
    private:
        mutable std::mutex mutex_;
        std::queue<T> queue_;
        std::condition_variable condition_;
        std::atomic<bool> shutdown_{false};
        
    public:
        /**
         * Add item to queue and notify waiting threads.
         */
        void push(T item) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!shutdown_.load()) {
                queue_.push(std::move(item));
                condition_.notify_one();
            }
        }
        
        /**
         * Try to pop item immediately without blocking.
         */
        bool try_pop(T& item) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) {
                return false;
            }
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        
        /**
         * Block until item is available or queue is shut down.
         */
        bool wait_and_pop(T& item) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { 
                return !queue_.empty() || shutdown_.load(); 
            });
            
            if (queue_.empty()) {
                return false; // Queue was shut down
            }
            
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        
        /**
         * Wait with timeout for item availability.
         */
        template<typename Rep, typename Period>
        bool wait_for_pop(T& item, const std::chrono::duration<Rep, Period>& timeout) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (condition_.wait_for(lock, timeout, [this] { 
                return !queue_.empty() || shutdown_.load(); 
            })) {
                if (!queue_.empty()) {
                    item = std::move(queue_.front());
                    queue_.pop();
                    return true;
                }
            }
            return false;
        }
        
        /**
         * Check if queue is empty.
         */
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }
        
        /**
         * Get queue size.
         */
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }
        
        /**
         * Shutdown queue and wake all waiting threads.
         */
        void shutdown() {
            shutdown_.store(true);
            condition_.notify_all();
        }
    };
    
    /**
     * Thread pool for parallel task execution.
     */
    class ThreadPool {
    private:
        std::vector<std::thread> workers_;
        ThreadSafeQueue<std::function<void()>> tasks_;
        std::atomic<bool> stop_{false};
        
    public:
        /**
         * Create thread pool with specified number of workers.
         */
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
            for (size_t i = 0; i < num_threads; ++i) {
                workers_.emplace_back([this] {
                    std::function<void()> task;
                    while (!stop_.load()) {
                        if (tasks_.wait_and_pop(task)) {
                            try {
                                task();
                            } catch (const std::exception& e) {
                                std::cerr << "Task exception: " << e.what() << std::endl;
                            }
                        }
                    }
                });
            }
        }
        
        /**
         * Destructor ensures all threads are joined.
         */
        ~ThreadPool() {
            stop_.store(true);
            tasks_.shutdown();
            
            for (std::thread& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }
        
        // Non-copyable and non-movable
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        ThreadPool(ThreadPool&&) = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;
        
        /**
         * Submit task and get future for result.
         */
        template<typename F, typename... Args>
        auto enqueue(F&& f, Args&&... args) 
            -> std::future<typename std::invoke_result_t<F, Args...>> {
            
            using return_type = typename std::invoke_result_t<F, Args...>;
            
            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
            
            std::future<return_type> result = task->get_future();
            
            if (stop_.load()) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            
            tasks_.push([task] { (*task)(); });
            return result;
        }
        
        /**
         * Get number of worker threads.
         */
        size_t size() const noexcept { return workers_.size(); }
    };
    
    /**
     * Lock-free atomic counter with memory ordering.
     */
    class AtomicCounter {
    private:
        std::atomic<uint64_t> value_{0};
        
    public:
        /**
         * Increment and return new value.
         */
        uint64_t increment() noexcept {
            return value_.fetch_add(1, std::memory_order_acq_rel) + 1;
        }
        
        /**
         * Decrement and return new value.
         */
        uint64_t decrement() noexcept {
            return value_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        }
        
        /**
         * Get current value.
         */
        uint64_t get() const noexcept {
            return value_.load(std::memory_order_acquire);
        }
        
        /**
         * Set value atomically.
         */
        void set(uint64_t new_value) noexcept {
            value_.store(new_value, std::memory_order_release);
        }
        
        /**
         * Compare and swap operation.
         */
        bool compare_exchange(uint64_t& expected, uint64_t desired) noexcept {
            return value_.compare_exchange_strong(expected, desired, 
                                                std::memory_order_acq_rel);
        }
    };
}

// =============================================================================
// STL Algorithms and Functional Programming
// =============================================================================

namespace FunctionalProgramming {
    
    /**
     * Monadic optional wrapper with chaining operations.
     */
    template<typename T>
    class Optional {
    private:
        std::optional<T> value_;
        
    public:
        Optional() = default;
        Optional(const T& value) : value_(value) {}
        Optional(T&& value) : value_(std::move(value)) {}
        Optional(std::nullopt_t) : value_(std::nullopt) {}
        
        /**
         * Map operation for functional chaining.
         */
        template<typename F>
        auto map(F&& func) const -> Optional<decltype(func(*value_))> {
            if (value_) {
                return Optional<decltype(func(*value_))>(func(*value_));
            }
            return std::nullopt;
        }
        
        /**
         * Flat map for chaining optional-returning functions.
         */
        template<typename F>
        auto flat_map(F&& func) const -> decltype(func(*value_)) {
            if (value_) {
                return func(*value_);
            }
            return std::nullopt;
        }
        
        /**
         * Filter values based on predicate.
         */
        template<typename Predicate>
        Optional filter(Predicate&& pred) const {
            if (value_ && pred(*value_)) {
                return *this;
            }
            return std::nullopt;
        }
        
        /**
         * Get value or default.
         */
        T value_or(const T& default_value) const {
            return value_.value_or(default_value);
        }
        
        /**
         * Check if value exists.
         */
        bool has_value() const noexcept { return value_.has_value(); }
        
        /**
         * Access value (throws if empty).
         */
        const T& value() const { return value_.value(); }
        T& value() { return value_.value(); }
        
        explicit operator bool() const noexcept { return has_value(); }
    };
    
    /**
     * Functional pipeline for data transformation.
     */
    template<typename Container>
    class Pipeline {
    private:
        Container data_;
        
    public:
        explicit Pipeline(Container data) : data_(std::move(data)) {}
        
        /**
         * Filter elements based on predicate.
         */
        template<typename Predicate>
        auto filter(Predicate&& pred) && {
            Container result;
            std::copy_if(data_.begin(), data_.end(), 
                        std::back_inserter(result), 
                        std::forward<Predicate>(pred));
            return Pipeline(std::move(result));
        }
        
        /**
         * Transform elements using function.
         */
        template<typename Function>
        auto map(Function&& func) && {
            using ResultType = decltype(func(*data_.begin()));
            std::vector<ResultType> result;
            result.reserve(data_.size());
            
            std::transform(data_.begin(), data_.end(), 
                          std::back_inserter(result), 
                          std::forward<Function>(func));
            return Pipeline(std::move(result));
        }
        
        /**
         * Reduce to single value.
         */
        template<typename BinaryOp>
        auto reduce(BinaryOp&& op) const {
            if (data_.empty()) {
                throw std::runtime_error("Cannot reduce empty container");
            }
            return std::reduce(data_.begin() + 1, data_.end(), 
                             *data_.begin(), std::forward<BinaryOp>(op));
        }
        
        /**
         * Take first n elements.
         */
        auto take(size_t n) && {
            Container result;
            auto end_it = (n < data_.size()) ? data_.begin() + n : data_.end();
            std::copy(data_.begin(), end_it, std::back_inserter(result));
            return Pipeline(std::move(result));
        }
        
        /**
         * Skip first n elements.
         */
        auto skip(size_t n) && {
            Container result;
            auto start_it = (n < data_.size()) ? data_.begin() + n : data_.end();
            std::copy(start_it, data_.end(), std::back_inserter(result));
            return Pipeline(std::move(result));
        }
        
        /**
         * Sort elements.
         */
        template<typename Compare = std::less<>>
        auto sort(Compare&& comp = Compare{}) && {
            std::sort(data_.begin(), data_.end(), std::forward<Compare>(comp));
            return Pipeline(std::move(data_));
        }
        
        /**
         * Get the underlying container.
         */
        const Container& get() const& { return data_; }
        Container&& get() && { return std::move(data_); }
    };
    
    /**
     * Helper function to create pipeline.
     */
    template<typename Container>
    auto make_pipeline(Container&& container) {
        return Pipeline(std::forward<Container>(container));
    }
    
    /**
     * Curry function for partial application.
     */
    template<typename F>
    class Curried;
    
    template<typename R, typename... Args>
    class Curried<R(Args...)> {
    private:
        std::function<R(Args...)> func_;
        
    public:
        explicit Curried(std::function<R(Args...)> f) : func_(std::move(f)) {}
        
        template<typename... PartialArgs>
        auto operator()(PartialArgs&&... partial_args) const {
            if constexpr (sizeof...(PartialArgs) >= sizeof...(Args)) {
                return func_(std::forward<PartialArgs>(partial_args)...);
            } else {
                return [func = func_, partial_args...](auto&&... remaining_args) {
                    return func(partial_args..., 
                               std::forward<decltype(remaining_args)>(remaining_args)...);
                };
            }
        }
    };
    
    /**
     * Helper function to create curried function.
     */
    template<typename F>
    auto curry(F&& func) {
        return Curried<std::decay_t<F>>(std::forward<F>(func));
    }
}

// =============================================================================
// Performance Optimization and Benchmarking
// =============================================================================

namespace Performance {
    
    /**
     * High-resolution timer for benchmarking.
     */
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        
    public:
        Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}
        
        /**
         * Reset timer to current time.
         */
        void reset() {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
        
        /**
         * Get elapsed time in microseconds.
         */
        int64_t elapsed_microseconds() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_).count();
        }
        
        /**
         * Get elapsed time in milliseconds.
         */
        double elapsed_milliseconds() const {
            return elapsed_microseconds() / 1000.0;
        }
        
        /**
         * Get elapsed time in seconds.
         */
        double elapsed_seconds() const {
            return elapsed_microseconds() / 1000000.0;
        }
    };
    
    /**
     * Benchmark function execution time.
     */
    template<typename F, typename... Args>
    auto benchmark(const std::string& name, int iterations, F&& func, Args&&... args) {
        std::cout << "Benchmarking " << name << " (" << iterations << " iterations)..." << std::endl;
        
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            std::forward<F>(func)(std::forward<Args>(args)...);
        }
        
        double total_time = timer.elapsed_milliseconds();
        double avg_time = total_time / iterations;
        
        std::cout << "  Total time: " << total_time << " ms" << std::endl;
        std::cout << "  Average time: " << avg_time << " ms" << std::endl;
        std::cout << "  Operations/sec: " << (1000.0 / avg_time) << std::endl;
        
        return avg_time;
    }
    
    /**
     * Memory usage tracker using RAII.
     */
    class MemoryTracker {
    private:
        size_t initial_memory_;
        std::string name_;
        
        size_t get_memory_usage() const {
            // Simplified memory tracking - in real implementation,
            // would use platform-specific APIs
            return 0; // Placeholder
        }
        
    public:
        explicit MemoryTracker(const std::string& name) 
            : initial_memory_(get_memory_usage()), name_(name) {
            std::cout << "Memory tracking started for: " << name_ << std::endl;
        }
        
        ~MemoryTracker() {
            size_t final_memory = get_memory_usage();
            std::cout << "Memory tracking finished for: " << name_ << std::endl;
            std::cout << "  Memory change: " 
                      << (final_memory - initial_memory_) << " bytes" << std::endl;
        }
    };
    
    /**
     * Cache-friendly data structure for performance.
     */
    template<typename T, size_t CacheLineSize = 64>
    class alignas(CacheLineSize) CacheFriendlyVector {
    private:
        static constexpr size_t elements_per_line = CacheLineSize / sizeof(T);
        std::vector<T> data_;
        
    public:
        /**
         * Add element with cache-friendly allocation.
         */
        void push_back(const T& value) {
            data_.push_back(value);
        }
        
        /**
         * Access element with prefetch hint.
         */
        const T& operator[](size_t index) const {
            // Prefetch next cache line
            if (index + elements_per_line < data_.size()) {
                __builtin_prefetch(&data_[index + elements_per_line], 0, 3);
            }
            return data_[index];
        }
        
        /**
         * Bulk operation with vectorization hints.
         */
        template<typename BinaryOp>
        T reduce_parallel(BinaryOp&& op) const {
            if (data_.empty()) return T{};
            
            // Use available execution policy or fallback
            #ifdef __cpp_lib_execution
            return std::reduce(std::execution::par_unseq, 
                             data_.begin(), data_.end(), 
                             T{}, std::forward<BinaryOp>(op));
            #else
            return std::accumulate(data_.begin(), data_.end(), 
                                 T{}, std::forward<BinaryOp>(op));
            #endif
        }
        
        size_t size() const noexcept { return data_.size(); }
        bool empty() const noexcept { return data_.empty(); }
    };
}

// =============================================================================
// Example Usage and Demonstrations
// =============================================================================

/**
 * Comprehensive demonstration of all C++ features.
 */
void demonstrate_advanced_cpp() {
    std::cout << "🚀 Advanced C++ Features Demonstration" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Test modern C++ features
    std::cout << "\n📝 Modern C++20 Features:" << std::endl;
    ModernCpp::Matrix<double> matrix1(2, 2, std::vector<double>{1, 2, 3, 4});
    ModernCpp::Matrix<double> matrix2(2, 2, std::vector<double>{5, 6, 7, 8});
    auto result = matrix1 * matrix2;
    std::cout << "Matrix multiplication result: " << result(0, 0) << std::endl;
    
    constexpr auto fact5 = ModernCpp::factorial(5);
    std::cout << "Compile-time factorial(5): " << fact5 << std::endl;
    
    auto sum_result = ModernCpp::sum(1, 2, 3, 4, 5);
    std::cout << "Variadic sum: " << sum_result << std::endl;
    
    // Test memory management
    std::cout << "\n🧠 Memory Management:" << std::endl;
    MemoryManagement::PoolAllocator<int> allocator;
    auto stats = allocator.get_stats();
    std::cout << "Pool stats - Used: " << stats.first 
              << ", Free: " << stats.second << std::endl;
    
    auto smart_ptr = MemoryManagement::shared_ptr_ts<int>(new int(42));
    std::cout << "Smart pointer value: " << *smart_ptr 
              << ", ref count: " << smart_ptr.use_count() << std::endl;
    
    // Test concurrent programming
    std::cout << "\n⚡ Concurrent Programming:" << std::endl;
    ConcurrentProgramming::ThreadPool thread_pool(4);
    
    auto future_result = thread_pool.enqueue([](int x) { return x * x; }, 10);
    std::cout << "Thread pool task result: " << future_result.get() << std::endl;
    
    ConcurrentProgramming::AtomicCounter counter;
    counter.increment();
    counter.increment();
    std::cout << "Atomic counter value: " << counter.get() << std::endl;
    
    // Test functional programming
    std::cout << "\n🔧 Functional Programming:" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto pipeline_result = FunctionalProgramming::make_pipeline(numbers)
        .filter([](int x) { return x % 2 == 0; })
        .map([](int x) { return x * x; })
        .take(3)
        .get();
    
    std::cout << "Pipeline result: ";
    for (int x : pipeline_result) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    // Test optional chaining
    auto optional_result = FunctionalProgramming::Optional<int>(42)
        .map([](int x) { return x * 2; })
        .filter([](int x) { return x > 50; })
        .value_or(0);
    
    std::cout << "Optional chaining result: " << optional_result << std::endl;
    
    // Test performance features
    std::cout << "\n⚡ Performance Features:" << std::endl;
    Performance::Timer timer;
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::cout << "Elapsed time: " << timer.elapsed_milliseconds() << " ms" << std::endl;
    
    // Benchmark example
    Performance::benchmark("Vector sum", 1000, []() {
        std::vector<int> v(1000);
        std::iota(v.begin(), v.end(), 1);
        return std::accumulate(v.begin(), v.end(), 0);
    });
    
    std::cout << "\n✅ All C++ demonstrations completed!" << std::endl;
}

/**
 * Entry point for demonstrations.
 */
int main() {
    try {
        demonstrate_advanced_cpp();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}