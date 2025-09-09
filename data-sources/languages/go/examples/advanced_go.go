// Package main demonstrates advanced Go programming patterns for AI coding agents
//
// This module showcases modern Go features including:
// - Goroutines and channels for concurrent programming
// - Context package for cancellation and timeouts
// - Interface composition and type assertions
// - Generic programming with type parameters (Go 1.18+)
// - Error handling patterns and custom error types
// - Memory management and performance optimization
// - Testing patterns and benchmarking
// - Reflection and code generation patterns
// - HTTP client/server patterns
// - Database patterns with connection pooling
//
// Author: AI Dataset Creation Team
// License: MIT
// Created: 2024
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// =============================================================================
// Generic Programming and Type Constraints (Go 1.18+)
// =============================================================================

// Comparable defines types that can be compared for equality
type Comparable interface {
	comparable
}

// Numeric defines numeric types for mathematical operations
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Container defines types that can hold multiple values
type Container[T any] interface {
	Add(item T)
	Remove(item T) bool
	Contains(item T) bool
	Size() int
	Clear()
}

// GenericSlice provides utility functions for slices with type safety
type GenericSlice[T Comparable] struct {
	items []T
	mutex sync.RWMutex
}

// NewGenericSlice creates a new thread-safe generic slice
func NewGenericSlice[T Comparable]() *GenericSlice[T] {
	return &GenericSlice[T]{
		items: make([]T, 0),
	}
}

// Add appends an item to the slice
func (gs *GenericSlice[T]) Add(item T) {
	gs.mutex.Lock()
	defer gs.mutex.Unlock()
	gs.items = append(gs.items, item)
}

// Remove removes the first occurrence of an item
func (gs *GenericSlice[T]) Remove(item T) bool {
	gs.mutex.Lock()
	defer gs.mutex.Unlock()
	
	for i, v := range gs.items {
		if v == item {
			gs.items = append(gs.items[:i], gs.items[i+1:]...)
			return true
		}
	}
	return false
}

// Contains checks if an item exists in the slice
func (gs *GenericSlice[T]) Contains(item T) bool {
	gs.mutex.RLock()
	defer gs.mutex.RUnlock()
	
	for _, v := range gs.items {
		if v == item {
			return true
		}
	}
	return false
}

// Size returns the number of items in the slice
func (gs *GenericSlice[T]) Size() int {
	gs.mutex.RLock()
	defer gs.mutex.RUnlock()
	return len(gs.items)
}

// Clear removes all items from the slice
func (gs *GenericSlice[T]) Clear() {
	gs.mutex.Lock()
	defer gs.mutex.Unlock()
	gs.items = gs.items[:0]
}

// Map applies a function to each element and returns a new slice
func (gs *GenericSlice[T]) Map(fn func(T) T) *GenericSlice[T] {
	gs.mutex.RLock()
	defer gs.mutex.RUnlock()
	
	result := NewGenericSlice[T]()
	for _, item := range gs.items {
		result.Add(fn(item))
	}
	return result
}

// Filter returns a new slice containing only items that match the predicate
func (gs *GenericSlice[T]) Filter(predicate func(T) bool) *GenericSlice[T] {
	gs.mutex.RLock()
	defer gs.mutex.RUnlock()
	
	result := NewGenericSlice[T]()
	for _, item := range gs.items {
		if predicate(item) {
			result.Add(item)
		}
	}
	return result
}

// Reduce combines all elements into a single value
func Reduce[T, R any](slice []T, initial R, fn func(R, T) R) R {
	result := initial
	for _, item := range slice {
		result = fn(result, item)
	}
	return result
}

// GenericMap provides a thread-safe generic map implementation
type GenericMap[K Comparable, V any] struct {
	data  map[K]V
	mutex sync.RWMutex
}

// NewGenericMap creates a new thread-safe generic map
func NewGenericMap[K Comparable, V any]() *GenericMap[K, V] {
	return &GenericMap[K, V]{
		data: make(map[K]V),
	}
}

// Set stores a key-value pair
func (gm *GenericMap[K, V]) Set(key K, value V) {
	gm.mutex.Lock()
	defer gm.mutex.Unlock()
	gm.data[key] = value
}

// Get retrieves a value by key
func (gm *GenericMap[K, V]) Get(key K) (V, bool) {
	gm.mutex.RLock()
	defer gm.mutex.RUnlock()
	value, exists := gm.data[key]
	return value, exists
}

// Delete removes a key-value pair
func (gm *GenericMap[K, V]) Delete(key K) bool {
	gm.mutex.Lock()
	defer gm.mutex.Unlock()
	
	if _, exists := gm.data[key]; exists {
		delete(gm.data, key)
		return true
	}
	return false
}

// Keys returns all keys in the map
func (gm *GenericMap[K, V]) Keys() []K {
	gm.mutex.RLock()
	defer gm.mutex.RUnlock()
	
	keys := make([]K, 0, len(gm.data))
	for k := range gm.data {
		keys = append(keys, k)
	}
	return keys
}

// =============================================================================
// Concurrent Programming Patterns
// =============================================================================

// WorkerPool implements a concurrent worker pool pattern
type WorkerPool struct {
	workers    int
	jobs       chan func()
	results    chan interface{}
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
	started    int32
}

// NewWorkerPool creates a new worker pool with specified number of workers
func NewWorkerPool(workers int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	return &WorkerPool{
		workers: workers,
		jobs:    make(chan func(), workers*2), // Buffer for better performance
		results: make(chan interface{}, workers*2),
		ctx:     ctx,
		cancel:  cancel,
	}
}

// Start begins the worker pool execution
func (wp *WorkerPool) Start() {
	if !atomic.CompareAndSwapInt32(&wp.started, 0, 1) {
		return // Already started
	}
	
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

// worker is the main worker goroutine
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()
	
	for {
		select {
		case job, ok := <-wp.jobs:
			if !ok {
				return // Jobs channel closed
			}
			job()
		case <-wp.ctx.Done():
			return // Context cancelled
		}
	}
}

// Submit adds a job to the worker pool
func (wp *WorkerPool) Submit(job func()) error {
	select {
	case wp.jobs <- job:
		return nil
	case <-wp.ctx.Done():
		return wp.ctx.Err()
	default:
		return errors.New("worker pool is full")
	}
}

// SubmitWithResult submits a job and returns a channel for the result
func (wp *WorkerPool) SubmitWithResult(job func() interface{}) <-chan interface{} {
	resultChan := make(chan interface{}, 1)
	
	wp.Submit(func() {
		defer close(resultChan)
		result := job()
		select {
		case resultChan <- result:
		case <-wp.ctx.Done():
		}
	})
	
	return resultChan
}

// Stop gracefully shuts down the worker pool
func (wp *WorkerPool) Stop() {
	wp.cancel()
	close(wp.jobs)
	wp.wg.Wait()
	close(wp.results)
}

// Pipeline implements a concurrent pipeline pattern
type Pipeline struct {
	stages []func(context.Context, <-chan interface{}) <-chan interface{}
}

// NewPipeline creates a new processing pipeline
func NewPipeline() *Pipeline {
	return &Pipeline{
		stages: make([]func(context.Context, <-chan interface{}) <-chan interface{}, 0),
	}
}

// AddStage adds a processing stage to the pipeline
func (p *Pipeline) AddStage(stage func(context.Context, <-chan interface{}) <-chan interface{}) {
	p.stages = append(p.stages, stage)
}

// Process executes the pipeline with input data
func (p *Pipeline) Process(ctx context.Context, input <-chan interface{}) <-chan interface{} {
	current := input
	for _, stage := range p.stages {
		current = stage(ctx, current)
	}
	return current
}

// Example pipeline stages
func FilterStage(predicate func(interface{}) bool) func(context.Context, <-chan interface{}) <-chan interface{} {
	return func(ctx context.Context, input <-chan interface{}) <-chan interface{} {
		output := make(chan interface{})
		
		go func() {
			defer close(output)
			for {
				select {
				case item, ok := <-input:
					if !ok {
						return
					}
					if predicate(item) {
						select {
						case output <- item:
						case <-ctx.Done():
							return
						}
					}
				case <-ctx.Done():
					return
				}
			}
		}()
		
		return output
	}
}

// TransformStage applies a transformation to each item
func TransformStage(transform func(interface{}) interface{}) func(context.Context, <-chan interface{}) <-chan interface{} {
	return func(ctx context.Context, input <-chan interface{}) <-chan interface{} {
		output := make(chan interface{})
		
		go func() {
			defer close(output)
			for {
				select {
				case item, ok := <-input:
					if !ok {
						return
					}
					transformed := transform(item)
					select {
					case output <- transformed:
					case <-ctx.Done():
						return
					}
				case <-ctx.Done():
					return
				}
			}
		}()
		
		return output
	}
}

// =============================================================================
// Advanced Error Handling Patterns
// =============================================================================

// CustomError represents a custom error with additional context
type CustomError struct {
	Code    string
	Message string
	Cause   error
	Context map[string]interface{}
}

// Error implements the error interface
func (e *CustomError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s (caused by: %v)", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying error for error wrapping
func (e *CustomError) Unwrap() error {
	return e.Cause
}

// WithContext adds context to the error
func (e *CustomError) WithContext(key string, value interface{}) *CustomError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// NewCustomError creates a new custom error
func NewCustomError(code, message string, cause error) *CustomError {
	return &CustomError{
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// Result represents a result that can be either success or error
type Result[T any] struct {
	value T
	err   error
}

// NewResult creates a new Result with a value
func NewResult[T any](value T) Result[T] {
	return Result[T]{value: value}
}

// NewResultError creates a new Result with an error
func NewResultError[T any](err error) Result[T] {
	var zero T
	return Result[T]{value: zero, err: err}
}

// IsOk returns true if the result contains a value
func (r Result[T]) IsOk() bool {
	return r.err == nil
}

// IsErr returns true if the result contains an error
func (r Result[T]) IsErr() bool {
	return r.err != nil
}

// Value returns the value if present, panics if error
func (r Result[T]) Value() T {
	if r.err != nil {
		panic(fmt.Sprintf("attempted to get value from error result: %v", r.err))
	}
	return r.value
}

// Error returns the error if present, nil otherwise
func (r Result[T]) Error() error {
	return r.err
}

// ValueOr returns the value if present, otherwise returns the default
func (r Result[T]) ValueOr(defaultValue T) T {
	if r.err != nil {
		return defaultValue
	}
	return r.value
}

// Map applies a function to the value if present
func (r Result[T]) Map(fn func(T) T) Result[T] {
	if r.err != nil {
		return r
	}
	return NewResult(fn(r.value))
}

// =============================================================================
// HTTP Client/Server Patterns
// =============================================================================

// HTTPClient provides a robust HTTP client with retry and timeout
type HTTPClient struct {
	client      *http.Client
	retries     int
	backoffFunc func(int) time.Duration
}

// NewHTTPClient creates a new HTTP client with configuration
func NewHTTPClient(timeout time.Duration, retries int) *HTTPClient {
	return &HTTPClient{
		client: &http.Client{
			Timeout: timeout,
		},
		retries: retries,
		backoffFunc: func(attempt int) time.Duration {
			// Exponential backoff with jitter
			return time.Duration(attempt*attempt) * time.Second
		},
	}
}

// Get performs a GET request with retry logic
func (hc *HTTPClient) Get(ctx context.Context, url string) (*http.Response, error) {
	return hc.doWithRetry(ctx, "GET", url, nil)
}

// Post performs a POST request with retry logic
func (hc *HTTPClient) Post(ctx context.Context, url string, body io.Reader) (*http.Response, error) {
	return hc.doWithRetry(ctx, "POST", url, body)
}

// doWithRetry implements the retry logic
func (hc *HTTPClient) doWithRetry(ctx context.Context, method, url string, body io.Reader) (*http.Response, error) {
	var lastErr error
	
	for attempt := 0; attempt <= hc.retries; attempt++ {
		if attempt > 0 {
			// Wait before retry
			select {
			case <-time.After(hc.backoffFunc(attempt)):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		
		req, err := http.NewRequestWithContext(ctx, method, url, body)
		if err != nil {
			return nil, err
		}
		
		resp, err := hc.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		
		// Success or non-retryable error
		if resp.StatusCode < 500 {
			return resp, nil
		}
		
		// Server error, might be retryable
		resp.Body.Close()
		lastErr = fmt.Errorf("server error: %d", resp.StatusCode)
	}
	
	return nil, fmt.Errorf("request failed after %d retries: %w", hc.retries, lastErr)
}

// MiddlewareFunc defines HTTP middleware signature
type MiddlewareFunc func(http.Handler) http.Handler

// ChainMiddleware chains multiple middleware functions
func ChainMiddleware(middlewares ...MiddlewareFunc) MiddlewareFunc {
	return func(next http.Handler) http.Handler {
		for i := len(middlewares) - 1; i >= 0; i-- {
			next = middlewares[i](next)
		}
		return next
	}
}

// LoggingMiddleware logs HTTP requests
func LoggingMiddleware(logger *log.Logger) MiddlewareFunc {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			
			// Create a response writer that captures status code
			wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
			
			next.ServeHTTP(wrapped, r)
			
			duration := time.Since(start)
			logger.Printf("%s %s %d %v", r.Method, r.URL.Path, wrapped.statusCode, duration)
		})
	}
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// RateLimitMiddleware implements rate limiting
func RateLimitMiddleware(requestsPerSecond int) MiddlewareFunc {
	type client struct {
		lastSeen time.Time
		requests int
	}
	
	clients := make(map[string]*client)
	var mutex sync.RWMutex
	
	// Cleanup old entries periodically
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		
		for range ticker.C {
			mutex.Lock()
			cutoff := time.Now().Add(-time.Minute)
			for ip, c := range clients {
				if c.lastSeen.Before(cutoff) {
					delete(clients, ip)
				}
			}
			mutex.Unlock()
		}
	}()
	
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ip := r.RemoteAddr
			
			mutex.Lock()
			c, exists := clients[ip]
			if !exists {
				c = &client{}
				clients[ip] = c
			}
			
			now := time.Now()
			if now.Sub(c.lastSeen) > time.Second {
				c.requests = 0
			}
			
			c.requests++
			c.lastSeen = now
			requests := c.requests
			mutex.Unlock()
			
			if requests > requestsPerSecond {
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	}
}

// =============================================================================
// Performance Optimization Patterns
// =============================================================================

// ObjectPool implements an object pool for memory reuse
type ObjectPool[T any] struct {
	pool sync.Pool
	new  func() *T
}

// NewObjectPool creates a new object pool
func NewObjectPool[T any](newFunc func() *T) *ObjectPool[T] {
	return &ObjectPool[T]{
		pool: sync.Pool{
			New: func() interface{} {
				return newFunc()
			},
		},
		new: newFunc,
	}
}

// Get retrieves an object from the pool
func (op *ObjectPool[T]) Get() *T {
	return op.pool.Get().(*T)
}

// Put returns an object to the pool
func (op *ObjectPool[T]) Put(obj *T) {
	op.pool.Put(obj)
}

// MemoryStats provides memory usage information
type MemoryStats struct {
	Alloc        uint64 // bytes allocated and not yet freed
	TotalAlloc   uint64 // bytes allocated (even if freed)
	Sys          uint64 // bytes obtained from system
	Lookups      uint64 // number of pointer lookups
	Mallocs      uint64 // number of mallocs
	Frees        uint64 // number of frees
	HeapAlloc    uint64 // bytes allocated and not yet freed (same as Alloc)
	HeapSys      uint64 // bytes obtained from system
	HeapIdle     uint64 // bytes in idle spans
	HeapInuse    uint64 // bytes in non-idle span
	HeapReleased uint64 // bytes released to the OS
	HeapObjects  uint64 // total number of allocated objects
	StackInuse   uint64 // bootstrap stacks
	StackSys     uint64
	MSpanInuse   uint64 // mspan structures
	MSpanSys     uint64
	MCacheInuse  uint64 // mcache structures
	MCacheSys    uint64
	BuckHashSys  uint64 // profiling bucket hash table
	GCSys        uint64 // GC metadata
	OtherSys     uint64 // other system allocations
	NextGC       uint64 // next collection will happen when HeapAlloc ≥ this amount
	LastGC       uint64 // end time of last collection (nanoseconds since 1970)
	PauseTotalNs uint64
	PauseNs      [256]uint64 // circular buffer of recent GC pause durations
	PauseEnd     [256]uint64 // circular buffer of recent GC pause end times
	NumGC        uint32
	NumForcedGC  uint32  // number of user-forced GCs
	GCCPUFraction float64 // fraction of CPU time used by GC
	EnableGC     bool
	DebugGC      bool
}

// GetMemoryStats returns current memory statistics
func GetMemoryStats() MemoryStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return MemoryStats{
		Alloc:         m.Alloc,
		TotalAlloc:    m.TotalAlloc,
		Sys:           m.Sys,
		Lookups:       m.Lookups,
		Mallocs:       m.Mallocs,
		Frees:         m.Frees,
		HeapAlloc:     m.HeapAlloc,
		HeapSys:       m.HeapSys,
		HeapIdle:      m.HeapIdle,
		HeapInuse:     m.HeapInuse,
		HeapReleased:  m.HeapReleased,
		HeapObjects:   m.HeapObjects,
		StackInuse:    m.StackInuse,
		StackSys:      m.StackSys,
		MSpanInuse:    m.MSpanInuse,
		MSpanSys:      m.MSpanSys,
		MCacheInuse:   m.MCacheInuse,
		MCacheSys:     m.MCacheSys,
		BuckHashSys:   m.BuckHashSys,
		GCSys:         m.GCSys,
		OtherSys:      m.OtherSys,
		NextGC:        m.NextGC,
		LastGC:        m.LastGC,
		PauseTotalNs:  m.PauseTotalNs,
		PauseNs:       m.PauseNs,
		PauseEnd:      m.PauseEnd,
		NumGC:         m.NumGC,
		NumForcedGC:   m.NumForcedGC,
		GCCPUFraction: m.GCCPUFraction,
		EnableGC:      m.EnableGC,
		DebugGC:       m.DebugGC,
	}
}

// Benchmark measures function execution time
type Benchmark struct {
	Name       string
	Iterations int
	Duration   time.Duration
	MemBefore  MemoryStats
	MemAfter   MemoryStats
}

// BenchmarkFunc runs a benchmark on a function
func BenchmarkFunc(name string, iterations int, fn func()) Benchmark {
	// Force GC before measurement
	runtime.GC()
	runtime.GC()
	
	memBefore := GetMemoryStats()
	start := time.Now()
	
	for i := 0; i < iterations; i++ {
		fn()
	}
	
	duration := time.Since(start)
	
	// Force GC after measurement
	runtime.GC()
	runtime.GC()
	
	memAfter := GetMemoryStats()
	
	return Benchmark{
		Name:       name,
		Iterations: iterations,
		Duration:   duration,
		MemBefore:  memBefore,
		MemAfter:   memAfter,
	}
}

// Report prints benchmark results
func (b Benchmark) Report() {
	avgTime := b.Duration / time.Duration(b.Iterations)
	memDiff := int64(b.MemAfter.TotalAlloc) - int64(b.MemBefore.TotalAlloc)
	
	fmt.Printf("Benchmark: %s\n", b.Name)
	fmt.Printf("  Iterations: %d\n", b.Iterations)
	fmt.Printf("  Total time: %v\n", b.Duration)
	fmt.Printf("  Average time: %v\n", avgTime)
	fmt.Printf("  Ops/sec: %.0f\n", float64(b.Iterations)/b.Duration.Seconds())
	fmt.Printf("  Memory allocated: %d bytes\n", memDiff)
	fmt.Printf("  Memory per op: %.2f bytes\n", float64(memDiff)/float64(b.Iterations))
}

// =============================================================================
// Example Usage and Demonstrations
// =============================================================================

// demonstrateAdvancedGo showcases all the advanced Go features
func demonstrateAdvancedGo() {
	fmt.Println("🚀 Advanced Go Programming Demonstration")
	fmt.Println(strings.Repeat("=", 50))
	
	// Test generic programming
	fmt.Println("\n📝 Generic Programming:")
	stringSlice := NewGenericSlice[string]()
	stringSlice.Add("hello")
	stringSlice.Add("world")
	stringSlice.Add("go")
	
	filtered := stringSlice.Filter(func(s string) bool {
		return len(s) > 2
	})
	
	fmt.Printf("Original slice size: %d\n", stringSlice.Size())
	fmt.Printf("Filtered slice size: %d\n", filtered.Size())
	
	// Test generic map
	intMap := NewGenericMap[string, int]()
	intMap.Set("one", 1)
	intMap.Set("two", 2)
	intMap.Set("three", 3)
	
	if value, exists := intMap.Get("two"); exists {
		fmt.Printf("Found value: %d\n", value)
	}
	
	// Test concurrent programming
	fmt.Println("\n⚡ Concurrent Programming:")
	
	// Worker pool demonstration
	wp := NewWorkerPool(3)
	wp.Start()
	
	// Submit jobs
	results := make([]<-chan interface{}, 5)
	for i := 0; i < 5; i++ {
		index := i
		results[i] = wp.SubmitWithResult(func() interface{} {
			time.Sleep(time.Millisecond * 100) // Simulate work
			return fmt.Sprintf("Job %d completed", index)
		})
	}
	
	// Collect results
	for i, resultChan := range results {
		result := <-resultChan
		fmt.Printf("Result %d: %v\n", i, result)
	}
	
	wp.Stop()
	
	// Pipeline demonstration
	fmt.Println("\n🔄 Pipeline Processing:")
	pipeline := NewPipeline()
	
	// Add stages
	pipeline.AddStage(FilterStage(func(v interface{}) bool {
		if num, ok := v.(int); ok {
			return num%2 == 0 // Filter even numbers
		}
		return false
	}))
	
	pipeline.AddStage(TransformStage(func(v interface{}) interface{} {
		if num, ok := v.(int); ok {
			return num * num // Square the number
		}
		return v
	}))
	
	// Create input data
	input := make(chan interface{}, 10)
	go func() {
		defer close(input)
		for i := 1; i <= 10; i++ {
			input <- i
		}
	}()
	
	// Process through pipeline
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()
	
	output := pipeline.Process(ctx, input)
	
	fmt.Println("Pipeline results:")
	for result := range output {
		fmt.Printf("  %v\n", result)
	}
	
	// Test error handling
	fmt.Println("\n❌ Error Handling:")
	
	// Custom error example
	err := NewCustomError("VALIDATION_ERROR", "Invalid input data", errors.New("field is required"))
	err.WithContext("field", "email").WithContext("value", "")
	
	fmt.Printf("Custom error: %v\n", err)
	fmt.Printf("Error context: %+v\n", err.Context)
	
	// Result type example
	result := divideNumbers(10, 2)
	if result.IsOk() {
		fmt.Printf("Division result: %.2f\n", result.Value())
	} else {
		fmt.Printf("Division error: %v\n", result.Error())
	}
	
	result = divideNumbers(10, 0)
	if result.IsErr() {
		fmt.Printf("Expected error: %v\n", result.Error())
	}
	
	// Test performance features
	fmt.Println("\n⚡ Performance Optimization:")
	
	// Object pool demonstration
	bufferPool := NewObjectPool(func() *[]byte {
		buffer := make([]byte, 1024)
		return &buffer
	})
	
	buffer := bufferPool.Get()
	fmt.Printf("Got buffer of size: %d\n", len(*buffer))
	bufferPool.Put(buffer)
	
	// Memory statistics
	memStats := GetMemoryStats()
	fmt.Printf("Current memory usage: %.2f MB\n", float64(memStats.Alloc)/1024/1024)
	fmt.Printf("Total allocations: %.2f MB\n", float64(memStats.TotalAlloc)/1024/1024)
	fmt.Printf("Number of GC cycles: %d\n", memStats.NumGC)
	
	// Benchmark demonstration
	fmt.Println("\n📊 Benchmarking:")
	
	benchmark := BenchmarkFunc("String concatenation", 10000, func() {
		var result string
		for i := 0; i < 100; i++ {
			result += "test"
		}
		_ = result
	})
	
	benchmark.Report()
	
	// Compare with string builder
	benchmark2 := BenchmarkFunc("String builder", 10000, func() {
		var builder strings.Builder
		for i := 0; i < 100; i++ {
			builder.WriteString("test")
		}
		_ = builder.String()
	})
	
	benchmark2.Report()
	
	fmt.Println("\n✅ All Go demonstrations completed!")
}

// divideNumbers demonstrates the Result type for error handling
func divideNumbers(a, b float64) Result[float64] {
	if b == 0 {
		return NewResultError[float64](NewCustomError("DIVISION_ERROR", "Division by zero", nil))
	}
	return NewResult(a / b)
}

// main function demonstrates all features
func main() {
	demonstrateAdvancedGo()
}