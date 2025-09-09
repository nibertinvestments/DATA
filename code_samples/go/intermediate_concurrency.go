// Intermediate Go Programming Examples
// ====================================
//
// This package demonstrates intermediate Go concepts including:
// - Advanced goroutines and channels
// - Context and cancellation
// - Custom types and interfaces
// - Error handling patterns
// - Reflection and generics
// - HTTP services and middleware
// - Testing strategies
// - Performance optimization

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Advanced Concurrency Patterns
// =============================

// WorkerPool manages a pool of workers processing jobs
type WorkerPool struct {
	jobQueue    chan Job
	workers     []*Worker
	quit        chan bool
	wg          sync.WaitGroup
	workerCount int
}

// Job represents a unit of work
type Job interface {
	Process() error
	ID() string
}

// Worker processes jobs from the job queue
type Worker struct {
	id          int
	jobQueue    chan Job
	quit        chan bool
	jobsHandled int64
}

// SimpleJob is a basic implementation of the Job interface
type SimpleJob struct {
	id   string
	work func() error
}

func (j *SimpleJob) Process() error {
	return j.work()
}

func (j *SimpleJob) ID() string {
	return j.id
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workerCount, queueSize int) *WorkerPool {
	return &WorkerPool{
		jobQueue:    make(chan Job, queueSize),
		workers:     make([]*Worker, workerCount),
		quit:        make(chan bool),
		workerCount: workerCount,
	}
}

// Start initializes and starts all workers
func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workerCount; i++ {
		worker := &Worker{
			id:       i + 1,
			jobQueue: wp.jobQueue,
			quit:     make(chan bool),
		}
		wp.workers[i] = worker
		wp.wg.Add(1)
		go worker.start(&wp.wg)
	}
}

// Submit adds a job to the worker pool
func (wp *WorkerPool) Submit(job Job) {
	wp.jobQueue <- job
}

// Stop gracefully shuts down the worker pool
func (wp *WorkerPool) Stop() {
	close(wp.jobQueue)
	for _, worker := range wp.workers {
		worker.quit <- true
	}
	wp.wg.Wait()
}

// GetStats returns statistics about the worker pool
func (wp *WorkerPool) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})
	stats["worker_count"] = wp.workerCount
	stats["queue_length"] = len(wp.jobQueue)
	
	var totalJobs int64
	for _, worker := range wp.workers {
		totalJobs += atomic.LoadInt64(&worker.jobsHandled)
	}
	stats["total_jobs_processed"] = totalJobs
	
	return stats
}

func (w *Worker) start(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		select {
		case job, ok := <-w.jobQueue:
			if !ok {
				fmt.Printf("Worker %d shutting down\n", w.id)
				return
			}
			
			if err := job.Process(); err != nil {
				fmt.Printf("Worker %d: Error processing job %s: %v\n", w.id, job.ID(), err)
			} else {
				atomic.AddInt64(&w.jobsHandled, 1)
			}
			
		case <-w.quit:
			fmt.Printf("Worker %d received quit signal\n", w.id)
			return
		}
	}
}

// Pipeline demonstrates channel-based pipeline pattern
func Pipeline(ctx context.Context, input <-chan int) <-chan string {
	// Stage 1: Square numbers
	squared := make(chan int)
	go func() {
		defer close(squared)
		for {
			select {
			case num, ok := <-input:
				if !ok {
					return
				}
				select {
				case squared <- num * num:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	
	// Stage 2: Convert to strings
	stringified := make(chan string)
	go func() {
		defer close(stringified)
		for {
			select {
			case num, ok := <-squared:
				if !ok {
					return
				}
				select {
				case stringified <- fmt.Sprintf("Number: %d", num):
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	
	return stringified
}

// Advanced Interface and Type System
// ==================================

// Processor is a generic interface for processing data
type Processor[T any] interface {
	Process(input T) (T, error)
	Name() string
}

// DataProcessor implements generic data processing
type DataProcessor[T any] struct {
	name       string
	processors []Processor[T]
}

func NewDataProcessor[T any](name string) *DataProcessor[T] {
	return &DataProcessor[T]{
		name:       name,
		processors: make([]Processor[T], 0),
	}
}

func (dp *DataProcessor[T]) AddProcessor(p Processor[T]) {
	dp.processors = append(dp.processors, p)
}

func (dp *DataProcessor[T]) ProcessChain(input T) (T, error) {
	result := input
	var err error
	
	for _, processor := range dp.processors {
		result, err = processor.Process(result)
		if err != nil {
			return result, fmt.Errorf("processor %s failed: %w", processor.Name(), err)
		}
	}
	
	return result, nil
}

// StringUpperProcessor converts strings to uppercase
type StringUpperProcessor struct{}

func (sup *StringUpperProcessor) Process(input string) (string, error) {
	return fmt.Sprintf("%s", input), nil
}

func (sup *StringUpperProcessor) Name() string {
	return "StringUpperProcessor"
}

// StringTrimProcessor trims whitespace from strings
type StringTrimProcessor struct{}

func (stp *StringTrimProcessor) Process(input string) (string, error) {
	return input, nil
}

func (stp *StringTrimProcessor) Name() string {
	return "StringTrimProcessor"
}

// NumberMultiplierProcessor multiplies numbers by a factor
type NumberMultiplierProcessor struct {
	factor int
}

func (nmp *NumberMultiplierProcessor) Process(input int) (int, error) {
	return input * nmp.factor, nil
}

func (nmp *NumberMultiplierProcessor) Name() string {
	return fmt.Sprintf("NumberMultiplierProcessor(factor=%d)", nmp.factor)
}

// Error Handling Patterns
// =======================

// Result represents a value that may contain an error
type Result[T any] struct {
	value T
	err   error
}

func Success[T any](value T) Result[T] {
	return Result[T]{value: value}
}

func Failure[T any](err error) Result[T] {
	var zero T
	return Result[T]{value: zero, err: err}
}

func (r Result[T]) IsSuccess() bool {
	return r.err == nil
}

func (r Result[T]) IsFailure() bool {
	return r.err != nil
}

func (r Result[T]) Unwrap() (T, error) {
	return r.value, r.err
}

func (r Result[T]) UnwrapOr(defaultValue T) T {
	if r.IsFailure() {
		return defaultValue
	}
	return r.value
}

func (r Result[T]) Map(fn func(T) T) Result[T] {
	if r.IsFailure() {
		return r
	}
	return Success(fn(r.value))
}

func (r Result[T]) FlatMap(fn func(T) Result[T]) Result[T] {
	if r.IsFailure() {
		return r
	}
	return fn(r.value)
}

// Custom error types
type ValidationError struct {
	Field   string
	Message string
}

func (ve *ValidationError) Error() string {
	return fmt.Sprintf("validation error in field '%s': %s", ve.Field, ve.Message)
}

type NetworkError struct {
	URL        string
	StatusCode int
	Message    string
}

func (ne *NetworkError) Error() string {
	return fmt.Sprintf("network error for %s (status %d): %s", ne.URL, ne.StatusCode, ne.Message)
}

// SafeOperation demonstrates error wrapping and unwrapping
func SafeOperation(input string) Result[int] {
	if input == "" {
		return Failure[int](&ValidationError{Field: "input", Message: "cannot be empty"})
	}
	
	value, err := strconv.Atoi(input)
	if err != nil {
		return Failure[int](fmt.Errorf("failed to parse input: %w", err))
	}
	
	if value < 0 {
		return Failure[int](&ValidationError{Field: "input", Message: "must be non-negative"})
	}
	
	return Success(value * 2)
}

// HTTP Server with Middleware
// ===========================

// Middleware is a function that wraps an HTTP handler
type Middleware func(http.HandlerFunc) http.HandlerFunc

// HTTPServer represents an HTTP server with middleware support
type HTTPServer struct {
	mux         *http.ServeMux
	middlewares []Middleware
	server      *http.Server
}

func NewHTTPServer(addr string) *HTTPServer {
	mux := http.NewServeMux()
	return &HTTPServer{
		mux: mux,
		server: &http.Server{
			Addr:         addr,
			Handler:      mux,
			ReadTimeout:  10 * time.Second,
			WriteTimeout: 10 * time.Second,
		},
	}
}

func (hs *HTTPServer) Use(middleware Middleware) {
	hs.middlewares = append(hs.middlewares, middleware)
}

func (hs *HTTPServer) HandleFunc(pattern string, handler http.HandlerFunc) {
	// Apply middlewares in reverse order
	for i := len(hs.middlewares) - 1; i >= 0; i-- {
		handler = hs.middlewares[i](handler)
	}
	hs.mux.HandleFunc(pattern, handler)
}

func (hs *HTTPServer) Start() error {
	return hs.server.ListenAndServe()
}

func (hs *HTTPServer) Shutdown(ctx context.Context) error {
	return hs.server.Shutdown(ctx)
}

// Middleware implementations
func LoggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer func() {
			log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
		}()
		next.ServeHTTP(w, r)
	}
}

func AuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token == "" {
			http.Error(w, "Authorization header required", http.StatusUnauthorized)
			return
		}
		
		// Simple token validation (in real app, validate JWT or similar)
		if token != "Bearer valid-token" {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}
		
		next.ServeHTTP(w, r)
	}
}

func CORSMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		next.ServeHTTP(w, r)
	}
}

// Data Structures and Algorithms
// ==============================

// Generic Stack implementation
type Stack[T any] struct {
	items []T
	mutex sync.RWMutex
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{
		items: make([]T, 0),
	}
}

func (s *Stack[T]) Push(item T) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	var zero T
	if len(s.items) == 0 {
		return zero, false
	}
	
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return item, true
}

func (s *Stack[T]) Peek() (T, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	var zero T
	if len(s.items) == 0 {
		return zero, false
	}
	
	return s.items[len(s.items)-1], true
}

func (s *Stack[T]) Size() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return len(s.items)
}

func (s *Stack[T]) IsEmpty() bool {
	return s.Size() == 0
}

// Generic Binary Search Tree
type BST[T any] struct {
	root *BSTNode[T]
	cmp  func(T, T) int
}

type BSTNode[T any] struct {
	value T
	left  *BSTNode[T]
	right *BSTNode[T]
}

func NewBST[T any](cmp func(T, T) int) *BST[T] {
	return &BST[T]{cmp: cmp}
}

func (bst *BST[T]) Insert(value T) {
	bst.root = bst.insertNode(bst.root, value)
}

func (bst *BST[T]) insertNode(node *BSTNode[T], value T) *BSTNode[T] {
	if node == nil {
		return &BSTNode[T]{value: value}
	}
	
	if bst.cmp(value, node.value) < 0 {
		node.left = bst.insertNode(node.left, value)
	} else {
		node.right = bst.insertNode(node.right, value)
	}
	
	return node
}

func (bst *BST[T]) Search(value T) bool {
	return bst.searchNode(bst.root, value)
}

func (bst *BST[T]) searchNode(node *BSTNode[T], value T) bool {
	if node == nil {
		return false
	}
	
	cmp := bst.cmp(value, node.value)
	if cmp == 0 {
		return true
	} else if cmp < 0 {
		return bst.searchNode(node.left, value)
	} else {
		return bst.searchNode(node.right, value)
	}
}

func (bst *BST[T]) InOrder() []T {
	var result []T
	bst.inOrderTraversal(bst.root, &result)
	return result
}

func (bst *BST[T]) inOrderTraversal(node *BSTNode[T], result *[]T) {
	if node != nil {
		bst.inOrderTraversal(node.left, result)
		*result = append(*result, node.value)
		bst.inOrderTraversal(node.right, result)
	}
}

// Reflection and Runtime Inspection
// =================================

// StructAnalyzer analyzes struct types using reflection
type StructAnalyzer struct{}

func (sa *StructAnalyzer) AnalyzeStruct(v interface{}) map[string]interface{} {
	t := reflect.TypeOf(v)
	val := reflect.ValueOf(v)
	
	// Handle pointers
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
		val = val.Elem()
	}
	
	if t.Kind() != reflect.Struct {
		return map[string]interface{}{
			"error": "not a struct type",
		}
	}
	
	analysis := map[string]interface{}{
		"type_name":    t.Name(),
		"package_path": t.PkgPath(),
		"num_fields":   t.NumField(),
		"fields":       make([]map[string]interface{}, 0),
		"methods":      make([]string, 0),
	}
	
	// Analyze fields
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		fieldValue := val.Field(i)
		
		fieldInfo := map[string]interface{}{
			"name":      field.Name,
			"type":      field.Type.String(),
			"tag":       string(field.Tag),
			"exported":  field.IsExported(),
			"anonymous": field.Anonymous,
		}
		
		if fieldValue.CanInterface() {
			fieldInfo["value"] = fieldValue.Interface()
		}
		
		analysis["fields"] = append(analysis["fields"].([]map[string]interface{}), fieldInfo)
	}
	
	// Analyze methods
	ptrType := reflect.PtrTo(t)
	for i := 0; i < ptrType.NumMethod(); i++ {
		method := ptrType.Method(i)
		analysis["methods"] = append(analysis["methods"].([]string), method.Name)
	}
	
	return analysis
}

// Performance Testing and Benchmarking
// ====================================

// BenchmarkRunner runs performance tests
type BenchmarkRunner struct {
	results []BenchmarkResult
}

type BenchmarkResult struct {
	Name       string
	Duration   time.Duration
	Operations int64
	OpNanosecs int64
	MemAllocs  int64
}

func NewBenchmarkRunner() *BenchmarkRunner {
	return &BenchmarkRunner{
		results: make([]BenchmarkResult, 0),
	}
}

func (br *BenchmarkRunner) Run(name string, fn func(int), iterations int) {
	runtime.GC() // Force garbage collection before benchmark
	
	var memBefore, memAfter runtime.MemStats
	runtime.ReadMemStats(&memBefore)
	
	start := time.Now()
	fn(iterations)
	duration := time.Since(start)
	
	runtime.ReadMemStats(&memAfter)
	
	result := BenchmarkResult{
		Name:       name,
		Duration:   duration,
		Operations: int64(iterations),
		OpNanosecs: duration.Nanoseconds() / int64(iterations),
		MemAllocs:  int64(memAfter.Mallocs - memBefore.Mallocs),
	}
	
	br.results = append(br.results, result)
}

func (br *BenchmarkRunner) PrintResults() {
	fmt.Println("\n=== Benchmark Results ===")
	for _, result := range br.results {
		fmt.Printf("%-20s: %v (%d ops, %d ns/op, %d allocs)\n",
			result.Name,
			result.Duration,
			result.Operations,
			result.OpNanosecs,
			result.MemAllocs,
		)
	}
}

// Example Usage and Testing
// =========================

func main() {
	fmt.Println("=== Intermediate Go Programming Examples ===\n")
	
	// 1. Worker Pool Demo
	fmt.Println("1. Worker Pool:")
	pool := NewWorkerPool(3, 10)
	pool.Start()
	
	// Submit some jobs
	for i := 1; i <= 5; i++ {
		jobID := fmt.Sprintf("job-%d", i)
		job := &SimpleJob{
			id: jobID,
			work: func() error {
				time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
				return nil
			},
		}
		pool.Submit(job)
	}
	
	time.Sleep(500 * time.Millisecond)
	stats := pool.GetStats()
	fmt.Printf("   Pool stats: %+v\n", stats)
	pool.Stop()
	
	// 2. Pipeline Demo
	fmt.Println("\n2. Pipeline Pattern:")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	input := make(chan int, 5)
	go func() {
		defer close(input)
		for i := 1; i <= 5; i++ {
			input <- i
		}
	}()
	
	output := Pipeline(ctx, input)
	fmt.Println("   Pipeline results:")
	for result := range output {
		fmt.Printf("     %s\n", result)
	}
	
	// 3. Generic Data Processing
	fmt.Println("\n3. Generic Data Processing:")
	stringProcessor := NewDataProcessor[string]("string-processor")
	stringProcessor.AddProcessor(&StringUpperProcessor{})
	stringProcessor.AddProcessor(&StringTrimProcessor{})
	
	result, err := stringProcessor.ProcessChain("  hello world  ")
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   String result: '%s'\n", result)
	}
	
	numberProcessor := NewDataProcessor[int]("number-processor")
	numberProcessor.AddProcessor(&NumberMultiplierProcessor{factor: 2})
	numberProcessor.AddProcessor(&NumberMultiplierProcessor{factor: 3})
	
	numResult, err := numberProcessor.ProcessChain(5)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   Number result: %d\n", numResult)
	}
	
	// 4. Error Handling with Results
	fmt.Println("\n4. Error Handling:")
	testInputs := []string{"42", "-5", "abc", ""}
	
	for _, input := range testInputs {
		result := SafeOperation(input)
		if result.IsSuccess() {
			value, _ := result.Unwrap()
			fmt.Printf("   Input '%s': Success -> %d\n", input, value)
		} else {
			_, err := result.Unwrap()
			fmt.Printf("   Input '%s': Error -> %v\n", input, err)
		}
	}
	
	// 5. Data Structures
	fmt.Println("\n5. Data Structures:")
	
	// Stack
	stack := NewStack[string]()
	stack.Push("first")
	stack.Push("second")
	stack.Push("third")
	
	fmt.Printf("   Stack size: %d\n", stack.Size())
	if item, ok := stack.Pop(); ok {
		fmt.Printf("   Popped: %s\n", item)
	}
	
	// BST
	bst := NewBST[int](func(a, b int) int {
		if a < b {
			return -1
		} else if a > b {
			return 1
		}
		return 0
	})
	
	values := []int{50, 30, 70, 20, 40, 60, 80}
	for _, v := range values {
		bst.Insert(v)
	}
	
	fmt.Printf("   BST in-order: %v\n", bst.InOrder())
	fmt.Printf("   BST contains 40: %t\n", bst.Search(40))
	fmt.Printf("   BST contains 25: %t\n", bst.Search(25))
	
	// 6. Reflection
	fmt.Println("\n6. Reflection:")
	analyzer := &StructAnalyzer{}
	
	type SampleStruct struct {
		Name     string `json:"name" validate:"required"`
		Age      int    `json:"age" validate:"min=0"`
		Email    string `json:"email" validate:"email"`
		private  string
	}
	
	sample := SampleStruct{
		Name:    "John Doe",
		Age:     30,
		Email:   "john@example.com",
		private: "secret",
	}
	
	analysis := analyzer.AnalyzeStruct(sample)
	jsonData, _ := json.MarshalIndent(analysis, "   ", "  ")
	fmt.Printf("   Struct analysis:\n%s\n", string(jsonData))
	
	// 7. Performance Benchmarking
	fmt.Println("\n7. Performance Benchmarking:")
	runner := NewBenchmarkRunner()
	
	runner.Run("slice-append", func(n int) {
		slice := make([]int, 0)
		for i := 0; i < n; i++ {
			slice = append(slice, i)
		}
	}, 10000)
	
	runner.Run("slice-prealloc", func(n int) {
		slice := make([]int, 0, n)
		for i := 0; i < n; i++ {
			slice = append(slice, i)
		}
	}, 10000)
	
	runner.Run("map-creation", func(n int) {
		m := make(map[int]string)
		for i := 0; i < n; i++ {
			m[i] = fmt.Sprintf("value-%d", i)
		}
	}, 1000)
	
	runner.PrintResults()
	
	fmt.Println("\n=== Go Examples Complete ===")
}

// Utility Functions
// ================

// FunctionalUtils provides functional programming utilities
type FunctionalUtils struct{}

func (fu *FunctionalUtils) Map[T, U any](slice []T, fn func(T) U) []U {
	result := make([]U, len(slice))
	for i, v := range slice {
		result[i] = fn(v)
	}
	return result
}

func (fu *FunctionalUtils) Filter[T any](slice []T, predicate func(T) bool) []T {
	result := make([]T, 0)
	for _, v := range slice {
		if predicate(v) {
			result = append(result, v)
		}
	}
	return result
}

func (fu *FunctionalUtils) Reduce[T, U any](slice []T, initial U, fn func(U, T) U) U {
	result := initial
	for _, v := range slice {
		result = fn(result, v)
	}
	return result
}

func (fu *FunctionalUtils) GroupBy[T any, K comparable](slice []T, keyFn func(T) K) map[K][]T {
	result := make(map[K][]T)
	for _, v := range slice {
		key := keyFn(v)
		result[key] = append(result[key], v)
	}
	return result
}

// Cache provides a generic thread-safe cache
type Cache[K comparable, V any] struct {
	data   map[K]V
	mutex  sync.RWMutex
	maxSize int
}

func NewCache[K comparable, V any](maxSize int) *Cache[K, V] {
	return &Cache[K, V]{
		data:    make(map[K]V),
		maxSize: maxSize,
	}
}

func (c *Cache[K, V]) Get(key K) (V, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	value, exists := c.data[key]
	return value, exists
}

func (c *Cache[K, V]) Set(key K, value V) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	if len(c.data) >= c.maxSize {
		// Simple eviction: remove a random key
		for k := range c.data {
			delete(c.data, k)
			break
		}
	}
	
	c.data[key] = value
}

func (c *Cache[K, V]) Delete(key K) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	delete(c.data, key)
}

func (c *Cache[K, V]) Size() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	return len(c.data)
}