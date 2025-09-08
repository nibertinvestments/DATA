// Comprehensive Go Examples
// Demonstrates interfaces, goroutines, channels, error handling, and Go idioms

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ========== Custom Types and Structs ==========

// User represents a user in the system
type User struct {
	ID       uint64    `json:"id"`
	Username string    `json:"username"`
	Email    string    `json:"email"`
	Age      *int      `json:"age,omitempty"`
	Active   bool      `json:"active"`
	Created  time.Time `json:"created"`
	Updated  time.Time `json:"updated"`
}

// NewUser creates a new user with default values
func NewUser(username, email string) *User {
	now := time.Now()
	return &User{
		Username: username,
		Email:    email,
		Active:   true,
		Created:  now,
		Updated:  now,
	}
}

// SetAge sets the user's age
func (u *User) SetAge(age int) *User {
	u.Age = &age
	u.Updated = time.Now()
	return u
}

// IsAdult returns true if the user is 18 or older
func (u *User) IsAdult() bool {
	return u.Age != nil && *u.Age >= 18
}

// Deactivate deactivates the user
func (u *User) Deactivate() {
	u.Active = false
	u.Updated = time.Now()
}

// String implements the Stringer interface
func (u *User) String() string {
	ageStr := "unknown"
	if u.Age != nil {
		ageStr = fmt.Sprintf("%d", *u.Age)
	}
	return fmt.Sprintf("User{ID: %d, Username: %s, Email: %s, Age: %s, Active: %t}",
		u.ID, u.Username, u.Email, ageStr, u.Active)
}

// Product represents a product in the system
type Product struct {
	ID          uint64    `json:"id"`
	Name        string    `json:"name"`
	Price       float64   `json:"price"`
	Category    string    `json:"category"`
	InStock     bool      `json:"in_stock"`
	Tags        []string  `json:"tags"`
	Created     time.Time `json:"created"`
	Updated     time.Time `json:"updated"`
}

// NewProduct creates a new product
func NewProduct(name string, price float64, category string) *Product {
	now := time.Now()
	return &Product{
		Name:     name,
		Price:    price,
		Category: category,
		InStock:  true,
		Tags:     []string{},
		Created:  now,
		Updated:  now,
	}
}

// AddTag adds a tag to the product
func (p *Product) AddTag(tag string) {
	p.Tags = append(p.Tags, tag)
	p.Updated = time.Now()
}

// UpdatePrice updates the product price
func (p *Product) UpdatePrice(price float64) error {
	if price < 0 {
		return errors.New("price cannot be negative")
	}
	p.Price = price
	p.Updated = time.Now()
	return nil
}

// String implements the Stringer interface
func (p *Product) String() string {
	return fmt.Sprintf("Product{ID: %d, Name: %s, Price: %.2f, Category: %s, InStock: %t}",
		p.ID, p.Name, p.Price, p.Category, p.InStock)
}

// ========== Custom Errors ==========

// AppError represents application-specific errors
type AppError struct {
	Code    string
	Message string
	Err     error
}

func (e *AppError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s (%v)", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
	return e.Err
}

// Common error constructors
func NewValidationError(message string) *AppError {
	return &AppError{Code: "VALIDATION_ERROR", Message: message}
}

func NewNotFoundError(resource string, id uint64) *AppError {
	return &AppError{
		Code:    "NOT_FOUND",
		Message: fmt.Sprintf("%s with ID %d not found", resource, id),
	}
}

func NewDuplicateError(resource, field, value string) *AppError {
	return &AppError{
		Code:    "DUPLICATE_ERROR",
		Message: fmt.Sprintf("%s with %s '%s' already exists", resource, field, value),
	}
}

// ========== Interfaces ==========

// Repository interface for CRUD operations
type Repository[T any] interface {
	Create(ctx context.Context, entity T) (uint64, error)
	GetByID(ctx context.Context, id uint64) (T, error)
	Update(ctx context.Context, id uint64, entity T) error
	Delete(ctx context.Context, id uint64) error
	List(ctx context.Context) ([]T, error)
	Count(ctx context.Context) (int, error)
}

// Validator interface for entity validation
type Validator[T any] interface {
	Validate(entity T) error
}

// Searchable interface for search operations
type Searchable[T any] interface {
	Search(ctx context.Context, query string) ([]T, error)
}

// EventPublisher interface for event publishing
type EventPublisher interface {
	Publish(ctx context.Context, event Event) error
}

// Event represents a domain event
type Event struct {
	Type      string                 `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

// ========== Repository Implementation ==========

// InMemoryRepository is a generic in-memory repository
type InMemoryRepository[T any] struct {
	mu     sync.RWMutex
	data   map[uint64]T
	nextID uint64
}

// NewInMemoryRepository creates a new in-memory repository
func NewInMemoryRepository[T any]() *InMemoryRepository[T] {
	return &InMemoryRepository[T]{
		data:   make(map[uint64]T),
		nextID: 1,
	}
}

// Create adds a new entity to the repository
func (r *InMemoryRepository[T]) Create(ctx context.Context, entity T) (uint64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	id := r.nextID
	r.data[id] = entity
	r.nextID++
	
	return id, nil
}

// GetByID retrieves an entity by ID
func (r *InMemoryRepository[T]) GetByID(ctx context.Context, id uint64) (T, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	entity, exists := r.data[id]
	if !exists {
		var zero T
		return zero, NewNotFoundError("Entity", id)
	}
	
	return entity, nil
}

// Update updates an existing entity
func (r *InMemoryRepository[T]) Update(ctx context.Context, id uint64, entity T) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.data[id]; !exists {
		return NewNotFoundError("Entity", id)
	}
	
	r.data[id] = entity
	return nil
}

// Delete removes an entity by ID
func (r *InMemoryRepository[T]) Delete(ctx context.Context, id uint64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.data[id]; !exists {
		return NewNotFoundError("Entity", id)
	}
	
	delete(r.data, id)
	return nil
}

// List returns all entities
func (r *InMemoryRepository[T]) List(ctx context.Context) ([]T, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	entities := make([]T, 0, len(r.data))
	for _, entity := range r.data {
		entities = append(entities, entity)
	}
	
	return entities, nil
}

// Count returns the number of entities
func (r *InMemoryRepository[T]) Count(ctx context.Context) (int, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	return len(r.data), nil
}

// ========== Service Layer ==========

// UserValidator validates user entities
type UserValidator struct {
	MinUsernameLength int
	MaxUsernameLength int
}

// NewUserValidator creates a new user validator
func NewUserValidator() *UserValidator {
	return &UserValidator{
		MinUsernameLength: 3,
		MaxUsernameLength: 30,
	}
}

// Validate validates a user entity
func (v *UserValidator) Validate(user *User) error {
	if len(user.Username) < v.MinUsernameLength {
		return NewValidationError(fmt.Sprintf("username must be at least %d characters", v.MinUsernameLength))
	}
	
	if len(user.Username) > v.MaxUsernameLength {
		return NewValidationError(fmt.Sprintf("username must be at most %d characters", v.MaxUsernameLength))
	}
	
	if !strings.Contains(user.Email, "@") || !strings.Contains(user.Email, ".") {
		return NewValidationError("invalid email format")
	}
	
	if user.Age != nil && *user.Age < 0 {
		return NewValidationError("age cannot be negative")
	}
	
	return nil
}

// UserService provides user-related operations
type UserService struct {
	repo      Repository[*User]
	validator Validator[*User]
	publisher EventPublisher
}

// NewUserService creates a new user service
func NewUserService(repo Repository[*User], validator Validator[*User], publisher EventPublisher) *UserService {
	return &UserService{
		repo:      repo,
		validator: validator,
		publisher: publisher,
	}
}

// CreateUser creates a new user
func (s *UserService) CreateUser(ctx context.Context, username, email string) (uint64, error) {
	user := NewUser(username, email)
	
	if err := s.validator.Validate(user); err != nil {
		return 0, err
	}
	
	// Check for duplicates
	users, err := s.repo.List(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to list users: %w", err)
	}
	
	for _, existingUser := range users {
		if existingUser.Username == username {
			return 0, NewDuplicateError("User", "username", username)
		}
		if existingUser.Email == email {
			return 0, NewDuplicateError("User", "email", email)
		}
	}
	
	id, err := s.repo.Create(ctx, user)
	if err != nil {
		return 0, fmt.Errorf("failed to create user: %w", err)
	}
	
	user.ID = id
	
	// Publish event
	event := Event{
		Type:      "user.created",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"user_id":  id,
			"username": username,
			"email":    email,
		},
	}
	
	if err := s.publisher.Publish(ctx, event); err != nil {
		log.Printf("Failed to publish user.created event: %v", err)
	}
	
	return id, nil
}

// GetUser retrieves a user by ID
func (s *UserService) GetUser(ctx context.Context, id uint64) (*User, error) {
	return s.repo.GetByID(ctx, id)
}

// UpdateUser updates an existing user
func (s *UserService) UpdateUser(ctx context.Context, id uint64, user *User) error {
	if err := s.validator.Validate(user); err != nil {
		return err
	}
	
	user.ID = id
	user.Updated = time.Now()
	
	if err := s.repo.Update(ctx, id, user); err != nil {
		return fmt.Errorf("failed to update user: %w", err)
	}
	
	// Publish event
	event := Event{
		Type:      "user.updated",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"user_id": id,
		},
	}
	
	if err := s.publisher.Publish(ctx, event); err != nil {
		log.Printf("Failed to publish user.updated event: %v", err)
	}
	
	return nil
}

// DeleteUser deletes a user by ID
func (s *UserService) DeleteUser(ctx context.Context, id uint64) error {
	if err := s.repo.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}
	
	// Publish event
	event := Event{
		Type:      "user.deleted",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"user_id": id,
		},
	}
	
	if err := s.publisher.Publish(ctx, event); err != nil {
		log.Printf("Failed to publish user.deleted event: %v", err)
	}
	
	return nil
}

// ListUsers returns all users
func (s *UserService) ListUsers(ctx context.Context) ([]*User, error) {
	return s.repo.List(ctx)
}

// ========== Event System ==========

// InMemoryEventPublisher is an in-memory event publisher
type InMemoryEventPublisher struct {
	subscribers map[string][]chan Event
	mu          sync.RWMutex
}

// NewInMemoryEventPublisher creates a new in-memory event publisher
func NewInMemoryEventPublisher() *InMemoryEventPublisher {
	return &InMemoryEventPublisher{
		subscribers: make(map[string][]chan Event),
	}
}

// Publish publishes an event
func (p *InMemoryEventPublisher) Publish(ctx context.Context, event Event) error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	subscribers := p.subscribers[event.Type]
	for _, ch := range subscribers {
		select {
		case ch <- event:
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Non-blocking send, skip if channel is full
		}
	}
	
	return nil
}

// Subscribe subscribes to events of a specific type
func (p *InMemoryEventPublisher) Subscribe(eventType string, bufferSize int) <-chan Event {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	ch := make(chan Event, bufferSize)
	p.subscribers[eventType] = append(p.subscribers[eventType], ch)
	
	return ch
}

// ========== Concurrency Patterns ==========

// WorkerPool represents a pool of workers
type WorkerPool struct {
	workerCount int
	jobQueue    chan func()
	quit        chan bool
	wg          sync.WaitGroup
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workerCount int) *WorkerPool {
	return &WorkerPool{
		workerCount: workerCount,
		jobQueue:    make(chan func(), 100),
		quit:        make(chan bool),
	}
}

// Start starts the worker pool
func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workerCount; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

// Stop stops the worker pool
func (wp *WorkerPool) Stop() {
	close(wp.quit)
	wp.wg.Wait()
}

// Submit submits a job to the worker pool
func (wp *WorkerPool) Submit(job func()) {
	wp.jobQueue <- job
}

// worker is the worker goroutine
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()
	
	for {
		select {
		case job := <-wp.jobQueue:
			log.Printf("Worker %d executing job", id)
			job()
		case <-wp.quit:
			log.Printf("Worker %d stopping", id)
			return
		}
	}
}

// ========== Pipeline Pattern ==========

// Pipeline represents a data processing pipeline
type Pipeline[T any] struct {
	stages []func(T) T
}

// NewPipeline creates a new pipeline
func NewPipeline[T any]() *Pipeline[T] {
	return &Pipeline[T]{
		stages: make([]func(T) T, 0),
	}
}

// AddStage adds a stage to the pipeline
func (p *Pipeline[T]) AddStage(stage func(T) T) *Pipeline[T] {
	p.stages = append(p.stages, stage)
	return p
}

// Execute executes the pipeline on input data
func (p *Pipeline[T]) Execute(input T) T {
	result := input
	for _, stage := range p.stages {
		result = stage(result)
	}
	return result
}

// ExecuteAsync executes the pipeline asynchronously using channels
func (p *Pipeline[T]) ExecuteAsync(ctx context.Context, input <-chan T) <-chan T {
	output := input
	
	for _, stage := range p.stages {
		output = p.pipelineStage(ctx, output, stage)
	}
	
	return output
}

// pipelineStage creates a pipeline stage with channels
func (p *Pipeline[T]) pipelineStage(ctx context.Context, input <-chan T, stage func(T) T) <-chan T {
	output := make(chan T)
	
	go func() {
		defer close(output)
		for {
			select {
			case data, ok := <-input:
				if !ok {
					return
				}
				select {
				case output <- stage(data):
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

// ========== HTTP Server ==========

// UserHandler handles HTTP requests for users
type UserHandler struct {
	service *UserService
}

// NewUserHandler creates a new user handler
func NewUserHandler(service *UserService) *UserHandler {
	return &UserHandler{service: service}
}

// CreateUser handles POST /users
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req struct {
		Username string `json:"username"`
		Email    string `json:"email"`
		Age      *int   `json:"age,omitempty"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	id, err := h.service.CreateUser(r.Context(), req.Username, req.Email)
	if err != nil {
		var appErr *AppError
		if errors.As(err, &appErr) {
			switch appErr.Code {
			case "VALIDATION_ERROR", "DUPLICATE_ERROR":
				http.Error(w, appErr.Message, http.StatusBadRequest)
			default:
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		} else {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
		return
	}
	
	if req.Age != nil {
		user, err := h.service.GetUser(r.Context(), id)
		if err == nil {
			user.SetAge(*req.Age)
			h.service.UpdateUser(r.Context(), id, user)
		}
	}
	
	resp := map[string]interface{}{
		"id":      id,
		"message": "User created successfully",
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(resp)
}

// GetUser handles GET /users/{id}
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Extract ID from URL path (simplified)
	path := strings.TrimPrefix(r.URL.Path, "/users/")
	id, err := strconv.ParseUint(path, 10, 64)
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}
	
	user, err := h.service.GetUser(r.Context(), id)
	if err != nil {
		var appErr *AppError
		if errors.As(err, &appErr) && appErr.Code == "NOT_FOUND" {
			http.Error(w, appErr.Message, http.StatusNotFound)
		} else {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

// ========== File Operations ==========

// SaveUsersToFile saves users to a JSON file
func SaveUsersToFile(users []*User, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	if err := encoder.Encode(users); err != nil {
		return fmt.Errorf("failed to encode users: %w", err)
	}
	
	return nil
}

// LoadUsersFromFile loads users from a JSON file
func LoadUsersFromFile(filename string) ([]*User, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	var users []*User
	decoder := json.NewDecoder(file)
	
	if err := decoder.Decode(&users); err != nil {
		return nil, fmt.Errorf("failed to decode users: %w", err)
	}
	
	return users, nil
}

// ProcessFileLines processes a file line by line
func ProcessFileLines(filename string, processor func(string) error) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	scanner := bufio.NewScanner(file)
	lineNum := 0
	
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		
		if err := processor(line); err != nil {
			return fmt.Errorf("error processing line %d: %w", lineNum, err)
		}
	}
	
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}
	
	return nil
}

// ========== Utility Functions ==========

// Filter filters a slice based on a predicate
func Filter[T any](slice []T, predicate func(T) bool) []T {
	var result []T
	for _, item := range slice {
		if predicate(item) {
			result = append(result, item)
		}
	}
	return result
}

// Map applies a function to each element of a slice
func Map[T, U any](slice []T, mapper func(T) U) []U {
	result := make([]U, len(slice))
	for i, item := range slice {
		result[i] = mapper(item)
	}
	return result
}

// Reduce reduces a slice to a single value
func Reduce[T, U any](slice []T, initial U, reducer func(U, T) U) U {
	result := initial
	for _, item := range slice {
		result = reducer(result, item)
	}
	return result
}

// Parallel applies a function to slice elements in parallel
func Parallel[T, U any](slice []T, mapper func(T) U, maxWorkers int) []U {
	if len(slice) == 0 {
		return []U{}
	}
	
	if maxWorkers <= 0 {
		maxWorkers = len(slice)
	}
	
	jobs := make(chan int, len(slice))
	results := make([]U, len(slice))
	
	var wg sync.WaitGroup
	
	// Start workers
	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = mapper(slice[i])
			}
		}()
	}
	
	// Send jobs
	for i := range slice {
		jobs <- i
	}
	close(jobs)
	
	wg.Wait()
	return results
}

// ========== Demonstration Functions ==========

func demonstrateBasicOperations() error {
	fmt.Println("=== Basic Operations Demo ===")
	
	repo := NewInMemoryRepository[*User]()
	validator := NewUserValidator()
	publisher := NewInMemoryEventPublisher()
	service := NewUserService(repo, validator, publisher)
	
	ctx := context.Background()
	
	// Subscribe to events
	userCreatedCh := publisher.Subscribe("user.created", 10)
	go func() {
		for event := range userCreatedCh {
			fmt.Printf("Event received: %s at %s\n", event.Type, event.Timestamp.Format(time.RFC3339))
		}
	}()
	
	// Create users
	id1, err := service.CreateUser(ctx, "alice", "alice@example.com")
	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}
	fmt.Printf("Created user with ID: %d\n", id1)
	
	id2, err := service.CreateUser(ctx, "bob", "bob@example.com")
	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}
	fmt.Printf("Created user with ID: %d\n", id2)
	
	// Get user
	user, err := service.GetUser(ctx, id1)
	if err != nil {
		return fmt.Errorf("failed to get user: %w", err)
	}
	fmt.Printf("Retrieved user: %s\n", user)
	
	// Update user
	user.SetAge(25)
	if err := service.UpdateUser(ctx, id1, user); err != nil {
		return fmt.Errorf("failed to update user: %w", err)
	}
	fmt.Printf("Updated user: %s\n", user)
	
	// List users
	users, err := service.ListUsers(ctx)
	if err != nil {
		return fmt.Errorf("failed to list users: %w", err)
	}
	fmt.Printf("Total users: %d\n", len(users))
	
	return nil
}

func demonstrateConcurrency() error {
	fmt.Println("\n=== Concurrency Demo ===")
	
	// Worker pool demo
	pool := NewWorkerPool(3)
	pool.Start()
	
	// Submit jobs
	for i := 0; i < 10; i++ {
		jobID := i
		pool.Submit(func() {
			time.Sleep(time.Duration(rand.Intn(1000)) * time.Millisecond)
			fmt.Printf("Job %d completed\n", jobID)
		})
	}
	
	time.Sleep(3 * time.Second)
	pool.Stop()
	
	// Pipeline demo
	pipeline := NewPipeline[int]()
	pipeline.AddStage(func(x int) int { return x * 2 })
	pipeline.AddStage(func(x int) int { return x + 10 })
	pipeline.AddStage(func(x int) int { return x * x })
	
	result := pipeline.Execute(5)
	fmt.Printf("Pipeline result: 5 -> %d\n", result)
	
	// Async pipeline demo
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	input := make(chan int, 10)
	go func() {
		defer close(input)
		for i := 1; i <= 5; i++ {
			input <- i
		}
	}()
	
	output := pipeline.ExecuteAsync(ctx, input)
	
	fmt.Print("Async pipeline results: ")
	for result := range output {
		fmt.Printf("%d ", result)
	}
	fmt.Println()
	
	return nil
}

func demonstrateFunctionalProgramming() error {
	fmt.Println("\n=== Functional Programming Demo ===")
	
	users := []*User{
		NewUser("alice", "alice@example.com").SetAge(25),
		NewUser("bob", "bob@example.com").SetAge(17),
		NewUser("charlie", "charlie@example.com").SetAge(30),
		NewUser("diana", "diana@example.com").SetAge(22),
	}
	
	// Filter adult users
	adults := Filter(users, func(u *User) bool { return u.IsAdult() })
	fmt.Printf("Adult users: %d out of %d\n", len(adults), len(users))
	
	// Map usernames
	usernames := Map(users, func(u *User) string { return u.Username })
	fmt.Printf("Usernames: %v\n", usernames)
	
	// Reduce to total age
	totalAge := Reduce(users, 0, func(acc int, u *User) int {
		if u.Age != nil {
			return acc + *u.Age
		}
		return acc
	})
	fmt.Printf("Total age: %d\n", totalAge)
	
	// Parallel processing
	results := Parallel([]int{1, 2, 3, 4, 5}, func(x int) int {
		time.Sleep(100 * time.Millisecond) // Simulate work
		return x * x
	}, 3)
	fmt.Printf("Parallel squares: %v\n", results)
	
	return nil
}

func demonstrateFileOperations() error {
	fmt.Println("\n=== File Operations Demo ===")
	
	users := []*User{
		NewUser("alice", "alice@example.com").SetAge(25),
		NewUser("bob", "bob@example.com").SetAge(30),
		NewUser("charlie", "charlie@example.com").SetAge(22),
	}
	
	filename := "/tmp/users.json"
	
	// Save users to file
	if err := SaveUsersToFile(users, filename); err != nil {
		return fmt.Errorf("failed to save users: %w", err)
	}
	fmt.Printf("Saved %d users to %s\n", len(users), filename)
	
	// Load users from file
	loadedUsers, err := LoadUsersFromFile(filename)
	if err != nil {
		return fmt.Errorf("failed to load users: %w", err)
	}
	fmt.Printf("Loaded %d users from %s\n", len(loadedUsers), filename)
	
	// Process file line by line
	logFile := "/tmp/sample.log"
	if err := os.WriteFile(logFile, []byte("line 1\nline 2\nline 3\n"), 0644); err != nil {
		return fmt.Errorf("failed to write log file: %w", err)
	}
	
	lineCount := 0
	if err := ProcessFileLines(logFile, func(line string) error {
		lineCount++
		fmt.Printf("Processing line %d: %s\n", lineCount, line)
		return nil
	}); err != nil {
		return fmt.Errorf("failed to process file: %w", err)
	}
	
	return nil
}

func demonstrateErrorHandling() error {
	fmt.Println("\n=== Error Handling Demo ===")
	
	repo := NewInMemoryRepository[*User]()
	validator := NewUserValidator()
	publisher := NewInMemoryEventPublisher()
	service := NewUserService(repo, validator, publisher)
	
	ctx := context.Background()
	
	// Test validation errors
	testCases := []struct {
		username string
		email    string
	}{
		{"", "invalid@email.com"},           // Empty username
		{"ab", "short@email.com"},           // Username too short
		{"validuser", "invalidemail"},       // Invalid email
		{strings.Repeat("a", 50), "long@email.com"}, // Username too long
	}
	
	for _, tc := range testCases {
		_, err := service.CreateUser(ctx, tc.username, tc.email)
		if err != nil {
			var appErr *AppError
			if errors.As(err, &appErr) {
				fmt.Printf("Validation error for '%s': %s\n", tc.username, appErr.Message)
			} else {
				fmt.Printf("Unexpected error for '%s': %v\n", tc.username, err)
			}
		}
	}
	
	// Test not found error
	_, err := service.GetUser(ctx, 999)
	if err != nil {
		fmt.Printf("Not found error: %v\n", err)
	}
	
	return nil
}

func main() {
	fmt.Println("=== Comprehensive Go Examples ===\n")
	
	if err := demonstrateBasicOperations(); err != nil {
		log.Fatalf("Basic operations demo failed: %v", err)
	}
	
	if err := demonstrateConcurrency(); err != nil {
		log.Fatalf("Concurrency demo failed: %v", err)
	}
	
	if err := demonstrateFunctionalProgramming(); err != nil {
		log.Fatalf("Functional programming demo failed: %v", err)
	}
	
	if err := demonstrateFileOperations(); err != nil {
		log.Fatalf("File operations demo failed: %v", err)
	}
	
	if err := demonstrateErrorHandling(); err != nil {
		log.Fatalf("Error handling demo failed: %v", err)
	}
	
	fmt.Println("\n=== Go Features Demonstrated ===")
	fmt.Println("- Interfaces and polymorphism")
	fmt.Println("- Goroutines and channels")
	fmt.Println("- Generics (Go 1.18+)")
	fmt.Println("- Error handling with custom error types")
	fmt.Println("- Context for cancellation and timeouts")
	fmt.Println("- Worker pools and pipeline patterns")
	fmt.Println("- JSON encoding/decoding")
	fmt.Println("- File I/O operations")
	fmt.Println("- HTTP server implementation")
	fmt.Println("- Functional programming patterns")
	fmt.Println("- Repository and service patterns")
	fmt.Println("- Event-driven architecture")
}