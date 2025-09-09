// Comprehensive Go Microservices and Concurrent Programming Examples
// Demonstrates Go's strengths in concurrent programming, networking, and system design

package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"golang.org/x/time/rate"
)

// ============ Domain Models and Value Objects ============

// UserID represents a unique user identifier with type safety
type UserID string

func NewUserID() UserID {
	b := make([]byte, 16)
	rand.Read(b)
	return UserID(base64.URLEncoding.EncodeToString(b))
}

func (id UserID) String() string {
	return string(id)
}

func (id UserID) IsValid() bool {
	return len(id) > 0
}

// Email value object with validation
type Email string

func NewEmail(address string) (Email, error) {
	if address == "" {
		return "", errors.New("email address cannot be empty")
	}
	
	// Simple email validation
	if !strings.Contains(address, "@") || !strings.Contains(address, ".") {
		return "", errors.New("invalid email format")
	}
	
	return Email(strings.ToLower(strings.TrimSpace(address))), nil
}

func (e Email) String() string {
	return string(e)
}

func (e Email) Domain() string {
	parts := strings.Split(string(e), "@")
	if len(parts) == 2 {
		return parts[1]
	}
	return ""
}

// Money value object for handling currency
type Money struct {
	Amount   int64  `json:"amount"`   // Amount in cents to avoid floating point issues
	Currency string `json:"currency"`
}

func NewMoney(amount float64, currency string) Money {
	return Money{
		Amount:   int64(amount * 100), // Convert to cents
		Currency: strings.ToUpper(currency),
	}
}

func (m Money) ToFloat() float64 {
	return float64(m.Amount) / 100.0
}

func (m Money) Add(other Money) (Money, error) {
	if m.Currency != other.Currency {
		return Money{}, fmt.Errorf("cannot add different currencies: %s and %s", m.Currency, other.Currency)
	}
	
	return Money{
		Amount:   m.Amount + other.Amount,
		Currency: m.Currency,
	}, nil
}

func (m Money) Subtract(other Money) (Money, error) {
	if m.Currency != other.Currency {
		return Money{}, fmt.Errorf("cannot subtract different currencies: %s and %s", m.Currency, other.Currency)
	}
	
	if m.Amount < other.Amount {
		return Money{}, errors.New("insufficient funds")
	}
	
	return Money{
		Amount:   m.Amount - other.Amount,
		Currency: m.Currency,
	}, nil
}

func (m Money) String() string {
	return fmt.Sprintf("%.2f %s", m.ToFloat(), m.Currency)
}

// UserRole represents user permissions
type UserRole int

const (
	RoleGuest UserRole = iota
	RoleUser
	RoleModerator
	RoleAdmin
	RoleSuperAdmin
)

var roleNames = map[UserRole]string{
	RoleGuest:      "guest",
	RoleUser:       "user",
	RoleModerator:  "moderator",
	RoleAdmin:      "admin",
	RoleSuperAdmin: "super_admin",
}

var rolePermissions = map[UserRole][]string{
	RoleGuest:      {"read:public"},
	RoleUser:       {"read:public", "read:own", "write:own"},
	RoleModerator:  {"read:public", "read:own", "write:own", "moderate:content"},
	RoleAdmin:      {"read:all", "write:all", "delete:all", "manage:users"},
	RoleSuperAdmin: {"*"},
}

func (r UserRole) String() string {
	if name, ok := roleNames[r]; ok {
		return name
	}
	return "unknown"
}

func (r UserRole) HasPermission(permission string) bool {
	permissions := rolePermissions[r]
	for _, perm := range permissions {
		if perm == "*" || perm == permission {
			return true
		}
	}
	return false
}

func (r UserRole) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.String())
}

func (r *UserRole) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	
	for role, name := range roleNames {
		if name == s {
			*r = role
			return nil
		}
	}
	
	return fmt.Errorf("unknown role: %s", s)
}

// TaskStatus with state transitions
type TaskStatus int

const (
	TaskStatusDraft TaskStatus = iota
	TaskStatusActive
	TaskStatusInProgress
	TaskStatusReview
	TaskStatusCompleted
	TaskStatusCancelled
	TaskStatusArchived
)

var taskStatusNames = map[TaskStatus]string{
	TaskStatusDraft:      "draft",
	TaskStatusActive:     "active",
	TaskStatusInProgress: "in_progress",
	TaskStatusReview:     "review",
	TaskStatusCompleted:  "completed",
	TaskStatusCancelled:  "cancelled",
	TaskStatusArchived:   "archived",
}

var taskStatusTransitions = map[TaskStatus][]TaskStatus{
	TaskStatusDraft:      {TaskStatusActive, TaskStatusCancelled},
	TaskStatusActive:     {TaskStatusInProgress, TaskStatusCancelled},
	TaskStatusInProgress: {TaskStatusReview, TaskStatusActive, TaskStatusCancelled},
	TaskStatusReview:     {TaskStatusCompleted, TaskStatusInProgress, TaskStatusCancelled},
	TaskStatusCompleted:  {TaskStatusArchived, TaskStatusActive},
	TaskStatusCancelled:  {TaskStatusActive},
	TaskStatusArchived:   {},
}

func (ts TaskStatus) String() string {
	if name, ok := taskStatusNames[ts]; ok {
		return name
	}
	return "unknown"
}

func (ts TaskStatus) CanTransitionTo(newStatus TaskStatus) bool {
	allowedTransitions := taskStatusTransitions[ts]
	for _, allowed := range allowedTransitions {
		if allowed == newStatus {
			return true
		}
	}
	return false
}

func (ts TaskStatus) MarshalJSON() ([]byte, error) {
	return json.Marshal(ts.String())
}

func (ts *TaskStatus) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	
	for status, name := range taskStatusNames {
		if name == s {
			*ts = status
			return nil
		}
	}
	
	return fmt.Errorf("unknown task status: %s", s)
}

// ============ Domain Entities ============

type User struct {
	ID          UserID    `json:"id"`
	FirstName   string    `json:"first_name"`
	LastName    string    `json:"last_name"`
	Email       Email     `json:"email"`
	Role        UserRole  `json:"role"`
	IsActive    bool      `json:"is_active"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	LastLoginAt *time.Time `json:"last_login_at,omitempty"`
	Profile     UserProfile `json:"profile"`
}

type UserProfile struct {
	Bio         string            `json:"bio,omitempty"`
	Avatar      string            `json:"avatar,omitempty"`
	DateOfBirth *time.Time        `json:"date_of_birth,omitempty"`
	Location    string            `json:"location,omitempty"`
	Website     string            `json:"website,omitempty"`
	SocialLinks map[string]string `json:"social_links,omitempty"`
}

func NewUser(firstName, lastName string, email Email, role UserRole) *User {
	now := time.Now()
	return &User{
		ID:        NewUserID(),
		FirstName: strings.TrimSpace(firstName),
		LastName:  strings.TrimSpace(lastName),
		Email:     email,
		Role:      role,
		IsActive:  true,
		CreatedAt: now,
		UpdatedAt: now,
		Profile: UserProfile{
			SocialLinks: make(map[string]string),
		},
	}
}

func (u *User) FullName() string {
	return fmt.Sprintf("%s %s", u.FirstName, u.LastName)
}

func (u *User) Age() *int {
	if u.Profile.DateOfBirth == nil {
		return nil
	}
	
	years := int(time.Since(*u.Profile.DateOfBirth).Hours() / 24 / 365)
	return &years
}

func (u *User) IsAdult() bool {
	age := u.Age()
	return age != nil && *age >= 18
}

func (u *User) HasPermission(permission string) bool {
	return u.IsActive && u.Role.HasPermission(permission)
}

func (u *User) UpdateProfile(profile UserProfile) {
	u.Profile = profile
	u.UpdatedAt = time.Now()
}

func (u *User) RecordLogin() {
	now := time.Now()
	u.LastLoginAt = &now
	u.UpdatedAt = now
}

func (u *User) Deactivate() {
	u.IsActive = false
	u.UpdatedAt = time.Now()
}

func (u *User) Activate() {
	u.IsActive = true
	u.UpdatedAt = time.Now()
}

func (u *User) Validate() error {
	if u.FirstName == "" {
		return errors.New("first name is required")
	}
	if u.LastName == "" {
		return errors.New("last name is required")
	}
	if u.Email == "" {
		return errors.New("email is required")
	}
	return nil
}

type Task struct {
	ID            int           `json:"id"`
	Title         string        `json:"title"`
	Description   string        `json:"description"`
	Status        TaskStatus    `json:"status"`
	Priority      int           `json:"priority"` // 1-4, higher is more urgent
	AssigneeID    UserID        `json:"assignee_id"`
	CreatorID     UserID        `json:"creator_id"`
	DueDate       *time.Time    `json:"due_date,omitempty"`
	CompletedAt   *time.Time    `json:"completed_at,omitempty"`
	EstimatedHours *float64     `json:"estimated_hours,omitempty"`
	ActualHours   float64       `json:"actual_hours"`
	Tags          []string      `json:"tags"`
	CreatedAt     time.Time     `json:"created_at"`
	UpdatedAt     time.Time     `json:"updated_at"`
}

func NewTask(title, description string, priority int, assigneeID, creatorID UserID) *Task {
	now := time.Now()
	return &Task{
		Title:       strings.TrimSpace(title),
		Description: strings.TrimSpace(description),
		Status:      TaskStatusDraft,
		Priority:    priority,
		AssigneeID:  assigneeID,
		CreatorID:   creatorID,
		Tags:        make([]string, 0),
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}

func (t *Task) ChangeStatus(newStatus TaskStatus) error {
	if !t.Status.CanTransitionTo(newStatus) {
		return fmt.Errorf("cannot transition from %s to %s", t.Status, newStatus)
	}
	
	t.Status = newStatus
	t.UpdatedAt = time.Now()
	
	if newStatus == TaskStatusCompleted {
		now := time.Now()
		t.CompletedAt = &now
	} else if t.Status == TaskStatusCompleted {
		t.CompletedAt = nil
	}
	
	return nil
}

func (t *Task) AddTag(tag string) {
	tag = strings.TrimSpace(strings.ToLower(tag))
	if tag == "" {
		return
	}
	
	for _, existingTag := range t.Tags {
		if existingTag == tag {
			return
		}
	}
	
	t.Tags = append(t.Tags, tag)
	t.UpdatedAt = time.Now()
}

func (t *Task) RemoveTag(tag string) {
	tag = strings.TrimSpace(strings.ToLower(tag))
	for i, existingTag := range t.Tags {
		if existingTag == tag {
			t.Tags = append(t.Tags[:i], t.Tags[i+1:]...)
			t.UpdatedAt = time.Now()
			return
		}
	}
}

func (t *Task) RecordWorkTime(hours float64) error {
	if hours <= 0 {
		return errors.New("work time must be positive")
	}
	
	t.ActualHours += hours
	t.UpdatedAt = time.Now()
	return nil
}

func (t *Task) IsOverdue() bool {
	return t.DueDate != nil && 
		   t.DueDate.Before(time.Now()) && 
		   t.Status != TaskStatusCompleted && 
		   t.Status != TaskStatusCancelled
}

func (t *Task) IsCompleted() bool {
	return t.Status == TaskStatusCompleted
}

func (t *Task) TimeToCompletion() *time.Duration {
	if t.CompletedAt == nil {
		return nil
	}
	
	duration := t.CompletedAt.Sub(t.CreatedAt)
	return &duration
}

func (t *Task) Validate() error {
	if t.Title == "" {
		return errors.New("title is required")
	}
	if len(t.Title) > 200 {
		return errors.New("title cannot exceed 200 characters")
	}
	if len(t.Description) > 2000 {
		return errors.New("description cannot exceed 2000 characters")
	}
	if t.Priority < 1 || t.Priority > 4 {
		return errors.New("priority must be between 1 and 4")
	}
	if !t.AssigneeID.IsValid() {
		return errors.New("assignee ID is required")
	}
	if !t.CreatorID.IsValid() {
		return errors.New("creator ID is required")
	}
	return nil
}

// ============ Repository Pattern with Interfaces ============

type UserRepository interface {
	Create(ctx context.Context, user *User) error
	GetByID(ctx context.Context, id UserID) (*User, error)
	GetByEmail(ctx context.Context, email Email) (*User, error)
	GetAll(ctx context.Context, options QueryOptions) ([]*User, error)
	Update(ctx context.Context, user *User) error
	Delete(ctx context.Context, id UserID) error
	Count(ctx context.Context, filter map[string]interface{}) (int, error)
}

type TaskRepository interface {
	Create(ctx context.Context, task *Task) error
	GetByID(ctx context.Context, id int) (*Task, error)
	GetAll(ctx context.Context, options QueryOptions) ([]*Task, error)
	GetByAssignee(ctx context.Context, assigneeID UserID) ([]*Task, error)
	GetByStatus(ctx context.Context, status TaskStatus) ([]*Task, error)
	GetOverdue(ctx context.Context) ([]*Task, error)
	Update(ctx context.Context, task *Task) error
	Delete(ctx context.Context, id int) error
	Count(ctx context.Context, filter map[string]interface{}) (int, error)
}

type QueryOptions struct {
	Limit  int
	Offset int
	SortBy string
	SortOrder string // "asc" or "desc"
	Filter map[string]interface{}
}

// In-memory implementations for demo
type InMemoryUserRepository struct {
	mu    sync.RWMutex
	users map[UserID]*User
}

func NewInMemoryUserRepository() *InMemoryUserRepository {
	return &InMemoryUserRepository{
		users: make(map[UserID]*User),
	}
}

func (r *InMemoryUserRepository) Create(ctx context.Context, user *User) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if err := user.Validate(); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}
	
	// Check for duplicate email
	for _, existingUser := range r.users {
		if existingUser.Email == user.Email {
			return errors.New("user with this email already exists")
		}
	}
	
	r.users[user.ID] = user
	return nil
}

func (r *InMemoryUserRepository) GetByID(ctx context.Context, id UserID) (*User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	user, exists := r.users[id]
	if !exists {
		return nil, errors.New("user not found")
	}
	
	// Return a copy to prevent external modifications
	userCopy := *user
	return &userCopy, nil
}

func (r *InMemoryUserRepository) GetByEmail(ctx context.Context, email Email) (*User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	for _, user := range r.users {
		if user.Email == email {
			userCopy := *user
			return &userCopy, nil
		}
	}
	
	return nil, errors.New("user not found")
}

func (r *InMemoryUserRepository) GetAll(ctx context.Context, options QueryOptions) ([]*User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	users := make([]*User, 0, len(r.users))
	for _, user := range r.users {
		// Apply filters
		if r.matchesFilter(user, options.Filter) {
			userCopy := *user
			users = append(users, &userCopy)
		}
	}
	
	// Apply pagination
	start := options.Offset
	if start >= len(users) {
		return []*User{}, nil
	}
	
	end := start + options.Limit
	if options.Limit == 0 || end > len(users) {
		end = len(users)
	}
	
	return users[start:end], nil
}

func (r *InMemoryUserRepository) Update(ctx context.Context, user *User) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if err := user.Validate(); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}
	
	if _, exists := r.users[user.ID]; !exists {
		return errors.New("user not found")
	}
	
	user.UpdatedAt = time.Now()
	r.users[user.ID] = user
	return nil
}

func (r *InMemoryUserRepository) Delete(ctx context.Context, id UserID) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.users[id]; !exists {
		return errors.New("user not found")
	}
	
	delete(r.users, id)
	return nil
}

func (r *InMemoryUserRepository) Count(ctx context.Context, filter map[string]interface{}) (int, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	count := 0
	for _, user := range r.users {
		if r.matchesFilter(user, filter) {
			count++
		}
	}
	
	return count, nil
}

func (r *InMemoryUserRepository) matchesFilter(user *User, filter map[string]interface{}) bool {
	if len(filter) == 0 {
		return true
	}
	
	for key, value := range filter {
		switch key {
		case "is_active":
			if user.IsActive != value.(bool) {
				return false
			}
		case "role":
			if user.Role.String() != value.(string) {
				return false
			}
		}
	}
	
	return true
}

type InMemoryTaskRepository struct {
	mu     sync.RWMutex
	tasks  map[int]*Task
	nextID int64
}

func NewInMemoryTaskRepository() *InMemoryTaskRepository {
	return &InMemoryTaskRepository{
		tasks: make(map[int]*Task),
		nextID: 1,
	}
}

func (r *InMemoryTaskRepository) Create(ctx context.Context, task *Task) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if err := task.Validate(); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}
	
	task.ID = int(atomic.AddInt64(&r.nextID, 1))
	r.tasks[task.ID] = task
	return nil
}

func (r *InMemoryTaskRepository) GetByID(ctx context.Context, id int) (*Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	task, exists := r.tasks[id]
	if !exists {
		return nil, errors.New("task not found")
	}
	
	taskCopy := *task
	taskCopy.Tags = make([]string, len(task.Tags))
	copy(taskCopy.Tags, task.Tags)
	
	return &taskCopy, nil
}

func (r *InMemoryTaskRepository) GetAll(ctx context.Context, options QueryOptions) ([]*Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	tasks := make([]*Task, 0, len(r.tasks))
	for _, task := range r.tasks {
		if r.matchesTaskFilter(task, options.Filter) {
			taskCopy := *task
			taskCopy.Tags = make([]string, len(task.Tags))
			copy(taskCopy.Tags, task.Tags)
			tasks = append(tasks, &taskCopy)
		}
	}
	
	// Apply pagination
	start := options.Offset
	if start >= len(tasks) {
		return []*Task{}, nil
	}
	
	end := start + options.Limit
	if options.Limit == 0 || end > len(tasks) {
		end = len(tasks)
	}
	
	return tasks[start:end], nil
}

func (r *InMemoryTaskRepository) GetByAssignee(ctx context.Context, assigneeID UserID) ([]*Task, error) {
	return r.GetAll(ctx, QueryOptions{
		Filter: map[string]interface{}{
			"assignee_id": assigneeID,
		},
	})
}

func (r *InMemoryTaskRepository) GetByStatus(ctx context.Context, status TaskStatus) ([]*Task, error) {
	return r.GetAll(ctx, QueryOptions{
		Filter: map[string]interface{}{
			"status": status,
		},
	})
}

func (r *InMemoryTaskRepository) GetOverdue(ctx context.Context) ([]*Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	var overdueTasks []*Task
	for _, task := range r.tasks {
		if task.IsOverdue() {
			taskCopy := *task
			taskCopy.Tags = make([]string, len(task.Tags))
			copy(taskCopy.Tags, task.Tags)
			overdueTasks = append(overdueTasks, &taskCopy)
		}
	}
	
	return overdueTasks, nil
}

func (r *InMemoryTaskRepository) Update(ctx context.Context, task *Task) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if err := task.Validate(); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}
	
	if _, exists := r.tasks[task.ID]; !exists {
		return errors.New("task not found")
	}
	
	task.UpdatedAt = time.Now()
	r.tasks[task.ID] = task
	return nil
}

func (r *InMemoryTaskRepository) Delete(ctx context.Context, id int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.tasks[id]; !exists {
		return errors.New("task not found")
	}
	
	delete(r.tasks, id)
	return nil
}

func (r *InMemoryTaskRepository) Count(ctx context.Context, filter map[string]interface{}) (int, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	count := 0
	for _, task := range r.tasks {
		if r.matchesTaskFilter(task, filter) {
			count++
		}
	}
	
	return count, nil
}

func (r *InMemoryTaskRepository) matchesTaskFilter(task *Task, filter map[string]interface{}) bool {
	if len(filter) == 0 {
		return true
	}
	
	for key, value := range filter {
		switch key {
		case "status":
			if task.Status != value.(TaskStatus) {
				return false
			}
		case "assignee_id":
			if task.AssigneeID != value.(UserID) {
				return false
			}
		case "priority":
			if task.Priority != value.(int) {
				return false
			}
		}
	}
	
	return true
}

// ============ Service Layer ============

type UserService struct {
	userRepo UserRepository
	logger   *log.Logger
}

func NewUserService(userRepo UserRepository, logger *log.Logger) *UserService {
	return &UserService{
		userRepo: userRepo,
		logger:   logger,
	}
}

func (s *UserService) CreateUser(ctx context.Context, firstName, lastName, email string, role UserRole) (*User, error) {
	s.logger.Printf("Creating user: %s %s <%s>", firstName, lastName, email)
	
	emailObj, err := NewEmail(email)
	if err != nil {
		return nil, fmt.Errorf("invalid email: %w", err)
	}
	
	user := NewUser(firstName, lastName, emailObj, role)
	
	if err := s.userRepo.Create(ctx, user); err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}
	
	s.logger.Printf("User created successfully: %s", user.ID)
	return user, nil
}

func (s *UserService) GetUser(ctx context.Context, id UserID) (*User, error) {
	return s.userRepo.GetByID(ctx, id)
}

func (s *UserService) UpdateUser(ctx context.Context, id UserID, updates map[string]interface{}) (*User, error) {
	user, err := s.userRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	
	// Apply updates using reflection (simplified for demo)
	v := reflect.ValueOf(user).Elem()
	t := v.Type()
	
	for i := 0; i < v.NumField(); i++ {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "" {
			continue
		}
		
		// Remove omitempty and other options
		jsonName := strings.Split(jsonTag, ",")[0]
		if value, exists := updates[jsonName]; exists && v.Field(i).CanSet() {
			v.Field(i).Set(reflect.ValueOf(value))
		}
	}
	
	if err := s.userRepo.Update(ctx, user); err != nil {
		return nil, fmt.Errorf("failed to update user: %w", err)
	}
	
	s.logger.Printf("User updated successfully: %s", user.ID)
	return user, nil
}

func (s *UserService) DeleteUser(ctx context.Context, id UserID) error {
	if err := s.userRepo.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}
	
	s.logger.Printf("User deleted successfully: %s", id)
	return nil
}

func (s *UserService) SearchUsers(ctx context.Context, filter map[string]interface{}, limit, offset int) ([]*User, error) {
	options := QueryOptions{
		Limit:  limit,
		Offset: offset,
		Filter: filter,
	}
	
	return s.userRepo.GetAll(ctx, options)
}

func (s *UserService) AuthenticateUser(ctx context.Context, email, password string) (*User, error) {
	emailObj, err := NewEmail(email)
	if err != nil {
		return nil, fmt.Errorf("invalid email: %w", err)
	}
	
	user, err := s.userRepo.GetByEmail(ctx, emailObj)
	if err != nil {
		return nil, errors.New("invalid credentials")
	}
	
	if !user.IsActive {
		return nil, errors.New("user account is deactivated")
	}
	
	// In a real application, you would verify the password hash
	// For demo purposes, we'll simulate successful authentication
	if password == "" {
		return nil, errors.New("invalid credentials")
	}
	
	user.RecordLogin()
	if err := s.userRepo.Update(ctx, user); err != nil {
		s.logger.Printf("Failed to update user login time: %v", err)
	}
	
	s.logger.Printf("User authenticated successfully: %s", user.ID)
	return user, nil
}

type TaskService struct {
	taskRepo TaskRepository
	userRepo UserRepository
	logger   *log.Logger
}

func NewTaskService(taskRepo TaskRepository, userRepo UserRepository, logger *log.Logger) *TaskService {
	return &TaskService{
		taskRepo: taskRepo,
		userRepo: userRepo,
		logger:   logger,
	}
}

func (s *TaskService) CreateTask(ctx context.Context, title, description string, priority int, assigneeID, creatorID UserID, dueDate *time.Time) (*Task, error) {
	s.logger.Printf("Creating task: %s", title)
	
	// Validate that assignee and creator exist
	if _, err := s.userRepo.GetByID(ctx, assigneeID); err != nil {
		return nil, fmt.Errorf("assignee not found: %w", err)
	}
	
	if _, err := s.userRepo.GetByID(ctx, creatorID); err != nil {
		return nil, fmt.Errorf("creator not found: %w", err)
	}
	
	task := NewTask(title, description, priority, assigneeID, creatorID)
	task.DueDate = dueDate
	
	if err := s.taskRepo.Create(ctx, task); err != nil {
		return nil, fmt.Errorf("failed to create task: %w", err)
	}
	
	s.logger.Printf("Task created successfully: %d", task.ID)
	return task, nil
}

func (s *TaskService) GetTask(ctx context.Context, id int) (*Task, error) {
	return s.taskRepo.GetByID(ctx, id)
}

func (s *TaskService) UpdateTaskStatus(ctx context.Context, id int, newStatus TaskStatus, userID UserID) (*Task, error) {
	task, err := s.taskRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	
	// Check permissions
	if task.AssigneeID != userID && task.CreatorID != userID {
		user, err := s.userRepo.GetByID(ctx, userID)
		if err != nil || !user.HasPermission("write:all") {
			return nil, errors.New("insufficient permissions")
		}
	}
	
	if err := task.ChangeStatus(newStatus); err != nil {
		return nil, fmt.Errorf("failed to change status: %w", err)
	}
	
	if err := s.taskRepo.Update(ctx, task); err != nil {
		return nil, fmt.Errorf("failed to update task: %w", err)
	}
	
	s.logger.Printf("Task %d status changed to %s by user %s", task.ID, newStatus, userID)
	return task, nil
}

func (s *TaskService) GetTasksByUser(ctx context.Context, userID UserID) ([]*Task, error) {
	return s.taskRepo.GetByAssignee(ctx, userID)
}

func (s *TaskService) GetOverdueTasks(ctx context.Context) ([]*Task, error) {
	return s.taskRepo.GetOverdue(ctx)
}

func (s *TaskService) GetTaskStatistics(ctx context.Context) (map[string]interface{}, error) {
	totalTasks, err := s.taskRepo.Count(ctx, nil)
	if err != nil {
		return nil, err
	}
	
	completedTasks, err := s.taskRepo.Count(ctx, map[string]interface{}{
		"status": TaskStatusCompleted,
	})
	if err != nil {
		return nil, err
	}
	
	overdueTasks, err := s.taskRepo.GetOverdue(ctx)
	if err != nil {
		return nil, err
	}
	
	completionRate := 0.0
	if totalTasks > 0 {
		completionRate = float64(completedTasks) / float64(totalTasks) * 100
	}
	
	return map[string]interface{}{
		"total_tasks":     totalTasks,
		"completed_tasks": completedTasks,
		"overdue_tasks":   len(overdueTasks),
		"completion_rate": completionRate,
	}, nil
}

// ============ Concurrent Processing ============

type WorkerPool struct {
	workerCount int
	jobQueue    chan Job
	resultQueue chan JobResult
	workers     []*Worker
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
}

type Job struct {
	ID   string
	Data interface{}
}

type JobResult struct {
	JobID string
	Data  interface{}
	Error error
}

type Worker struct {
	id         int
	jobQueue   chan Job
	resultQueue chan JobResult
	quit       chan bool
	processor  func(Job) (interface{}, error)
}

func NewWorkerPool(workerCount int, processor func(Job) (interface{}, error)) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	
	pool := &WorkerPool{
		workerCount: workerCount,
		jobQueue:    make(chan Job, workerCount*2),
		resultQueue: make(chan JobResult, workerCount*2),
		ctx:         ctx,
		cancel:      cancel,
	}
	
	// Create workers
	for i := 0; i < workerCount; i++ {
		worker := &Worker{
			id:          i + 1,
			jobQueue:    pool.jobQueue,
			resultQueue: pool.resultQueue,
			quit:        make(chan bool),
			processor:   processor,
		}
		pool.workers = append(pool.workers, worker)
	}
	
	return pool
}

func (wp *WorkerPool) Start() {
	for _, worker := range wp.workers {
		wp.wg.Add(1)
		go func(w *Worker) {
			defer wp.wg.Done()
			w.Start(wp.ctx)
		}(worker)
	}
}

func (wp *WorkerPool) Stop() {
	wp.cancel()
	wp.wg.Wait()
	close(wp.jobQueue)
	close(wp.resultQueue)
}

func (wp *WorkerPool) SubmitJob(job Job) {
	select {
	case wp.jobQueue <- job:
	case <-wp.ctx.Done():
	}
}

func (wp *WorkerPool) GetResult() <-chan JobResult {
	return wp.resultQueue
}

func (w *Worker) Start(ctx context.Context) {
	for {
		select {
		case job := <-w.jobQueue:
			result, err := w.processor(job)
			w.resultQueue <- JobResult{
				JobID: job.ID,
				Data:  result,
				Error: err,
			}
		case <-ctx.Done():
			return
		case <-w.quit:
			return
		}
	}
}

// ============ Rate Limiting ============

type RateLimiter struct {
	limiter *rate.Limiter
}

func NewRateLimiter(requestsPerSecond int) *RateLimiter {
	return &RateLimiter{
		limiter: rate.NewLimiter(rate.Limit(requestsPerSecond), requestsPerSecond),
	}
}

func (rl *RateLimiter) Allow() bool {
	return rl.limiter.Allow()
}

func (rl *RateLimiter) Wait(ctx context.Context) error {
	return rl.limiter.Wait(ctx)
}

// Middleware for HTTP rate limiting
func (rl *RateLimiter) Middleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !rl.Allow() {
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		next(w, r)
	}
}

// ============ HTTP Handlers ============

type HTTPServer struct {
	userService *UserService
	taskService *TaskService
	rateLimiter *RateLimiter
	logger      *log.Logger
}

func NewHTTPServer(userService *UserService, taskService *TaskService, logger *log.Logger) *HTTPServer {
	return &HTTPServer{
		userService: userService,
		taskService: taskService,
		rateLimiter: NewRateLimiter(100), // 100 requests per second
		logger:      logger,
	}
}

func (s *HTTPServer) SetupRoutes() *mux.Router {
	r := mux.NewRouter()
	
	// Middleware
	r.Use(s.loggingMiddleware)
	r.Use(s.corsMiddleware)
	
	// User routes
	userRouter := r.PathPrefix("/api/users").Subrouter()
	userRouter.HandleFunc("", s.rateLimiter.Middleware(s.createUser)).Methods("POST")
	userRouter.HandleFunc("/{id}", s.rateLimiter.Middleware(s.getUser)).Methods("GET")
	userRouter.HandleFunc("/{id}", s.rateLimiter.Middleware(s.updateUser)).Methods("PUT")
	userRouter.HandleFunc("/{id}", s.rateLimiter.Middleware(s.deleteUser)).Methods("DELETE")
	userRouter.HandleFunc("", s.rateLimiter.Middleware(s.listUsers)).Methods("GET")
	
	// Task routes
	taskRouter := r.PathPrefix("/api/tasks").Subrouter()
	taskRouter.HandleFunc("", s.rateLimiter.Middleware(s.createTask)).Methods("POST")
	taskRouter.HandleFunc("/{id}", s.rateLimiter.Middleware(s.getTask)).Methods("GET")
	taskRouter.HandleFunc("/{id}/status", s.rateLimiter.Middleware(s.updateTaskStatus)).Methods("PUT")
	taskRouter.HandleFunc("", s.rateLimiter.Middleware(s.listTasks)).Methods("GET")
	taskRouter.HandleFunc("/statistics", s.rateLimiter.Middleware(s.getTaskStatistics)).Methods("GET")
	
	// Health check
	r.HandleFunc("/health", s.healthCheck).Methods("GET")
	
	return r
}

func (s *HTTPServer) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		s.logger.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

func (s *HTTPServer) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		next.ServeHTTP(w, r)
	})
}

func (s *HTTPServer) createUser(w http.ResponseWriter, r *http.Request) {
	var req struct {
		FirstName string   `json:"first_name"`
		LastName  string   `json:"last_name"`
		Email     string   `json:"email"`
		Role      UserRole `json:"role"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	user, err := s.userService.CreateUser(r.Context(), req.FirstName, req.LastName, req.Email, req.Role)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

func (s *HTTPServer) getUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := UserID(vars["id"])
	
	user, err := s.userService.GetUser(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

func (s *HTTPServer) updateUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := UserID(vars["id"])
	
	var updates map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updates); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	user, err := s.userService.UpdateUser(r.Context(), userID, updates)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

func (s *HTTPServer) deleteUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := UserID(vars["id"])
	
	if err := s.userService.DeleteUser(r.Context(), userID); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

func (s *HTTPServer) listUsers(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
	
	if limit == 0 {
		limit = 10
	}
	
	filter := make(map[string]interface{})
	if role := r.URL.Query().Get("role"); role != "" {
		filter["role"] = role
	}
	if active := r.URL.Query().Get("active"); active != "" {
		filter["is_active"] = active == "true"
	}
	
	users, err := s.userService.SearchUsers(r.Context(), filter, limit, offset)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"users": users,
		"limit": limit,
		"offset": offset,
	})
}

func (s *HTTPServer) createTask(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Title       string    `json:"title"`
		Description string    `json:"description"`
		Priority    int       `json:"priority"`
		AssigneeID  UserID    `json:"assignee_id"`
		CreatorID   UserID    `json:"creator_id"`
		DueDate     *time.Time `json:"due_date,omitempty"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	task, err := s.taskService.CreateTask(r.Context(), req.Title, req.Description, req.Priority, req.AssigneeID, req.CreatorID, req.DueDate)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(task)
}

func (s *HTTPServer) getTask(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid task ID", http.StatusBadRequest)
		return
	}
	
	task, err := s.taskService.GetTask(r.Context(), taskID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(task)
}

func (s *HTTPServer) updateTaskStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid task ID", http.StatusBadRequest)
		return
	}
	
	var req struct {
		Status TaskStatus `json:"status"`
		UserID UserID     `json:"user_id"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	task, err := s.taskService.UpdateTaskStatus(r.Context(), taskID, req.Status, req.UserID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(task)
}

func (s *HTTPServer) listTasks(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
	
	if limit == 0 {
		limit = 10
	}
	
	// Get tasks by assignee if specified
	if assigneeID := r.URL.Query().Get("assignee_id"); assigneeID != "" {
		tasks, err := s.taskService.GetTasksByUser(r.Context(), UserID(assigneeID))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tasks": tasks,
		})
		return
	}
	
	// Get overdue tasks if specified
	if r.URL.Query().Get("overdue") == "true" {
		tasks, err := s.taskService.GetOverdueTasks(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tasks": tasks,
		})
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Use query parameters to filter tasks",
	})
}

func (s *HTTPServer) getTaskStatistics(w http.ResponseWriter, r *http.Request) {
	stats, err := s.taskService.GetTaskStatistics(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (s *HTTPServer) healthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"version":   "1.0.0",
		"runtime": map[string]interface{}{
			"goroutines": runtime.NumGoroutine(),
			"memory":     getMemUsage(),
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func getMemUsage() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return map[string]interface{}{
		"alloc":      bToMb(m.Alloc),
		"total_alloc": bToMb(m.TotalAlloc),
		"sys":        bToMb(m.Sys),
		"gc_cycles":  m.NumGC,
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

// ============ Demo Application ============

func runDemo() {
	logger := log.New(os.Stdout, "[DEMO] ", log.LstdFlags|log.Lshortfile)
	
	logger.Println("=== Go Microservices Examples Demo ===")
	
	// Setup repositories and services
	userRepo := NewInMemoryUserRepository()
	taskRepo := NewInMemoryTaskRepository()
	userService := NewUserService(userRepo, logger)
	taskService := NewTaskService(taskRepo, userRepo, logger)
	
	ctx := context.Background()
	
	// Create sample users
	logger.Println("Creating sample users...")
	
	admin, err := userService.CreateUser(ctx, "Admin", "User", "admin@example.com", RoleAdmin)
	if err != nil {
		logger.Fatalf("Failed to create admin user: %v", err)
	}
	
	alice, err := userService.CreateUser(ctx, "Alice", "Johnson", "alice@example.com", RoleUser)
	if err != nil {
		logger.Fatalf("Failed to create alice user: %v", err)
	}
	
	bob, err := userService.CreateUser(ctx, "Bob", "Smith", "bob@example.com", RoleUser)
	if err != nil {
		logger.Fatalf("Failed to create bob user: %v", err)
	}
	
	logger.Printf("Created users: %s, %s, %s", admin.ID, alice.ID, bob.ID)
	
	// Create sample tasks
	logger.Println("Creating sample tasks...")
	
	dueDate := time.Now().Add(7 * 24 * time.Hour)
	task1, err := taskService.CreateTask(ctx, "Implement authentication", "Add JWT-based authentication system", 4, alice.ID, admin.ID, &dueDate)
	if err != nil {
		logger.Fatalf("Failed to create task 1: %v", err)
	}
	
	task2, err := taskService.CreateTask(ctx, "Database design", "Design the database schema", 3, bob.ID, admin.ID, nil)
	if err != nil {
		logger.Fatalf("Failed to create task 2: %v", err)
	}
	
	logger.Printf("Created tasks: %d, %d", task1.ID, task2.ID)
	
	// Update task statuses
	logger.Println("Updating task statuses...")
	
	_, err = taskService.UpdateTaskStatus(ctx, task1.ID, TaskStatusInProgress, alice.ID)
	if err != nil {
		logger.Printf("Failed to update task 1 status: %v", err)
	}
	
	_, err = taskService.UpdateTaskStatus(ctx, task2.ID, TaskStatusCompleted, bob.ID)
	if err != nil {
		logger.Printf("Failed to update task 2 status: %v", err)
	}
	
	// Get statistics
	logger.Println("Getting task statistics...")
	
	stats, err := taskService.GetTaskStatistics(ctx)
	if err != nil {
		logger.Printf("Failed to get statistics: %v", err)
	} else {
		logger.Printf("Task statistics: %+v", stats)
	}
	
	// Test concurrent processing
	logger.Println("Testing concurrent processing...")
	
	processor := func(job Job) (interface{}, error) {
		// Simulate work
		time.Sleep(100 * time.Millisecond)
		return fmt.Sprintf("Processed job %s: %v", job.ID, job.Data), nil
	}
	
	pool := NewWorkerPool(3, processor)
	pool.Start()
	
	// Submit jobs
	for i := 0; i < 10; i++ {
		job := Job{
			ID:   fmt.Sprintf("job_%d", i),
			Data: fmt.Sprintf("data_%d", i),
		}
		pool.SubmitJob(job)
	}
	
	// Collect results
	go func() {
		for i := 0; i < 10; i++ {
			result := <-pool.GetResult()
			if result.Error != nil {
				logger.Printf("Job %s failed: %v", result.JobID, result.Error)
			} else {
				logger.Printf("Job result: %s", result.Data)
			}
		}
	}()
	
	// Wait a bit for processing
	time.Sleep(2 * time.Second)
	pool.Stop()
	
	// Test value objects
	logger.Println("Testing value objects...")
	
	money1 := NewMoney(100.50, "USD")
	money2 := NewMoney(25.25, "USD")
	
	total, err := money1.Add(money2)
	if err != nil {
		logger.Printf("Money addition failed: %v", err)
	} else {
		logger.Printf("Money calculation: %s + %s = %s", money1, money2, total)
	}
	
	email, err := NewEmail("test@example.com")
	if err != nil {
		logger.Printf("Email creation failed: %v", err)
	} else {
		logger.Printf("Email domain: %s", email.Domain())
	}
	
	// Test user permissions
	logger.Println("Testing user permissions...")
	logger.Printf("Admin can manage users: %v", admin.HasPermission("manage:users"))
	logger.Printf("Alice can manage users: %v", alice.HasPermission("manage:users"))
	logger.Printf("Alice can read own data: %v", alice.HasPermission("read:own"))
	
	// Start HTTP server (this would normally run indefinitely)
	logger.Println("Setting up HTTP server...")
	
	httpServer := NewHTTPServer(userService, taskService, logger)
	router := httpServer.SetupRoutes()
	
	server := &http.Server{
		Addr:         ":8080",
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	
	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		
		logger.Println("Shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		if err := server.Shutdown(ctx); err != nil {
			logger.Printf("Server shutdown error: %v", err)
		}
	}()
	
	logger.Printf("Server starting on %s", server.Addr)
	logger.Println("Demo completed. Server would normally run here.")
	
	// For demo purposes, we'll just show that the server is ready
	logger.Println("\n=== Go Features Demonstrated ===")
	logger.Println("🚀 Concurrent programming with goroutines and channels")
	logger.Println("🏗️  Interface-based architecture and dependency injection")
	logger.Println("🔒 Type safety with custom types and validation")
	logger.Println("⚡ Worker pools for concurrent task processing")
	logger.Println("🌐 HTTP server with middleware and routing")
	logger.Println("📊 Rate limiting and performance monitoring")
	logger.Println("🛡️  Context-based request handling and cancellation")
	logger.Println("💰 Value objects with business logic")
	logger.Println("🎯 Repository pattern with in-memory implementation")
	logger.Println("📦 Service layer with business logic separation")
	logger.Println("🔧 JSON marshaling/unmarshaling with custom types")
	logger.Println("⚙️  Graceful shutdown and signal handling")
	logger.Println("📈 Memory and performance monitoring")
	logger.Println("🔄 State machines with type-safe transitions")
	logger.Println("🏛️  Clean architecture principles")
}

func main() {
	runDemo()
}