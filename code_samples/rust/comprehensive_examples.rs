// Comprehensive Rust Examples
// Demonstrates ownership, borrowing, traits, generics, error handling, and concurrency

use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Debug};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{self, Read, Write, BufRead, BufReader};
use std::error::Error;
use std::result::Result;

// ========== Structs and Enums ==========

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
    pub age: Option<u8>,
    pub active: bool,
}

impl User {
    pub fn new(id: u64, username: String, email: String) -> Self {
        Self {
            id,
            username,
            email,
            age: None,
            active: true,
        }
    }

    pub fn with_age(mut self, age: u8) -> Self {
        self.age = Some(age);
        self
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn is_adult(&self) -> bool {
        self.age.map_or(false, |age| age >= 18)
    }
}

impl Display for User {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "User(id={}, username={}, email={})", 
               self.id, self.username, self.email)
    }
}

#[derive(Debug, Clone)]
pub struct Product {
    pub id: u64,
    pub name: String,
    pub price: f64,
    pub category: ProductCategory,
    pub in_stock: bool,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductCategory {
    Electronics,
    Books,
    Clothing,
    Home,
    Sports,
    Other(String),
}

impl Display for ProductCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProductCategory::Electronics => write!(f, "Electronics"),
            ProductCategory::Books => write!(f, "Books"),
            ProductCategory::Clothing => write!(f, "Clothing"),
            ProductCategory::Home => write!(f, "Home"),
            ProductCategory::Sports => write!(f, "Sports"),
            ProductCategory::Other(name) => write!(f, "{}", name),
        }
    }
}

// ========== Error Handling ==========

#[derive(Debug)]
pub enum AppError {
    UserNotFound(u64),
    ProductNotFound(u64),
    InvalidInput(String),
    DatabaseError(String),
    IoError(io::Error),
}

impl Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::UserNotFound(id) => write!(f, "User with ID {} not found", id),
            AppError::ProductNotFound(id) => write!(f, "Product with ID {} not found", id),
            AppError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            AppError::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl Error for AppError {}

impl From<io::Error> for AppError {
    fn from(error: io::Error) -> Self {
        AppError::IoError(error)
    }
}

type AppResult<T> = Result<T, AppError>;

// ========== Traits ==========

pub trait Repository<T, K> {
    fn create(&mut self, entity: T) -> AppResult<K>;
    fn read(&self, id: &K) -> AppResult<Option<&T>>;
    fn update(&mut self, id: &K, entity: T) -> AppResult<()>;
    fn delete(&mut self, id: &K) -> AppResult<bool>;
    fn list(&self) -> Vec<&T>;
}

pub trait Validator<T> {
    fn validate(&self, entity: &T) -> AppResult<()>;
}

pub trait Searchable<T> {
    fn search(&self, query: &str) -> Vec<&T>;
}

// ========== Generic Repository Implementation ==========

pub struct InMemoryRepository<T> {
    data: HashMap<u64, T>,
    next_id: u64,
}

impl<T> InMemoryRepository<T> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Repository<T, u64> for InMemoryRepository<T> {
    fn create(&mut self, entity: T) -> AppResult<u64> {
        let id = self.next_id;
        self.data.insert(id, entity);
        self.next_id += 1;
        Ok(id)
    }

    fn read(&self, id: &u64) -> AppResult<Option<&T>> {
        Ok(self.data.get(id))
    }

    fn update(&mut self, id: &u64, entity: T) -> AppResult<()> {
        if self.data.contains_key(id) {
            self.data.insert(*id, entity);
            Ok(())
        } else {
            Err(AppError::InvalidInput(format!("ID {} not found", id)))
        }
    }

    fn delete(&mut self, id: &u64) -> AppResult<bool> {
        Ok(self.data.remove(id).is_some())
    }

    fn list(&self) -> Vec<&T> {
        self.data.values().collect()
    }
}

// ========== Service Layer ==========

pub struct UserService {
    repository: InMemoryRepository<User>,
    validator: UserValidator,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            repository: InMemoryRepository::new(),
            validator: UserValidator::new(),
        }
    }

    pub fn create_user(&mut self, username: String, email: String) -> AppResult<u64> {
        let user = User::new(0, username, email); // ID will be assigned by repository
        self.validator.validate(&user)?;
        
        // Check for duplicate username/email
        for existing_user in self.repository.list() {
            if existing_user.username == user.username {
                return Err(AppError::InvalidInput("Username already exists".to_string()));
            }
            if existing_user.email == user.email {
                return Err(AppError::InvalidInput("Email already exists".to_string()));
            }
        }

        self.repository.create(user)
    }

    pub fn get_user(&self, id: u64) -> AppResult<&User> {
        match self.repository.read(&id)? {
            Some(user) => Ok(user),
            None => Err(AppError::UserNotFound(id)),
        }
    }

    pub fn update_user(&mut self, id: u64, mut user: User) -> AppResult<()> {
        // Ensure the ID matches
        user.id = id;
        self.validator.validate(&user)?;
        self.repository.update(&id, user)
    }

    pub fn delete_user(&mut self, id: u64) -> AppResult<bool> {
        self.repository.delete(&id)
    }

    pub fn list_users(&self) -> Vec<&User> {
        self.repository.list()
    }

    pub fn find_active_users(&self) -> Vec<&User> {
        self.repository.list().into_iter()
            .filter(|user| user.active)
            .collect()
    }

    pub fn find_adult_users(&self) -> Vec<&User> {
        self.repository.list().into_iter()
            .filter(|user| user.is_adult())
            .collect()
    }
}

impl Searchable<User> for UserService {
    fn search(&self, query: &str) -> Vec<&User> {
        let query_lower = query.to_lowercase();
        self.repository.list().into_iter()
            .filter(|user| {
                user.username.to_lowercase().contains(&query_lower) ||
                user.email.to_lowercase().contains(&query_lower)
            })
            .collect()
    }
}

pub struct UserValidator {
    min_username_length: usize,
    max_username_length: usize,
}

impl UserValidator {
    pub fn new() -> Self {
        Self {
            min_username_length: 3,
            max_username_length: 30,
        }
    }

    fn is_valid_email(&self, email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }

    fn is_valid_username(&self, username: &str) -> bool {
        let len = username.len();
        len >= self.min_username_length && 
        len <= self.max_username_length &&
        username.chars().all(|c| c.is_alphanumeric() || c == '_')
    }
}

impl Validator<User> for UserValidator {
    fn validate(&self, user: &User) -> AppResult<()> {
        if !self.is_valid_username(&user.username) {
            return Err(AppError::InvalidInput(
                format!("Invalid username: {}", user.username)
            ));
        }

        if !self.is_valid_email(&user.email) {
            return Err(AppError::InvalidInput(
                format!("Invalid email: {}", user.email)
            ));
        }

        if let Some(age) = user.age {
            if age > 120 {
                return Err(AppError::InvalidInput("Age cannot exceed 120".to_string()));
            }
        }

        Ok(())
    }
}

// ========== Functional Programming Patterns ==========

pub fn process_users<F, R>(users: &[User], processor: F) -> Vec<R>
where
    F: Fn(&User) -> R,
{
    users.iter().map(processor).collect()
}

pub fn filter_users<F>(users: &[User], predicate: F) -> Vec<&User>
where
    F: Fn(&User) -> bool,
{
    users.iter().filter(|user| predicate(user)).collect()
}

pub fn fold_user_ages(users: &[User]) -> u32 {
    users.iter()
        .filter_map(|user| user.age)
        .map(|age| age as u32)
        .fold(0, |acc, age| acc + age)
}

pub fn find_user_by_email<'a>(users: &'a [User], email: &str) -> Option<&'a User> {
    users.iter().find(|user| user.email == email)
}

// ========== Concurrency Examples ==========

pub struct ThreadSafeCounter {
    value: Arc<Mutex<i32>>,
}

impl ThreadSafeCounter {
    pub fn new(initial_value: i32) -> Self {
        Self {
            value: Arc::new(Mutex::new(initial_value)),
        }
    }

    pub fn increment(&self) -> AppResult<i32> {
        let mut value = self.value.lock()
            .map_err(|_| AppError::DatabaseError("Failed to acquire lock".to_string()))?;
        *value += 1;
        Ok(*value)
    }

    pub fn get(&self) -> AppResult<i32> {
        let value = self.value.lock()
            .map_err(|_| AppError::DatabaseError("Failed to acquire lock".to_string()))?;
        Ok(*value)
    }
}

impl Clone for ThreadSafeCounter {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
        }
    }
}

pub struct SharedUserService {
    users: Arc<RwLock<Vec<User>>>,
    next_id: Arc<Mutex<u64>>,
}

impl SharedUserService {
    pub fn new() -> Self {
        Self {
            users: Arc::new(RwLock::new(Vec::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }

    pub fn add_user(&self, username: String, email: String) -> AppResult<u64> {
        let id = {
            let mut next_id = self.next_id.lock()
                .map_err(|_| AppError::DatabaseError("Failed to acquire ID lock".to_string()))?;
            let id = *next_id;
            *next_id += 1;
            id
        };

        let user = User::new(id, username, email);
        
        let mut users = self.users.write()
            .map_err(|_| AppError::DatabaseError("Failed to acquire write lock".to_string()))?;
        users.push(user);
        
        Ok(id)
    }

    pub fn get_user_count(&self) -> AppResult<usize> {
        let users = self.users.read()
            .map_err(|_| AppError::DatabaseError("Failed to acquire read lock".to_string()))?;
        Ok(users.len())
    }

    pub fn find_user(&self, id: u64) -> AppResult<Option<User>> {
        let users = self.users.read()
            .map_err(|_| AppError::DatabaseError("Failed to acquire read lock".to_string()))?;
        Ok(users.iter().find(|u| u.id == id).cloned())
    }
}

impl Clone for SharedUserService {
    fn clone(&self) -> Self {
        Self {
            users: Arc::clone(&self.users),
            next_id: Arc::clone(&self.next_id),
        }
    }
}

// ========== File I/O and Serialization ==========

pub fn save_users_to_file(users: &[User], filename: &str) -> AppResult<()> {
    let mut file = File::create(filename)?;
    
    for user in users {
        let line = format!("{}|{}|{}|{}|{}\n", 
                          user.id, user.username, user.email, 
                          user.age.map_or("None".to_string(), |a| a.to_string()),
                          user.active);
        file.write_all(line.as_bytes())?;
    }
    
    Ok(())
}

pub fn load_users_from_file(filename: &str) -> AppResult<Vec<User>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut users = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('|').collect();
        
        if parts.len() != 5 {
            return Err(AppError::InvalidInput(
                "Invalid file format".to_string()
            ));
        }
        
        let id: u64 = parts[0].parse()
            .map_err(|_| AppError::InvalidInput("Invalid ID".to_string()))?;
        let username = parts[1].to_string();
        let email = parts[2].to_string();
        let age = if parts[3] == "None" {
            None
        } else {
            Some(parts[3].parse()
                .map_err(|_| AppError::InvalidInput("Invalid age".to_string()))?)
        };
        let active: bool = parts[4].parse()
            .map_err(|_| AppError::InvalidInput("Invalid active flag".to_string()))?;
        
        let mut user = User::new(id, username, email);
        user.age = age;
        user.active = active;
        users.push(user);
    }
    
    Ok(users)
}

// ========== Iterator Patterns ==========

pub struct UserIterator<'a> {
    users: &'a [User],
    index: usize,
}

impl<'a> UserIterator<'a> {
    pub fn new(users: &'a [User]) -> Self {
        Self { users, index: 0 }
    }
}

impl<'a> Iterator for UserIterator<'a> {
    type Item = &'a User;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.users.len() {
            let user = &self.users[self.index];
            self.index += 1;
            Some(user)
        } else {
            None
        }
    }
}

pub fn fibonacci() -> impl Iterator<Item = u64> {
    let mut a = 0;
    let mut b = 1;
    std::iter::from_fn(move || {
        let next = a + b;
        a = b;
        b = next;
        Some(a)
    })
}

// ========== Pattern Matching ==========

pub fn categorize_user(user: &User) -> String {
    match (user.age, user.active) {
        (Some(age), true) if age >= 18 => "Active Adult".to_string(),
        (Some(age), true) if age < 18 => "Active Minor".to_string(),
        (Some(_), false) => "Inactive User".to_string(),
        (None, true) => "Active User (Unknown Age)".to_string(),
        (None, false) => "Inactive User (Unknown Age)".to_string(),
    }
}

pub fn process_result<T, E>(result: Result<T, E>) -> String
where
    T: Display,
    E: Display,
{
    match result {
        Ok(value) => format!("Success: {}", value),
        Err(error) => format!("Error: {}", error),
    }
}

// ========== Demonstration Functions ==========

pub fn demonstrate_basic_operations() -> AppResult<()> {
    println!("=== Basic Operations Demo ===");
    
    let mut user_service = UserService::new();
    
    // Create users
    let user1_id = user_service.create_user("alice".to_string(), "alice@example.com".to_string())?;
    let user2_id = user_service.create_user("bob".to_string(), "bob@example.com".to_string())?;
    
    println!("Created user 1 with ID: {}", user1_id);
    println!("Created user 2 with ID: {}", user2_id);
    
    // Get users
    let user1 = user_service.get_user(user1_id)?;
    println!("Retrieved user: {}", user1);
    
    // Update user
    let mut updated_user = user1.clone();
    updated_user.age = Some(25);
    user_service.update_user(user1_id, updated_user)?;
    
    let updated_user = user_service.get_user(user1_id)?;
    println!("Updated user: {} (Age: {:?})", updated_user, updated_user.age);
    
    // List users
    let all_users = user_service.list_users();
    println!("All users: {}", all_users.len());
    
    // Search users
    let search_results = user_service.search("alice");
    println!("Search results for 'alice': {}", search_results.len());
    
    Ok(())
}

pub fn demonstrate_functional_programming() -> AppResult<()> {
    println!("\n=== Functional Programming Demo ===");
    
    let users = vec![
        User::new(1, "alice".to_string(), "alice@example.com".to_string()).with_age(25),
        User::new(2, "bob".to_string(), "bob@example.com".to_string()).with_age(17),
        User::new(3, "charlie".to_string(), "charlie@example.com".to_string()).with_age(30),
    ];
    
    // Map operation
    let usernames: Vec<String> = process_users(&users, |user| user.username.clone());
    println!("Usernames: {:?}", usernames);
    
    // Filter operation
    let adults = filter_users(&users, |user| user.is_adult());
    println!("Adult users: {}", adults.len());
    
    // Fold operation
    let total_age = fold_user_ages(&users);
    println!("Total age of all users: {}", total_age);
    
    // Find operation
    if let Some(user) = find_user_by_email(&users, "bob@example.com") {
        println!("Found user by email: {}", user);
    }
    
    // Iterator patterns
    println!("User categorization:");
    for user in UserIterator::new(&users) {
        println!("  {}: {}", user.username, categorize_user(user));
    }
    
    // Fibonacci sequence
    let fib_numbers: Vec<u64> = fibonacci().take(10).collect();
    println!("First 10 Fibonacci numbers: {:?}", fib_numbers);
    
    Ok(())
}

pub fn demonstrate_concurrency() -> AppResult<()> {
    println!("\n=== Concurrency Demo ===");
    
    let counter = ThreadSafeCounter::new(0);
    let mut handles = vec![];
    
    // Spawn multiple threads to increment counter
    for i in 0..5 {
        let counter_clone = counter.clone();
        let handle = thread::spawn(move || {
            for _ in 0..10 {
                let value = counter_clone.increment().unwrap();
                println!("Thread {}: Counter value = {}", i, value);
                thread::sleep(Duration::from_millis(10));
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final counter value: {}", counter.get()?);
    
    // Shared user service
    let shared_service = SharedUserService::new();
    let mut handles = vec![];
    
    for i in 0..3 {
        let service_clone = shared_service.clone();
        let handle = thread::spawn(move || {
            for j in 0..3 {
                let username = format!("user{}_{}", i, j);
                let email = format!("user{}_{}@example.com", i, j);
                match service_clone.add_user(username, email) {
                    Ok(id) => println!("Thread {}: Created user with ID {}", i, id),
                    Err(e) => println!("Thread {}: Error creating user: {}", i, e),
                }
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Total users in shared service: {}", shared_service.get_user_count()?);
    
    Ok(())
}

pub fn demonstrate_file_operations() -> AppResult<()> {
    println!("\n=== File Operations Demo ===");
    
    let users = vec![
        User::new(1, "alice".to_string(), "alice@example.com".to_string()).with_age(25),
        User::new(2, "bob".to_string(), "bob@example.com".to_string()).with_age(30),
        User::new(3, "charlie".to_string(), "charlie@example.com".to_string()),
    ];
    
    let filename = "/tmp/users.txt";
    
    // Save users to file
    save_users_to_file(&users, filename)?;
    println!("Saved {} users to {}", users.len(), filename);
    
    // Load users from file
    let loaded_users = load_users_from_file(filename)?;
    println!("Loaded {} users from {}", loaded_users.len(), filename);
    
    // Verify data integrity
    for (original, loaded) in users.iter().zip(loaded_users.iter()) {
        if original.id == loaded.id && original.username == loaded.username {
            println!("✓ User {} data integrity verified", original.username);
        } else {
            println!("✗ User data integrity failed");
        }
    }
    
    Ok(())
}

pub fn demonstrate_error_handling() -> AppResult<()> {
    println!("\n=== Error Handling Demo ===");
    
    let mut user_service = UserService::new();
    
    // Test various error scenarios
    let test_cases = vec![
        ("", "invalid@email.com"),           // Invalid username
        ("validuser", "invalidemail"),       // Invalid email
        ("ab", "short@email.com"),           // Username too short
        ("a".repeat(50).as_str(), "long@email.com"), // Username too long
    ];
    
    for (username, email) in test_cases {
        let result = user_service.create_user(username.to_string(), email.to_string());
        println!("Create user '{}': {}", username, process_result(result));
    }
    
    // Test user not found error
    let result = user_service.get_user(999);
    println!("Get non-existent user: {}", process_result(result));
    
    // Test file operation error
    let result = load_users_from_file("/nonexistent/file.txt");
    println!("Load from non-existent file: {}", process_result(result));
    
    Ok(())
}

pub fn main() -> AppResult<()> {
    println!("=== Comprehensive Rust Examples ===\n");
    
    demonstrate_basic_operations()?;
    demonstrate_functional_programming()?;
    demonstrate_concurrency()?;
    demonstrate_file_operations()?;
    demonstrate_error_handling()?;
    
    println!("\n=== Rust Features Demonstrated ===");
    println!("- Ownership and borrowing");
    println!("- Traits and generics");
    println!("- Pattern matching");
    println!("- Error handling with Result<T, E>");
    println!("- Concurrency with threads and Arc/Mutex");
    println!("- Iterator patterns");
    println!("- File I/O operations");
    println!("- Functional programming patterns");
    println!("- Memory safety without garbage collection");
    println!("- Zero-cost abstractions");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_creation() {
        let user = User::new(1, "testuser".to_string(), "test@example.com".to_string());
        assert_eq!(user.id, 1);
        assert_eq!(user.username, "testuser");
        assert_eq!(user.email, "test@example.com");
        assert!(user.active);
        assert_eq!(user.age, None);
    }

    #[test]
    fn test_user_service() {
        let mut service = UserService::new();
        let user_id = service.create_user("alice".to_string(), "alice@example.com".to_string()).unwrap();
        
        let user = service.get_user(user_id).unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.email, "alice@example.com");
    }

    #[test]
    fn test_user_validation() {
        let validator = UserValidator::new();
        
        let valid_user = User::new(1, "validuser".to_string(), "valid@example.com".to_string());
        assert!(validator.validate(&valid_user).is_ok());
        
        let invalid_user = User::new(1, "ab".to_string(), "invalid".to_string());
        assert!(validator.validate(&invalid_user).is_err());
    }

    #[test]
    fn test_thread_safe_counter() {
        let counter = ThreadSafeCounter::new(0);
        assert_eq!(counter.get().unwrap(), 0);
        
        assert_eq!(counter.increment().unwrap(), 1);
        assert_eq!(counter.increment().unwrap(), 2);
        assert_eq!(counter.get().unwrap(), 2);
    }

    #[test]
    fn test_fibonacci() {
        let fib_numbers: Vec<u64> = fibonacci().take(5).collect();
        assert_eq!(fib_numbers, vec![1, 1, 2, 3, 5]);
    }
}