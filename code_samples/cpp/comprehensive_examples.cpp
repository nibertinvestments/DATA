/**
 * Comprehensive C++ Examples
 * Demonstrates modern C++ features, STL, templates, RAII, and advanced patterns
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <stack>
#include <optional>
#include <variant>
#include <array>
#include <type_traits>
#include <random>
#include <numeric>
#include <regex>

// ========== Modern C++ Classes and RAII ==========

/**
 * User class demonstrating modern C++ features
 */
class User {
private:
    std::string username_;
    std::string email_;
    std::optional<int> age_;
    bool active_;
    std::chrono::system_clock::time_point created_at_;
    std::vector<std::string> roles_;

public:
    // Constructor with member initializer list
    User(std::string username, std::string email)
        : username_(std::move(username))
        , email_(std::move(email))
        , active_(true)
        , created_at_(std::chrono::system_clock::now()) {
        validate();
    }

    // Constructor delegation
    User(std::string username, std::string email, int age)
        : User(std::move(username), std::move(email)) {
        age_ = age;
    }

    // Rule of 5 - move semantics
    User(const User& other) = default;
    User& operator=(const User& other) = default;
    User(User&& other) noexcept = default;
    User& operator=(User&& other) noexcept = default;
    ~User() = default;

    // Getters with const correctness
    const std::string& username() const noexcept { return username_; }
    const std::string& email() const noexcept { return email_; }
    std::optional<int> age() const noexcept { return age_; }
    bool active() const noexcept { return active_; }
    const auto& created_at() const noexcept { return created_at_; }
    const std::vector<std::string>& roles() const noexcept { return roles_; }

    // Setters with validation
    void set_username(const std::string& username) {
        username_ = username;
        validate();
    }

    void set_age(int age) {
        if (age < 0 || age > 150) {
            throw std::invalid_argument("Age must be between 0 and 150");
        }
        age_ = age;
    }

    void set_active(bool active) noexcept { active_ = active; }

    void add_role(std::string role) {
        roles_.emplace_back(std::move(role));
    }

    // Business logic methods
    bool is_adult() const noexcept {
        return age_.has_value() && age_.value() >= 18;
    }

    bool has_role(const std::string& role) const {
        return std::find(roles_.begin(), roles_.end(), role) != roles_.end();
    }

    void deactivate() noexcept {
        active_ = false;
    }

    // Comparison operators
    bool operator==(const User& other) const noexcept {
        return username_ == other.username_ && email_ == other.email_;
    }

    bool operator!=(const User& other) const noexcept {
        return !(*this == other);
    }

    bool operator<(const User& other) const noexcept {
        return username_ < other.username_;
    }

    // Stream output operator
    friend std::ostream& operator<<(std::ostream& os, const User& user) {
        os << "User{username: " << user.username_
           << ", email: " << user.email_
           << ", age: " << (user.age_ ? std::to_string(*user.age_) : "N/A")
           << ", active: " << std::boolalpha << user.active_ << "}";
        return os;
    }

private:
    void validate() const {
        if (username_.empty()) {
            throw std::invalid_argument("Username cannot be empty");
        }
        if (email_.find('@') == std::string::npos) {
            throw std::invalid_argument("Invalid email format");
        }
    }
};

// ========== Template Classes and Generic Programming ==========

/**
 * Generic Repository template class
 */
template<typename T, typename KeyType = std::size_t>
class Repository {
private:
    std::unordered_map<KeyType, std::unique_ptr<T>> storage_;
    KeyType next_id_;
    mutable std::shared_mutex mutex_;

public:
    Repository() : next_id_(1) {}

    // Copy constructor deleted (non-copyable due to unique_ptr)
    Repository(const Repository&) = delete;
    Repository& operator=(const Repository&) = delete;

    // Move constructor
    Repository(Repository&&) noexcept = default;
    Repository& operator=(Repository&&) noexcept = default;

    // Create entity with perfect forwarding
    template<typename... Args>
    KeyType create(Args&&... args) {
        std::unique_lock lock(mutex_);
        auto id = next_id_++;
        storage_[id] = std::make_unique<T>(std::forward<Args>(args)...);
        return id;
    }

    // Find by ID
    std::optional<std::reference_wrapper<const T>> find_by_id(const KeyType& id) const {
        std::shared_lock lock(mutex_);
        auto it = storage_.find(id);
        if (it != storage_.end()) {
            return std::cref(*it->second);
        }
        return std::nullopt;
    }

    // Update entity
    bool update(const KeyType& id, const T& entity) {
        std::unique_lock lock(mutex_);
        auto it = storage_.find(id);
        if (it != storage_.end()) {
            *it->second = entity;
            return true;
        }
        return false;
    }

    // Delete entity
    bool remove(const KeyType& id) {
        std::unique_lock lock(mutex_);
        return storage_.erase(id) > 0;
    }

    // Get all entities
    std::vector<std::reference_wrapper<const T>> find_all() const {
        std::shared_lock lock(mutex_);
        std::vector<std::reference_wrapper<const T>> result;
        result.reserve(storage_.size());
        
        for (const auto& [id, entity] : storage_) {
            result.emplace_back(*entity);
        }
        
        return result;
    }

    // Find with predicate
    template<typename Predicate>
    std::vector<std::reference_wrapper<const T>> find_if(Predicate&& pred) const {
        std::shared_lock lock(mutex_);
        std::vector<std::reference_wrapper<const T>> result;
        
        for (const auto& [id, entity] : storage_) {
            if (pred(*entity)) {
                result.emplace_back(*entity);
            }
        }
        
        return result;
    }

    // Count entities
    std::size_t count() const {
        std::shared_lock lock(mutex_);
        return storage_.size();
    }

    // Check if empty
    bool empty() const {
        std::shared_lock lock(mutex_);
        return storage_.empty();
    }
};

// ========== Template Specialization and Concepts ==========

/**
 * Generic algorithm implementations
 */
namespace algorithms {

// Concept for comparable types (C++20)
template<typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a == b } -> std::convertible_to<bool>;
};

// Template function with concept
template<Comparable T>
std::vector<T> merge_sorted(const std::vector<T>& left, const std::vector<T>& right) {
    std::vector<T> result;
    result.reserve(left.size() + right.size());
    
    auto left_it = left.begin();
    auto right_it = right.begin();
    
    while (left_it != left.end() && right_it != right.end()) {
        if (*left_it <= *right_it) {
            result.push_back(*left_it++);
        } else {
            result.push_back(*right_it++);
        }
    }
    
    result.insert(result.end(), left_it, left.end());
    result.insert(result.end(), right_it, right.end());
    
    return result;
}

// Template function for functional programming
template<typename Container, typename Predicate>
auto filter(const Container& container, Predicate&& pred) {
    Container result;
    std::copy_if(container.begin(), container.end(), 
                 std::back_inserter(result), std::forward<Predicate>(pred));
    return result;
}

template<typename Container, typename Transform>
auto transform(const Container& container, Transform&& func) {
    using ValueType = std::invoke_result_t<Transform, typename Container::value_type>;
    std::vector<ValueType> result;
    result.reserve(container.size());
    
    std::transform(container.begin(), container.end(), 
                   std::back_inserter(result), std::forward<Transform>(func));
    return result;
}

template<typename Container, typename T, typename BinaryOp>
T reduce(const Container& container, T init, BinaryOp&& op) {
    return std::accumulate(container.begin(), container.end(), 
                          std::move(init), std::forward<BinaryOp>(op));
}

} // namespace algorithms

// ========== RAII and Resource Management ==========

/**
 * File handler with RAII
 */
class FileHandler {
private:
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file_;

public:
    explicit FileHandler(const std::string& filename, const std::string& mode)
        : file_(std::fopen(filename.c_str(), mode.c_str()), &std::fclose) {
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

    // Non-copyable, movable
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    FileHandler(FileHandler&&) = default;
    FileHandler& operator=(FileHandler&&) = default;

    std::string read_all() {
        if (!file_) return "";
        
        std::fseek(file_.get(), 0, SEEK_END);
        long size = std::ftell(file_.get());
        std::fseek(file_.get(), 0, SEEK_SET);
        
        std::string content(size, '\0');
        std::fread(&content[0], 1, size, file_.get());
        
        return content;
    }

    void write(const std::string& content) {
        if (!file_) {
            throw std::runtime_error("File not open for writing");
        }
        std::fwrite(content.c_str(), 1, content.size(), file_.get());
        std::fflush(file_.get());
    }

    bool is_open() const noexcept {
        return file_ != nullptr;
    }
};

/**
 * Timer class for performance measurement
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string name_;

public:
    explicit Timer(std::string name) 
        : start_time_(std::chrono::high_resolution_clock::now())
        , name_(std::move(name)) {}

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_).count();
        std::cout << name_ << " took " << duration << " microseconds\n";
    }

    // Get elapsed time without destroying timer
    std::chrono::microseconds elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
    }
};

// ========== Concurrency and Threading ==========

/**
 * Thread-safe counter
 */
class ThreadSafeCounter {
private:
    mutable std::mutex mutex_;
    int value_;

public:
    explicit ThreadSafeCounter(int initial_value = 0) : value_(initial_value) {}

    void increment() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++value_;
    }

    void decrement() {
        std::lock_guard<std::mutex> lock(mutex_);
        --value_;
    }

    int get() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }

    void add(int delta) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ += delta;
    }
};

/**
 * Thread pool implementation
 */
class ThreadPool {
private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;

public:
    explicit ThreadPool(std::size_t num_threads) : stop_(false) {
        for (std::size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            if (stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            
            tasks_.emplace([task] { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        
        condition_.notify_all();
        
        for (std::thread& worker : threads_) {
            worker.join();
        }
    }
};

/**
 * Producer-Consumer pattern
 */
template<typename T>
class ProducerConsumer {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool finished_;

public:
    ProducerConsumer() : finished_(false) {}

    void produce(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        condition_.notify_one();
    }

    std::optional<T> consume() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || finished_; });
        
        if (queue_.empty()) {
            return std::nullopt;
        }
        
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// ========== Design Patterns ==========

/**
 * Singleton pattern with thread safety
 */
class DatabaseConnection {
private:
    static std::unique_ptr<DatabaseConnection> instance_;
    static std::once_flag init_flag_;
    
    std::string connection_string_;
    bool connected_;

    DatabaseConnection() : connected_(false) {}

public:
    static DatabaseConnection& getInstance() {
        std::call_once(init_flag_, []() {
            instance_ = std::unique_ptr<DatabaseConnection>(new DatabaseConnection());
        });
        return *instance_;
    }

    void connect(const std::string& connection_string) {
        connection_string_ = connection_string;
        connected_ = true;
        std::cout << "Connected to database: " << connection_string << "\n";
    }

    void disconnect() {
        connected_ = false;
        std::cout << "Disconnected from database\n";
    }

    bool is_connected() const { return connected_; }
    const std::string& get_connection_string() const { return connection_string_; }

    // Non-copyable, non-movable
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;
    DatabaseConnection(DatabaseConnection&&) = delete;
    DatabaseConnection& operator=(DatabaseConnection&&) = delete;
};

// Static member definitions
std::unique_ptr<DatabaseConnection> DatabaseConnection::instance_ = nullptr;
std::once_flag DatabaseConnection::init_flag_;

/**
 * Observer pattern
 */
template<typename EventType>
class Observable {
public:
    using Observer = std::function<void(const EventType&)>;

private:
    std::vector<Observer> observers_;
    mutable std::shared_mutex mutex_;

public:
    void subscribe(Observer observer) {
        std::unique_lock lock(mutex_);
        observers_.emplace_back(std::move(observer));
    }

    void notify(const EventType& event) {
        std::shared_lock lock(mutex_);
        for (const auto& observer : observers_) {
            observer(event);
        }
    }

    void clear_observers() {
        std::unique_lock lock(mutex_);
        observers_.clear();
    }

    std::size_t observer_count() const {
        std::shared_lock lock(mutex_);
        return observers_.size();
    }
};

// ========== Smart Pointers and Memory Management ==========

/**
 * Custom smart pointer implementation for educational purposes
 */
template<typename T>
class unique_ptr_impl {
private:
    T* ptr_;

public:
    explicit unique_ptr_impl(T* ptr = nullptr) : ptr_(ptr) {}

    ~unique_ptr_impl() {
        delete ptr_;
    }

    // Move constructor
    unique_ptr_impl(unique_ptr_impl&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    // Move assignment
    unique_ptr_impl& operator=(unique_ptr_impl&& other) noexcept {
        if (this != &other) {
            delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Delete copy constructor and assignment
    unique_ptr_impl(const unique_ptr_impl&) = delete;
    unique_ptr_impl& operator=(const unique_ptr_impl&) = delete;

    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }

    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

    void reset(T* new_ptr = nullptr) {
        delete ptr_;
        ptr_ = new_ptr;
    }

    explicit operator bool() const { return ptr_ != nullptr; }
};

// ========== STL Algorithms and Data Structures ==========

/**
 * Custom data structure: Binary Search Tree
 */
template<typename T>
class BinarySearchTree {
private:
    struct Node {
        T data;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        
        explicit Node(T value) : data(std::move(value)) {}
    };

    std::unique_ptr<Node> root_;

    void insert_recursive(std::unique_ptr<Node>& node, T value) {
        if (!node) {
            node = std::make_unique<Node>(std::move(value));
            return;
        }

        if (value < node->data) {
            insert_recursive(node->left, std::move(value));
        } else if (value > node->data) {
            insert_recursive(node->right, std::move(value));
        }
    }

    bool search_recursive(const std::unique_ptr<Node>& node, const T& value) const {
        if (!node) return false;
        if (node->data == value) return true;
        if (value < node->data) return search_recursive(node->left, value);
        return search_recursive(node->right, value);
    }

    void inorder_recursive(const std::unique_ptr<Node>& node, std::vector<T>& result) const {
        if (!node) return;
        inorder_recursive(node->left, result);
        result.push_back(node->data);
        inorder_recursive(node->right, result);
    }

public:
    void insert(T value) {
        insert_recursive(root_, std::move(value));
    }

    bool search(const T& value) const {
        return search_recursive(root_, value);
    }

    std::vector<T> inorder_traversal() const {
        std::vector<T> result;
        inorder_recursive(root_, result);
        return result;
    }

    bool empty() const { return !root_; }
};

// ========== Exception Handling ==========

/**
 * Custom exception hierarchy
 */
class AppException : public std::exception {
protected:
    std::string message_;

public:
    explicit AppException(std::string message) : message_(std::move(message)) {}
    const char* what() const noexcept override { return message_.c_str(); }
};

class ValidationException : public AppException {
public:
    explicit ValidationException(const std::string& field, const std::string& value)
        : AppException("Validation failed for field '" + field + "' with value '" + value + "'") {}
};

class NotFoundException : public AppException {
public:
    explicit NotFoundException(const std::string& resource, const std::string& identifier)
        : AppException(resource + " not found: " + identifier) {}
};

// ========== Demonstration Functions ==========

void demonstrate_basic_features() {
    std::cout << "=== Basic C++ Features Demo ===\n";

    // Modern initialization
    User user1{"alice", "alice@example.com"};
    User user2{"bob", "bob@example.com", 25};

    std::cout << "Created users:\n";
    std::cout << "- " << user1 << "\n";
    std::cout << "- " << user2 << "\n";

    // Optional and variants
    std::optional<int> maybe_age = user1.age();
    if (maybe_age) {
        std::cout << "User1 age: " << *maybe_age << "\n";
    } else {
        std::cout << "User1 age not set\n";
    }

    // STL containers
    std::vector<User> users{user1, user2};
    std::sort(users.begin(), users.end());

    std::cout << "Sorted users:\n";
    for (const auto& user : users) {
        std::cout << "- " << user.username() << "\n";
    }

    std::cout << "\n";
}

void demonstrate_templates_and_generics() {
    std::cout << "=== Templates and Generics Demo ===\n";

    // Repository pattern
    Repository<User> user_repo;

    auto id1 = user_repo.create("charlie", "charlie@example.com");
    auto id2 = user_repo.create("diana", "diana@example.com", 30);

    std::cout << "Created users with IDs: " << id1 << ", " << id2 << "\n";

    // Find operations
    auto found_user = user_repo.find_by_id(id1);
    if (found_user) {
        std::cout << "Found user: " << found_user->get() << "\n";
    }

    // Algorithms
    std::vector<int> numbers{5, 2, 8, 1, 9, 3};
    auto sorted_left = std::vector<int>{1, 3, 5};
    auto sorted_right = std::vector<int>{2, 8, 9};
    
    auto merged = algorithms::merge_sorted(sorted_left, sorted_right);
    std::cout << "Merged sorted arrays: ";
    for (int n : merged) std::cout << n << " ";
    std::cout << "\n";

    // Functional programming
    auto even_numbers = algorithms::filter(numbers, [](int n) { return n % 2 == 0; });
    auto doubled = algorithms::transform(numbers, [](int n) { return n * 2; });
    auto sum = algorithms::reduce(numbers, 0, std::plus<int>{});

    std::cout << "Even numbers: ";
    for (int n : even_numbers) std::cout << n << " ";
    std::cout << "\n";

    std::cout << "Sum of all numbers: " << sum << "\n";

    std::cout << "\n";
}

void demonstrate_concurrency() {
    std::cout << "=== Concurrency Demo ===\n";

    // Thread-safe counter
    ThreadSafeCounter counter;
    std::vector<std::thread> threads;

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&counter, i]() {
            for (int j = 0; j < 10; ++j) {
                counter.increment();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "Thread " << i << " finished\n";
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "Final counter value: " << counter.get() << "\n";

    // Thread pool
    {
        ThreadPool pool(4);
        std::vector<std::future<int>> results;

        for (int i = 0; i < 8; ++i) {
            results.emplace_back(
                pool.enqueue([i] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    return i * i;
                })
            );
        }

        std::cout << "Thread pool results: ";
        for (auto& result : results) {
            std::cout << result.get() << " ";
        }
        std::cout << "\n";
    }

    // Producer-Consumer
    ProducerConsumer<int> pc;
    
    std::thread producer([&pc]() {
        for (int i = 0; i < 10; ++i) {
            pc.produce(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        pc.finish();
    });

    std::thread consumer([&pc]() {
        while (auto item = pc.consume()) {
            std::cout << "Consumed: " << *item << "\n";
        }
    });

    producer.join();
    consumer.join();

    std::cout << "\n";
}

void demonstrate_smart_pointers() {
    std::cout << "=== Smart Pointers Demo ===\n";

    // Standard smart pointers
    auto unique_user = std::make_unique<User>("unique_user", "unique@example.com");
    std::cout << "Unique user: " << *unique_user << "\n";

    auto shared_user1 = std::make_shared<User>("shared_user", "shared@example.com");
    auto shared_user2 = shared_user1; // Shared ownership
    
    std::cout << "Shared user reference count: " << shared_user1.use_count() << "\n";
    std::cout << "Shared user: " << *shared_user1 << "\n";

    // Custom smart pointer
    unique_ptr_impl<User> custom_ptr(new User("custom", "custom@example.com"));
    std::cout << "Custom smart pointer: " << *custom_ptr << "\n";

    // Memory management with containers
    std::vector<std::unique_ptr<User>> user_collection;
    user_collection.push_back(std::make_unique<User>("vec_user1", "vec1@example.com"));
    user_collection.push_back(std::make_unique<User>("vec_user2", "vec2@example.com"));

    std::cout << "Users in collection: " << user_collection.size() << "\n";

    std::cout << "\n";
}

void demonstrate_data_structures() {
    std::cout << "=== Data Structures Demo ===\n";

    // Binary Search Tree
    BinarySearchTree<int> bst;
    std::vector<int> values{5, 3, 7, 1, 9, 4, 6};

    for (int value : values) {
        bst.insert(value);
    }

    std::cout << "BST inorder traversal: ";
    auto traversal = bst.inorder_traversal();
    for (int value : traversal) {
        std::cout << value << " ";
    }
    std::cout << "\n";

    std::cout << "Search for 7: " << (bst.search(7) ? "found" : "not found") << "\n";
    std::cout << "Search for 10: " << (bst.search(10) ? "found" : "not found") << "\n";

    // STL containers showcase
    std::map<std::string, int> word_count;
    std::string text = "hello world hello cpp world";
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        word_count[word]++;
    }

    std::cout << "Word frequencies:\n";
    for (const auto& [word, count] : word_count) {
        std::cout << "- " << word << ": " << count << "\n";
    }

    std::cout << "\n";
}

void demonstrate_file_operations() {
    std::cout << "=== File Operations Demo ===\n";

    try {
        // RAII file handling
        {
            FileHandler writer("/tmp/cpp_demo.txt", "w");
            writer.write("Hello from C++!\nThis is a test file.\n");
        } // File automatically closed here

        {
            FileHandler reader("/tmp/cpp_demo.txt", "r");
            std::string content = reader.read_all();
            std::cout << "File content:\n" << content << "\n";
        }

        // C++ streams
        std::ofstream outfile("/tmp/users.txt");
        std::vector<User> users{
            User{"stream_user1", "stream1@example.com"},
            User{"stream_user2", "stream2@example.com"}
        };

        for (const auto& user : users) {
            outfile << user << "\n";
        }
        outfile.close();

        std::ifstream infile("/tmp/users.txt");
        std::string line;
        std::cout << "Users from file:\n";
        while (std::getline(infile, line)) {
            std::cout << "- " << line << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "File operation error: " << e.what() << "\n";
    }

    std::cout << "\n";
}

void demonstrate_design_patterns() {
    std::cout << "=== Design Patterns Demo ===\n";

    // Singleton pattern
    auto& db1 = DatabaseConnection::getInstance();
    auto& db2 = DatabaseConnection::getInstance();
    
    std::cout << "Same instance: " << (&db1 == &db2 ? "yes" : "no") << "\n";
    
    db1.connect("postgresql://localhost:5432/mydb");
    std::cout << "Connection status: " << (db2.is_connected() ? "connected" : "disconnected") << "\n";

    // Observer pattern
    Observable<std::string> event_system;
    
    event_system.subscribe([](const std::string& event) {
        std::cout << "Observer 1 received: " << event << "\n";
    });
    
    event_system.subscribe([](const std::string& event) {
        std::cout << "Observer 2 received: " << event << "\n";
    });

    event_system.notify("System started");
    event_system.notify("User logged in");

    std::cout << "Observer count: " << event_system.observer_count() << "\n";

    std::cout << "\n";
}

void demonstrate_performance_timing() {
    std::cout << "=== Performance Timing Demo ===\n";

    {
        Timer timer("Vector operations");
        std::vector<int> large_vector(1000000);
        std::iota(large_vector.begin(), large_vector.end(), 1);
        
        auto sum = std::accumulate(large_vector.begin(), large_vector.end(), 0LL);
        std::cout << "Sum of 1M numbers: " << sum << "\n";
    }

    {
        Timer timer("String operations");
        std::string result;
        for (int i = 0; i < 10000; ++i) {
            result += "test" + std::to_string(i) + " ";
        }
        std::cout << "Generated string length: " << result.length() << "\n";
    }

    std::cout << "\n";
}

int main() {
    std::cout << "=== Comprehensive C++ Examples ===\n\n";

    try {
        demonstrate_basic_features();
        demonstrate_templates_and_generics();
        demonstrate_concurrency();
        demonstrate_smart_pointers();
        demonstrate_data_structures();
        demonstrate_file_operations();
        demonstrate_design_patterns();
        demonstrate_performance_timing();

        std::cout << "=== C++ Features Demonstrated ===\n";
        std::cout << "- Modern C++ (C++11/14/17/20) features\n";
        std::cout << "- RAII and resource management\n";
        std::cout << "- Templates and generic programming\n";
        std::cout << "- STL containers and algorithms\n";
        std::cout << "- Smart pointers and memory management\n";
        std::cout << "- Concurrency and threading\n";
        std::cout << "- Move semantics and perfect forwarding\n";
        std::cout << "- Exception handling\n";
        std::cout << "- Design patterns (Singleton, Observer)\n";
        std::cout << "- Custom data structures\n";
        std::cout << "- File I/O operations\n";
        std::cout << "- Performance measurement\n";
        std::cout << "- Functional programming patterns\n";
        std::cout << "- Type traits and concepts\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}