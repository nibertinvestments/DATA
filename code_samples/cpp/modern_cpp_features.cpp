// Comprehensive C++ Modern Programming Examples
// Demonstrates C++17/20/23 features, RAII, templates, and best practices

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <stack>
#include <optional>
#include <variant>
#include <any>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>
#include <atomic>
#include <condition_variable>
#include <random>
#include <fstream>
#include <sstream>
#include <regex>
#include <array>
#include <tuple>
#include <type_traits>
#include <concepts>
#include <ranges>
#include <span>
#include <format>

// ============ Modern C++ Type System and Concepts ============

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<typename T>
concept Printable = requires(T t) {
    std::cout << t;
};

template<typename T>
concept Container = requires(T t) {
    t.begin();
    t.end();
    t.size();
};

template<typename T>
concept Comparable = requires(T a, T b) {
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
    { a < b } -> std::convertible_to<bool>;
};

// ============ Value Objects and Strong Types ============

class Money {
private:
    int64_t cents_;
    std::string currency_;

public:
    Money(double amount, const std::string& currency) 
        : cents_(static_cast<int64_t>(amount * 100)), currency_(currency) {
        if (currency.empty()) {
            throw std::invalid_argument("Currency cannot be empty");
        }
    }

    static Money fromCents(int64_t cents, const std::string& currency) {
        Money m(0, currency);
        m.cents_ = cents;
        return m;
    }

    double amount() const { return static_cast<double>(cents_) / 100.0; }
    int64_t cents() const { return cents_; }
    const std::string& currency() const { return currency_; }

    Money operator+(const Money& other) const {
        if (currency_ != other.currency_) {
            throw std::invalid_argument("Cannot add different currencies");
        }
        return fromCents(cents_ + other.cents_, currency_);
    }

    Money operator-(const Money& other) const {
        if (currency_ != other.currency_) {
            throw std::invalid_argument("Cannot subtract different currencies");
        }
        if (cents_ < other.cents_) {
            throw std::runtime_error("Insufficient funds");
        }
        return fromCents(cents_ - other.cents_, currency_);
    }

    Money operator*(double multiplier) const {
        return fromCents(static_cast<int64_t>(cents_ * multiplier), currency_);
    }

    bool operator==(const Money& other) const {
        return cents_ == other.cents_ && currency_ == other.currency_;
    }

    bool operator<(const Money& other) const {
        if (currency_ != other.currency_) {
            throw std::invalid_argument("Cannot compare different currencies");
        }
        return cents_ < other.cents_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Money& money) {
        return os << std::format("{:.2f} {}", money.amount(), money.currency_);
    }
};

template<typename T>
class StrongId {
private:
    T value_;

public:
    explicit StrongId(T value) : value_(std::move(value)) {}
    
    const T& get() const { return value_; }
    
    bool operator==(const StrongId& other) const { return value_ == other.value_; }
    bool operator!=(const StrongId& other) const { return !(*this == other); }
    bool operator<(const StrongId& other) const { return value_ < other.value_; }
    
    template<typename Hash = std::hash<T>>
    struct Hasher {
        std::size_t operator()(const StrongId& id) const {
            return Hash{}(id.value_);
        }
    };
};

using UserId = StrongId<int64_t>;
using ProductId = StrongId<std::string>;
using OrderId = StrongId<std::string>;

// ============ Smart Pointers and RAII ============

class Resource {
private:
    std::string name_;
    mutable std::mutex mutex_;
    bool is_open_;

public:
    explicit Resource(const std::string& name) : name_(name), is_open_(false) {
        std::cout << "Resource '" << name_ << "' constructed\n";
    }

    ~Resource() {
        std::lock_guard lock(mutex_);
        if (is_open_) {
            close();
        }
        std::cout << "Resource '" << name_ << "' destroyed\n";
    }

    // Non-copyable but movable
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    Resource(Resource&& other) noexcept : name_(std::move(other.name_)), is_open_(other.is_open_) {
        other.is_open_ = false;
        std::cout << "Resource '" << name_ << "' moved\n";
    }

    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            std::lock_guard lock1(mutex_);
            std::lock_guard lock2(other.mutex_);
            
            if (is_open_) {
                close();
            }
            
            name_ = std::move(other.name_);
            is_open_ = other.is_open_;
            other.is_open_ = false;
        }
        return *this;
    }

    void open() {
        std::lock_guard lock(mutex_);
        if (!is_open_) {
            is_open_ = true;
            std::cout << "Resource '" << name_ << "' opened\n";
        }
    }

    void close() {
        // Note: mutex should already be locked when calling this private method
        if (is_open_) {
            is_open_ = false;
            std::cout << "Resource '" << name_ << "' closed\n";
        }
    }

    bool isOpen() const {
        std::lock_guard lock(mutex_);
        return is_open_;
    }

    const std::string& getName() const { return name_; }
};

class ResourceManager {
private:
    std::vector<std::unique_ptr<Resource>> resources_;
    mutable std::shared_mutex mutex_;

public:
    std::shared_ptr<Resource> createResource(const std::string& name) {
        std::unique_lock lock(mutex_);
        auto resource = std::make_shared<Resource>(name);
        resource->open();
        return resource;
    }

    std::vector<std::shared_ptr<Resource>> getAllResources() const {
        std::shared_lock lock(mutex_);
        std::vector<std::shared_ptr<Resource>> result;
        for (const auto& res : resources_) {
            if (res) {
                result.push_back(std::shared_ptr<Resource>(res.get(), [](Resource*){})); // Non-owning shared_ptr
            }
        }
        return result;
    }

    size_t getResourceCount() const {
        std::shared_lock lock(mutex_);
        return resources_.size();
    }
};

// ============ Template Metaprogramming and SFINAE ============

template<typename T>
struct is_smart_pointer : std::false_type {};

template<typename T>
struct is_smart_pointer<std::unique_ptr<T>> : std::true_type {};

template<typename T>
struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};

template<typename T>
struct is_smart_pointer<std::weak_ptr<T>> : std::true_type {};

template<typename T>
constexpr bool is_smart_pointer_v = is_smart_pointer<T>::value;

// SFINAE example
template<typename T>
auto getValue(const T& t) -> decltype(t.value()) {
    return t.value();
}

template<typename T>
auto getValue(const T& t) -> std::enable_if_t<!std::is_same_v<decltype(t.value()), void>, T> {
    return t;
}

// Variadic template for type-safe printf
template<typename... Args>
void print(const std::string& format, Args&&... args) {
    std::cout << std::vformat(format, std::make_format_args(args...)) << '\n';
}

// Template specialization for container printing
template<Container T>
void printContainer(const T& container) {
    std::cout << "[";
    bool first = true;
    for (const auto& item : container) {
        if (!first) std::cout << ", ";
        std::cout << item;
        first = false;
    }
    std::cout << "]\n";
}

// ============ Design Patterns ============

// Singleton with thread safety
template<typename T>
class Singleton {
private:
    static std::once_flag initialized_;
    static std::unique_ptr<T> instance_;

public:
    static T& getInstance() {
        std::call_once(initialized_, []() {
            instance_ = std::make_unique<T>();
        });
        return *instance_;
    }

    // Delete copy and move
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;

protected:
    Singleton() = default;
    virtual ~Singleton() = default;
};

template<typename T>
std::once_flag Singleton<T>::initialized_;

template<typename T>
std::unique_ptr<T> Singleton<T>::instance_;

// Observer pattern with modern C++
template<typename EventType>
class Observable {
private:
    std::vector<std::function<void(const EventType&)>> observers_;
    mutable std::mutex mutex_;

public:
    void subscribe(std::function<void(const EventType&)> observer) {
        std::lock_guard lock(mutex_);
        observers_.push_back(std::move(observer));
    }

    void notify(const EventType& event) {
        std::shared_lock lock(mutex_);
        for (const auto& observer : observers_) {
            observer(event);
        }
    }

    size_t observerCount() const {
        std::shared_lock lock(mutex_);
        return observers_.size();
    }
};

// Strategy pattern with concepts
template<typename T>
concept SortStrategy = requires(T strategy, std::vector<int>& data) {
    strategy.sort(data);
};

class BubbleSort {
public:
    void sort(std::vector<int>& data) {
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

class QuickSort {
public:
    void sort(std::vector<int>& data) {
        quicksort(data, 0, static_cast<int>(data.size()) - 1);
    }

private:
    void quicksort(std::vector<int>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quicksort(arr, low, pi - 1);
            quicksort(arr, pi + 1, high);
        }
    }

    int partition(std::vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; ++j) {
            if (arr[j] < pivot) {
                ++i;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

template<SortStrategy Strategy>
class Sorter {
private:
    Strategy strategy_;

public:
    explicit Sorter(Strategy strategy) : strategy_(std::move(strategy)) {}

    void sort(std::vector<int>& data) {
        strategy_.sort(data);
    }
};

// ============ Domain Models ============

enum class UserRole {
    Guest,
    User,
    Moderator,
    Admin
};

std::string_view toString(UserRole role) {
    switch (role) {
        case UserRole::Guest: return "Guest";
        case UserRole::User: return "User";
        case UserRole::Moderator: return "Moderator";
        case UserRole::Admin: return "Admin";
    }
    return "Unknown";
}

class User {
private:
    UserId id_;
    std::string firstName_;
    std::string lastName_;
    std::string email_;
    UserRole role_;
    bool isActive_;
    std::chrono::system_clock::time_point createdAt_;
    std::optional<std::chrono::system_clock::time_point> lastLoginAt_;

public:
    User(UserId id, std::string firstName, std::string lastName, 
         std::string email, UserRole role = UserRole::User)
        : id_(std::move(id)), firstName_(std::move(firstName)), 
          lastName_(std::move(lastName)), email_(std::move(email)),
          role_(role), isActive_(true), createdAt_(std::chrono::system_clock::now()) {
        validate();
    }

    // Getters
    const UserId& getId() const { return id_; }
    const std::string& getFirstName() const { return firstName_; }
    const std::string& getLastName() const { return lastName_; }
    const std::string& getEmail() const { return email_; }
    UserRole getRole() const { return role_; }
    bool isActive() const { return isActive_; }
    auto getCreatedAt() const { return createdAt_; }
    const auto& getLastLoginAt() const { return lastLoginAt_; }

    // Business methods
    std::string getFullName() const {
        return firstName_ + " " + lastName_;
    }

    void updateProfile(const std::string& firstName, const std::string& lastName, const std::string& email) {
        firstName_ = firstName;
        lastName_ = lastName;
        email_ = email;
        validate();
    }

    void changeRole(UserRole newRole) {
        role_ = newRole;
    }

    void recordLogin() {
        lastLoginAt_ = std::chrono::system_clock::now();
    }

    void deactivate() { isActive_ = false; }
    void activate() { isActive_ = true; }

    bool hasPermission(std::string_view permission) const {
        // Simplified permission system
        switch (role_) {
            case UserRole::Guest:
                return permission == "read";
            case UserRole::User:
                return permission == "read" || permission == "write_own";
            case UserRole::Moderator:
                return permission == "read" || permission == "write_own" || permission == "moderate";
            case UserRole::Admin:
                return true; // Admin has all permissions
        }
        return false;
    }

    friend std::ostream& operator<<(std::ostream& os, const User& user) {
        return os << std::format("User{{id={}, name='{}', email='{}', role={}}}",
                                user.id_.get(), user.getFullName(), user.email_, toString(user.role_));
    }

private:
    void validate() const {
        if (firstName_.empty()) {
            throw std::invalid_argument("First name cannot be empty");
        }
        if (lastName_.empty()) {
            throw std::invalid_argument("Last name cannot be empty");
        }
        if (email_.empty() || email_.find('@') == std::string::npos) {
            throw std::invalid_argument("Invalid email address");
        }
    }
};

enum class TaskStatus {
    Draft,
    Active,
    InProgress,
    Review,
    Completed,
    Cancelled
};

std::string_view toString(TaskStatus status) {
    switch (status) {
        case TaskStatus::Draft: return "Draft";
        case TaskStatus::Active: return "Active";
        case TaskStatus::InProgress: return "InProgress";
        case TaskStatus::Review: return "Review";
        case TaskStatus::Completed: return "Completed";
        case TaskStatus::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

class Task {
private:
    int id_;
    std::string title_;
    std::string description_;
    TaskStatus status_;
    int priority_;
    UserId assigneeId_;
    UserId creatorId_;
    std::chrono::system_clock::time_point createdAt_;
    std::chrono::system_clock::time_point updatedAt_;
    std::optional<std::chrono::system_clock::time_point> dueDate_;
    std::optional<std::chrono::system_clock::time_point> completedAt_;
    std::vector<std::string> tags_;

public:
    Task(int id, std::string title, std::string description, int priority,
         UserId assigneeId, UserId creatorId)
        : id_(id), title_(std::move(title)), description_(std::move(description)),
          status_(TaskStatus::Draft), priority_(priority), assigneeId_(std::move(assigneeId)),
          creatorId_(std::move(creatorId)), createdAt_(std::chrono::system_clock::now()),
          updatedAt_(createdAt_) {
        validate();
    }

    // Getters
    int getId() const { return id_; }
    const std::string& getTitle() const { return title_; }
    const std::string& getDescription() const { return description_; }
    TaskStatus getStatus() const { return status_; }
    int getPriority() const { return priority_; }
    const UserId& getAssigneeId() const { return assigneeId_; }
    const UserId& getCreatorId() const { return creatorId_; }
    auto getCreatedAt() const { return createdAt_; }
    auto getUpdatedAt() const { return updatedAt_; }
    const auto& getDueDate() const { return dueDate_; }
    const auto& getCompletedAt() const { return completedAt_; }
    const auto& getTags() const { return tags_; }

    // Business methods
    void updateDetails(const std::string& title, const std::string& description, int priority) {
        title_ = title;
        description_ = description;
        priority_ = priority;
        touch();
        validate();
    }

    void changeStatus(TaskStatus newStatus) {
        status_ = newStatus;
        touch();
        
        if (newStatus == TaskStatus::Completed) {
            completedAt_ = std::chrono::system_clock::now();
        } else if (status_ == TaskStatus::Completed) {
            completedAt_.reset();
        }
    }

    void setDueDate(std::chrono::system_clock::time_point dueDate) {
        dueDate_ = dueDate;
        touch();
    }

    void addTag(const std::string& tag) {
        if (std::find(tags_.begin(), tags_.end(), tag) == tags_.end()) {
            tags_.push_back(tag);
            touch();
        }
    }

    void removeTag(const std::string& tag) {
        auto it = std::find(tags_.begin(), tags_.end(), tag);
        if (it != tags_.end()) {
            tags_.erase(it);
            touch();
        }
    }

    bool isOverdue() const {
        return dueDate_.has_value() && 
               dueDate_.value() < std::chrono::system_clock::now() &&
               status_ != TaskStatus::Completed && 
               status_ != TaskStatus::Cancelled;
    }

    bool isCompleted() const {
        return status_ == TaskStatus::Completed;
    }

    auto getTimeToCompletion() const -> std::optional<std::chrono::duration<double>> {
        if (!completedAt_.has_value()) {
            return std::nullopt;
        }
        return completedAt_.value() - createdAt_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Task& task) {
        return os << std::format("Task{{id={}, title='{}', status={}, priority={}}}",
                                task.id_, task.title_, toString(task.status_), task.priority_);
    }

private:
    void validate() const {
        if (title_.empty()) {
            throw std::invalid_argument("Title cannot be empty");
        }
        if (title_.length() > 200) {
            throw std::invalid_argument("Title cannot exceed 200 characters");
        }
        if (description_.length() > 2000) {
            throw std::invalid_argument("Description cannot exceed 2000 characters");
        }
        if (priority_ < 1 || priority_ > 4) {
            throw std::invalid_argument("Priority must be between 1 and 4");
        }
    }

    void touch() {
        updatedAt_ = std::chrono::system_clock::now();
    }
};

// ============ Repository Pattern ============

template<typename T, typename IdType>
class Repository {
public:
    virtual ~Repository() = default;
    virtual void save(const T& entity) = 0;
    virtual std::optional<T> findById(const IdType& id) const = 0;
    virtual std::vector<T> findAll() const = 0;
    virtual bool deleteById(const IdType& id) = 0;
    virtual size_t count() const = 0;
};

template<typename T, typename IdType>
class InMemoryRepository : public Repository<T, IdType> {
private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<IdType, T, typename IdType::template Hasher<>> data_;

public:
    void save(const T& entity) override {
        std::unique_lock lock(mutex_);
        if constexpr (std::is_same_v<T, User>) {
            data_[entity.getId()] = entity;
        } else if constexpr (std::is_same_v<T, Task>) {
            data_[IdType(entity.getId())] = entity;
        }
    }

    std::optional<T> findById(const IdType& id) const override {
        std::shared_lock lock(mutex_);
        auto it = data_.find(id);
        if (it != data_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<T> findAll() const override {
        std::shared_lock lock(mutex_);
        std::vector<T> result;
        result.reserve(data_.size());
        
        std::ranges::transform(data_, std::back_inserter(result),
                             [](const auto& pair) { return pair.second; });
        return result;
    }

    bool deleteById(const IdType& id) override {
        std::unique_lock lock(mutex_);
        return data_.erase(id) > 0;
    }

    size_t count() const override {
        std::shared_lock lock(mutex_);
        return data_.size();
    }

    // Additional query methods
    template<typename Predicate>
    std::vector<T> findWhere(Predicate predicate) const {
        std::shared_lock lock(mutex_);
        std::vector<T> result;
        
        std::ranges::copy_if(data_ | std::views::values, std::back_inserter(result), predicate);
        return result;
    }
};

// ============ Service Layer ============

class UserService {
private:
    std::unique_ptr<Repository<User, UserId>> userRepository_;
    Observable<User> userEvents_;

public:
    explicit UserService(std::unique_ptr<Repository<User, UserId>> repository)
        : userRepository_(std::move(repository)) {}

    User createUser(const std::string& firstName, const std::string& lastName, 
                   const std::string& email, UserRole role = UserRole::User) {
        // Check for duplicate email
        auto existingUsers = userRepository_->findAll();
        auto emailExists = std::ranges::any_of(existingUsers, 
            [&email](const User& user) { return user.getEmail() == email; });
        
        if (emailExists) {
            throw std::runtime_error("User with this email already exists");
        }

        static int64_t nextId = 1;
        UserId id(nextId++);
        
        User user(id, firstName, lastName, email, role);
        userRepository_->save(user);
        
        userEvents_.notify(user);
        return user;
    }

    std::optional<User> getUserById(const UserId& id) const {
        return userRepository_->findById(id);
    }

    std::vector<User> getAllUsers() const {
        return userRepository_->findAll();
    }

    std::vector<User> getActiveUsers() const {
        auto repo = dynamic_cast<InMemoryRepository<User, UserId>*>(userRepository_.get());
        if (repo) {
            return repo->findWhere([](const User& user) { return user.isActive(); });
        }
        
        auto users = userRepository_->findAll();
        std::vector<User> activeUsers;
        std::ranges::copy_if(users, std::back_inserter(activeUsers),
                           [](const User& user) { return user.isActive(); });
        return activeUsers;
    }

    bool updateUser(const UserId& id, const std::string& firstName, 
                   const std::string& lastName, const std::string& email) {
        auto user = userRepository_->findById(id);
        if (!user.has_value()) {
            return false;
        }

        user->updateProfile(firstName, lastName, email);
        userRepository_->save(*user);
        return true;
    }

    bool deleteUser(const UserId& id) {
        return userRepository_->deleteById(id);
    }

    void subscribeToUserEvents(std::function<void(const User&)> callback) {
        userEvents_.subscribe(std::move(callback));
    }

    size_t getUserCount() const {
        return userRepository_->count();
    }
};

// ============ Concurrency and Threading ============

class ThreadSafeCounter {
private:
    std::atomic<int> count_{0};

public:
    void increment() { ++count_; }
    void decrement() { --count_; }
    int get() const { return count_.load(); }
    
    int incrementAndGet() { return ++count_; }
    int decrementAndGet() { return --count_; }
};

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;

public:
    void push(T item) {
        std::unique_lock lock(mutex_);
        queue_.push(std::move(item));
        condition_.notify_one();
    }

    bool tryPop(T& item) {
        std::unique_lock lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void waitAndPop(T& item) {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        item = std::move(queue_.front());
        queue_.pop();
    }

    bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }
};

class WorkerPool {
private:
    std::vector<std::thread> workers_;
    ThreadSafeQueue<std::function<void()>> taskQueue_;
    std::atomic<bool> stop_{false};

public:
    explicit WorkerPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers_.emplace_back([this] {
                while (!stop_) {
                    std::function<void()> task;
                    if (taskQueue_.tryPop(task)) {
                        task();
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            });
        }
    }

    ~WorkerPool() {
        stop_ = true;
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;
        
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<ReturnType> result = task->get_future();
        
        taskQueue_.push([task]() { (*task)(); });
        
        return result;
    }

    size_t getWorkerCount() const { return workers_.size(); }
};

// ============ Algorithms and Functional Programming ============

namespace algorithms {

template<std::ranges::range Range, typename Predicate>
auto filter(Range&& range, Predicate pred) {
    return range | std::views::filter(pred);
}

template<std::ranges::range Range, typename Transform>
auto map(Range&& range, Transform transform) {
    return range | std::views::transform(transform);
}

template<std::ranges::range Range, typename T, typename BinaryOp>
T reduce(Range&& range, T init, BinaryOp op) {
    return std::ranges::fold_left(range, init, op);
}

template<std::ranges::range Range>
auto sort(Range&& range) {
    std::vector<std::ranges::range_value_t<Range>> result(range.begin(), range.end());
    std::ranges::sort(result);
    return result;
}

template<std::ranges::range Range, typename Compare>
auto sort(Range&& range, Compare comp) {
    std::vector<std::ranges::range_value_t<Range>> result(range.begin(), range.end());
    std::ranges::sort(result, comp);
    return result;
}

template<typename T>
class Maybe {
private:
    std::optional<T> value_;

public:
    Maybe() = default;
    Maybe(T value) : value_(std::move(value)) {}
    Maybe(std::nullopt_t) : value_(std::nullopt) {}

    bool hasValue() const { return value_.has_value(); }
    const T& getValue() const { return value_.value(); }
    
    template<typename F>
    auto map(F func) const -> Maybe<std::invoke_result_t<F, T>> {
        if (hasValue()) {
            return Maybe<std::invoke_result_t<F, T>>(func(getValue()));
        }
        return Maybe<std::invoke_result_t<F, T>>();
    }

    template<typename F>
    auto flatMap(F func) const -> std::invoke_result_t<F, T> {
        if (hasValue()) {
            return func(getValue());
        }
        return std::invoke_result_t<F, T>();
    }

    template<typename F>
    Maybe filter(F predicate) const {
        if (hasValue() && predicate(getValue())) {
            return *this;
        }
        return Maybe();
    }

    T getOrElse(T defaultValue) const {
        return hasValue() ? getValue() : defaultValue;
    }
};

} // namespace algorithms

// ============ Demo Application ============

class DemoApplication {
private:
    std::unique_ptr<UserService> userService_;
    WorkerPool workerPool_;

public:
    DemoApplication() : workerPool_(4) {
        auto userRepo = std::make_unique<InMemoryRepository<User, UserId>>();
        userService_ = std::make_unique<UserService>(std::move(userRepo));
        
        // Subscribe to user events
        userService_->subscribeToUserEvents([](const User& user) {
            std::cout << "User event: " << user << " created\n";
        });
    }

    void run() {
        std::cout << "=== C++ Modern Programming Examples Demo ===\n\n";

        demoValueObjects();
        demoSmartPointers();
        demoTemplatesAndConcepts();
        demoDesignPatterns();
        demoDomainModels();
        demoConcurrency();
        demoAlgorithms();
        demoFunctionalProgramming();
        
        std::cout << "\n=== C++ Features Demonstrated ===\n";
        std::cout << "🚀 C++17/20/23 modern features\n";
        std::cout << "🔧 RAII and smart pointers\n";
        std::cout << "📐 Template metaprogramming and concepts\n";
        std::cout << "🎯 Strong typing and value objects\n";
        std::cout << "🏗️  Design patterns (Singleton, Observer, Strategy)\n";
        std::cout << "⚡ Concurrency and threading\n";
        std::cout << "🔒 Thread-safe data structures\n";
        std::cout << "📊 STL algorithms and ranges\n";
        std::cout << "🎨 Functional programming concepts\n";
        std::cout << "💎 Domain-driven design patterns\n";
        std::cout << "🛡️  Exception safety and error handling\n";
        std::cout << "📈 Memory management best practices\n";
        std::cout << "🔄 Move semantics and perfect forwarding\n";
        std::cout << "📝 Modern C++ idioms and best practices\n";
    }

private:
    void demoValueObjects() {
        std::cout << "=== Value Objects Demo ===\n";
        
        Money money1(100.50, "USD");
        Money money2(25.25, "USD");
        
        auto total = money1 + money2;
        std::cout << money1 << " + " << money2 << " = " << total << "\n";
        
        UserId userId(12345);
        std::cout << "User ID: " << userId.get() << "\n\n";
    }

    void demoSmartPointers() {
        std::cout << "=== Smart Pointers and RAII Demo ===\n";
        
        ResourceManager manager;
        {
            auto resource1 = manager.createResource("Database Connection");
            auto resource2 = manager.createResource("File Handle");
            
            std::cout << "Resources created and opened\n";
            std::cout << "Resource 1 is open: " << resource1->isOpen() << "\n";
        } // Resources automatically cleaned up here
        
        std::cout << "Resources cleaned up automatically\n\n";
    }

    void demoTemplatesAndConcepts() {
        std::cout << "=== Templates and Concepts Demo ===\n";
        
        std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};
        std::cout << "Original vector: ";
        printContainer(numbers);
        
        // Template with concepts
        Sorter<BubbleSort> bubbleSorter(BubbleSort{});
        auto numbersCopy = numbers;
        bubbleSorter.sort(numbersCopy);
        
        std::cout << "Bubble sorted: ";
        printContainer(numbersCopy);
        
        print("Formatted output: {} items processed", numbers.size());
        std::cout << "\n";
    }

    void demoDesignPatterns() {
        std::cout << "=== Design Patterns Demo ===\n";
        
        // Observer pattern
        Observable<std::string> eventSource;
        eventSource.subscribe([](const std::string& event) {
            std::cout << "Observer 1 received: " << event << "\n";
        });
        eventSource.subscribe([](const std::string& event) {
            std::cout << "Observer 2 received: " << event << "\n";
        });
        
        eventSource.notify("Test Event");
        std::cout << "Active observers: " << eventSource.observerCount() << "\n\n";
    }

    void demoDomainModels() {
        std::cout << "=== Domain Models Demo ===\n";
        
        try {
            auto user = userService_->createUser("Alice", "Johnson", "alice@example.com", UserRole::User);
            std::cout << "Created: " << user << "\n";
            
            auto user2 = userService_->createUser("Bob", "Smith", "bob@example.com", UserRole::Admin);
            std::cout << "Created: " << user2 << "\n";
            
            std::cout << "Total users: " << userService_->getUserCount() << "\n";
            std::cout << "Active users: " << userService_->getActiveUsers().size() << "\n";
            
            // Create a task
            UserId assigneeId(1);
            UserId creatorId(2);
            Task task(1, "Implement feature", "Add new functionality", 3, assigneeId, creatorId);
            task.addTag("backend");
            task.addTag("urgent");
            
            std::cout << "Created: " << task << "\n";
            std::cout << "Task is overdue: " << std::boolalpha << task.isOverdue() << "\n";
            
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
        
        std::cout << "\n";
    }

    void demoConcurrency() {
        std::cout << "=== Concurrency Demo ===\n";
        
        ThreadSafeCounter counter;
        std::vector<std::future<void>> futures;
        
        // Launch multiple threads to increment counter
        for (int i = 0; i < 10; ++i) {
            futures.push_back(workerPool_.enqueue([&counter]() {
                for (int j = 0; j < 100; ++j) {
                    counter.increment();
                }
            }));
        }
        
        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        std::cout << "Final counter value: " << counter.get() << "\n";
        
        // Thread-safe queue demo
        ThreadSafeQueue<int> queue;
        queue.push(1);
        queue.push(2);
        queue.push(3);
        
        int value;
        while (queue.tryPop(value)) {
            std::cout << "Popped: " << value << "\n";
        }
        
        std::cout << "\n";
    }

    void demoAlgorithms() {
        std::cout << "=== STL Algorithms and Ranges Demo ===\n";
        
        std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        
        // Using C++20 ranges
        auto evenNumbers = numbers | std::views::filter([](int n) { return n % 2 == 0; });
        auto squares = evenNumbers | std::views::transform([](int n) { return n * n; });
        
        std::cout << "Even squares: ";
        for (auto square : squares) {
            std::cout << square << " ";
        }
        std::cout << "\n";
        
        // Using custom algorithms
        auto filtered = algorithms::filter(numbers, [](int n) { return n > 5; });
        auto mapped = algorithms::map(filtered, [](int n) { return n * 2; });
        
        std::cout << "Filtered and mapped: ";
        for (auto value : mapped) {
            std::cout << value << " ";
        }
        std::cout << "\n";
        
        auto sum = algorithms::reduce(numbers, 0, std::plus<int>{});
        std::cout << "Sum: " << sum << "\n\n";
    }

    void demoFunctionalProgramming() {
        std::cout << "=== Functional Programming Demo ===\n";
        
        using algorithms::Maybe;
        
        Maybe<int> value(42);
        Maybe<int> empty;
        
        auto doubled = value.map([](int x) { return x * 2; });
        auto filtered = doubled.filter([](int x) { return x > 50; });
        
        std::cout << "Value: " << value.getOrElse(0) << "\n";
        std::cout << "Doubled: " << doubled.getOrElse(0) << "\n";
        std::cout << "Filtered: " << filtered.getOrElse(0) << "\n";
        std::cout << "Empty: " << empty.getOrElse(-1) << "\n";
        
        // Chain operations
        auto result = Maybe<int>(10)
            .map([](int x) { return x * 3; })
            .filter([](int x) { return x > 20; })
            .map([](int x) { return x + 5; });
        
        std::cout << "Chained operations result: " << result.getOrElse(0) << "\n\n";
    }
};

// ============ Template Specializations ============

template<>
std::once_flag Singleton<DemoApplication>::initialized_;

template<>
std::unique_ptr<DemoApplication> Singleton<DemoApplication>::instance_;

// ============ Main Function ============

int main() {
    try {
        DemoApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Application error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}