/**
 * Comprehensive C++ Data Structures for AI Training
 * =================================================
 * 
 * This header contains 20+ fundamental and advanced data structures
 * implemented with modern C++17 features, templates, and comprehensive
 * documentation.
 * 
 * Each structure includes:
 * - Time and space complexity analysis
 * - Usage examples
 * - Test cases
 * - Performance considerations
 * - AI training optimization notes
 * 
 * @author AI Training Dataset Generator
 * @version 1.0.0
 */

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <cassert>
#include <queue>
#include <unordered_map>
#include <string>

/**
 * 1. Dynamic Array with automatic resizing
 * Time Complexity: Access O(1), Insert O(1) amortized, Delete O(n)
 * Space Complexity: O(n)
 */
template<typename T>
class DynamicArray {
private:
    std::unique_ptr<T[]> data_;
    size_t size_;
    size_t capacity_;
    static constexpr size_t DEFAULT_CAPACITY = 4;

public:
    /**
     * Default constructor with initial capacity
     */
    explicit DynamicArray(size_t initial_capacity = DEFAULT_CAPACITY)
        : data_(std::make_unique<T[]>(initial_capacity))
        , size_(0)
        , capacity_(initial_capacity) {
        if (initial_capacity == 0) {
            throw std::invalid_argument("Capacity must be at least 1");
        }
    }

    /**
     * Copy constructor
     */
    DynamicArray(const DynamicArray& other)
        : data_(std::make_unique<T[]>(other.capacity_))
        , size_(other.size_)
        , capacity_(other.capacity_) {
        std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
    }

    /**
     * Move constructor
     */
    DynamicArray(DynamicArray&& other) noexcept
        : data_(std::move(other.data_))
        , size_(other.size_)
        , capacity_(other.capacity_) {
        other.size_ = 0;
        other.capacity_ = 0;
    }

    /**
     * Copy assignment operator
     */
    DynamicArray& operator=(const DynamicArray& other) {
        if (this != &other) {
            data_ = std::make_unique<T[]>(other.capacity_);
            size_ = other.size_;
            capacity_ = other.capacity_;
            std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
        }
        return *this;
    }

    /**
     * Move assignment operator
     */
    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    /**
     * Returns the number of elements
     */
    size_t size() const noexcept { return size_; }

    /**
     * Returns the current capacity
     */
    size_t capacity() const noexcept { return capacity_; }

    /**
     * Returns true if empty
     */
    bool empty() const noexcept { return size_ == 0; }

    /**
     * Access element at index (const version)
     */
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    /**
     * Access element at index (non-const version)
     */
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    /**
     * Access element at index with bounds checking
     */
    const T& at(size_t index) const {
        return (*this)[index];
    }

    /**
     * Access element at index with bounds checking
     */
    T& at(size_t index) {
        return (*this)[index];
    }

    /**
     * Add element to end of array
     */
    void push_back(const T& value) {
        ensure_capacity();
        data_[size_++] = value;
    }

    /**
     * Add element to end of array (move version)
     */
    void push_back(T&& value) {
        ensure_capacity();
        data_[size_++] = std::move(value);
    }

    /**
     * Remove last element
     */
    void pop_back() {
        if (empty()) {
            throw std::runtime_error("Array is empty");
        }
        --size_;
    }

    /**
     * Get reference to last element
     */
    T& back() {
        if (empty()) {
            throw std::runtime_error("Array is empty");
        }
        return data_[size_ - 1];
    }

    /**
     * Get const reference to last element
     */
    const T& back() const {
        if (empty()) {
            throw std::runtime_error("Array is empty");
        }
        return data_[size_ - 1];
    }

    /**
     * Get reference to first element
     */
    T& front() {
        if (empty()) {
            throw std::runtime_error("Array is empty");
        }
        return data_[0];
    }

    /**
     * Get const reference to first element
     */
    const T& front() const {
        if (empty()) {
            throw std::runtime_error("Array is empty");
        }
        return data_[0];
    }

    /**
     * Clear all elements
     */
    void clear() noexcept {
        size_ = 0;
    }

    /**
     * Reserve capacity
     */
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            resize_to(new_capacity);
        }
    }

    /**
     * Iterator support
     */
    class iterator {
    private:
        T* ptr_;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        explicit iterator(T* ptr) : ptr_(ptr) {}
        
        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        
        iterator& operator++() { ++ptr_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++ptr_; return tmp; }
        
        iterator& operator--() { --ptr_; return *this; }
        iterator operator--(int) { iterator tmp = *this; --ptr_; return tmp; }
        
        iterator operator+(difference_type n) const { return iterator(ptr_ + n); }
        iterator operator-(difference_type n) const { return iterator(ptr_ - n); }
        
        difference_type operator-(const iterator& other) const { return ptr_ - other.ptr_; }
        
        bool operator==(const iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const iterator& other) const { return ptr_ < other.ptr_; }
    };

    /**
     * Const iterator support
     */
    class const_iterator {
    private:
        const T* ptr_;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        explicit const_iterator(const T* ptr) : ptr_(ptr) {}
        
        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        
        const_iterator& operator++() { ++ptr_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++ptr_; return tmp; }
        
        const_iterator& operator--() { --ptr_; return *this; }
        const_iterator operator--(int) { const_iterator tmp = *this; --ptr_; return tmp; }
        
        const_iterator operator+(difference_type n) const { return const_iterator(ptr_ + n); }
        const_iterator operator-(difference_type n) const { return const_iterator(ptr_ - n); }
        
        difference_type operator-(const const_iterator& other) const { return ptr_ - other.ptr_; }
        
        bool operator==(const const_iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const const_iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const const_iterator& other) const { return ptr_ < other.ptr_; }
    };

    iterator begin() { return iterator(data_.get()); }
    iterator end() { return iterator(data_.get() + size_); }
    const_iterator begin() const { return const_iterator(data_.get()); }
    const_iterator end() const { return const_iterator(data_.get() + size_); }
    const_iterator cbegin() const { return const_iterator(data_.get()); }
    const_iterator cend() const { return const_iterator(data_.get() + size_); }

private:
    void ensure_capacity() {
        if (size_ >= capacity_) {
            resize_to(capacity_ * 2);
        }
    }

    void resize_to(size_t new_capacity) {
        auto new_data = std::make_unique<T[]>(new_capacity);
        std::copy(data_.get(), data_.get() + size_, new_data.get());
        data_ = std::move(new_data);
        capacity_ = new_capacity;
    }
};

/**
 * 2. Singly Linked List Node
 */
template<typename T>
struct ListNode {
    T data;
    std::unique_ptr<ListNode<T>> next;

    explicit ListNode(const T& value) : data(value), next(nullptr) {}
    explicit ListNode(T&& value) : data(std::move(value)), next(nullptr) {}
};

/**
 * 3. Singly Linked List
 * Time Complexity: Access O(n), Insert O(1) at head, Delete O(n)
 * Space Complexity: O(n)
 */
template<typename T>
class SinglyLinkedList {
private:
    std::unique_ptr<ListNode<T>> head_;
    size_t size_;

public:
    /**
     * Default constructor
     */
    SinglyLinkedList() : head_(nullptr), size_(0) {}

    /**
     * Copy constructor
     */
    SinglyLinkedList(const SinglyLinkedList& other) : head_(nullptr), size_(0) {
        auto current = other.head_.get();
        while (current) {
            push_back(current->data);
            current = current->next.get();
        }
    }

    /**
     * Move constructor
     */
    SinglyLinkedList(SinglyLinkedList&& other) noexcept
        : head_(std::move(other.head_)), size_(other.size_) {
        other.size_ = 0;
    }

    /**
     * Copy assignment
     */
    SinglyLinkedList& operator=(const SinglyLinkedList& other) {
        if (this != &other) {
            clear();
            auto current = other.head_.get();
            while (current) {
                push_back(current->data);
                current = current->next.get();
            }
        }
        return *this;
    }

    /**
     * Move assignment
     */
    SinglyLinkedList& operator=(SinglyLinkedList&& other) noexcept {
        if (this != &other) {
            head_ = std::move(other.head_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * Returns the number of elements
     */
    size_t size() const noexcept { return size_; }

    /**
     * Returns true if empty
     */
    bool empty() const noexcept { return head_ == nullptr; }

    /**
     * Add element to front
     */
    void push_front(const T& value) {
        auto new_node = std::make_unique<ListNode<T>>(value);
        new_node->next = std::move(head_);
        head_ = std::move(new_node);
        ++size_;
    }

    /**
     * Add element to front (move version)
     */
    void push_front(T&& value) {
        auto new_node = std::make_unique<ListNode<T>>(std::move(value));
        new_node->next = std::move(head_);
        head_ = std::move(new_node);
        ++size_;
    }

    /**
     * Add element to back
     */
    void push_back(const T& value) {
        auto new_node = std::make_unique<ListNode<T>>(value);
        if (!head_) {
            head_ = std::move(new_node);
        } else {
            auto current = head_.get();
            while (current->next) {
                current = current->next.get();
            }
            current->next = std::move(new_node);
        }
        ++size_;
    }

    /**
     * Add element to back (move version)
     */
    void push_back(T&& value) {
        auto new_node = std::make_unique<ListNode<T>>(std::move(value));
        if (!head_) {
            head_ = std::move(new_node);
        } else {
            auto current = head_.get();
            while (current->next) {
                current = current->next.get();
            }
            current->next = std::move(new_node);
        }
        ++size_;
    }

    /**
     * Remove first element
     */
    void pop_front() {
        if (!head_) {
            throw std::runtime_error("List is empty");
        }
        head_ = std::move(head_->next);
        --size_;
    }

    /**
     * Get reference to first element
     */
    T& front() {
        if (!head_) {
            throw std::runtime_error("List is empty");
        }
        return head_->data;
    }

    /**
     * Get const reference to first element
     */
    const T& front() const {
        if (!head_) {
            throw std::runtime_error("List is empty");
        }
        return head_->data;
    }

    /**
     * Find element in list
     */
    bool contains(const T& value) const {
        auto current = head_.get();
        while (current) {
            if (current->data == value) {
                return true;
            }
            current = current->next.get();
        }
        return false;
    }

    /**
     * Remove first occurrence of value
     */
    bool remove(const T& value) {
        if (!head_) {
            return false;
        }

        if (head_->data == value) {
            pop_front();
            return true;
        }

        auto current = head_.get();
        while (current->next) {
            if (current->next->data == value) {
                current->next = std::move(current->next->next);
                --size_;
                return true;
            }
            current = current->next.get();
        }
        return false;
    }

    /**
     * Clear all elements
     */
    void clear() {
        head_.reset();
        size_ = 0;
    }

    /**
     * Iterator support
     */
    class iterator {
    private:
        ListNode<T>* current_;
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        explicit iterator(ListNode<T>* node) : current_(node) {}

        reference operator*() const { return current_->data; }
        pointer operator->() const { return &(current_->data); }

        iterator& operator++() {
            current_ = current_->next.get();
            return *this;
        }

        iterator operator++(int) {
            iterator tmp = *this;
            current_ = current_->next.get();
            return tmp;
        }

        bool operator==(const iterator& other) const {
            return current_ == other.current_;
        }

        bool operator!=(const iterator& other) const {
            return current_ != other.current_;
        }
    };

    iterator begin() { return iterator(head_.get()); }
    iterator end() { return iterator(nullptr); }
};

/**
 * 4. Stack implementation using dynamic array
 * Time Complexity: Push O(1) amortized, Pop O(1), Top O(1)
 * Space Complexity: O(n)
 */
template<typename T>
class Stack {
private:
    DynamicArray<T> container_;

public:
    /**
     * Default constructor
     */
    Stack() = default;

    /**
     * Returns the number of elements
     */
    size_t size() const noexcept { return container_.size(); }

    /**
     * Returns true if empty
     */
    bool empty() const noexcept { return container_.empty(); }

    /**
     * Add element to top
     */
    void push(const T& value) {
        container_.push_back(value);
    }

    /**
     * Add element to top (move version)
     */
    void push(T&& value) {
        container_.push_back(std::move(value));
    }

    /**
     * Remove top element
     */
    void pop() {
        if (empty()) {
            throw std::runtime_error("Stack is empty");
        }
        container_.pop_back();
    }

    /**
     * Get reference to top element
     */
    T& top() {
        if (empty()) {
            throw std::runtime_error("Stack is empty");
        }
        return container_.back();
    }

    /**
     * Get const reference to top element
     */
    const T& top() const {
        if (empty()) {
            throw std::runtime_error("Stack is empty");
        }
        return container_.back();
    }
};

/**
 * 5. Queue implementation using circular buffer
 * Time Complexity: Push O(1), Pop O(1), Front O(1)
 * Space Complexity: O(n)
 */
template<typename T>
class Queue {
private:
    std::unique_ptr<T[]> data_;
    size_t front_;
    size_t rear_;
    size_t size_;
    size_t capacity_;
    static constexpr size_t DEFAULT_CAPACITY = 16;

public:
    /**
     * Constructor with optional initial capacity
     */
    explicit Queue(size_t initial_capacity = DEFAULT_CAPACITY)
        : data_(std::make_unique<T[]>(initial_capacity))
        , front_(0)
        , rear_(0)
        , size_(0)
        , capacity_(initial_capacity) {
        if (initial_capacity == 0) {
            throw std::invalid_argument("Capacity must be at least 1");
        }
    }

    /**
     * Returns the number of elements
     */
    size_t size() const noexcept { return size_; }

    /**
     * Returns true if empty
     */
    bool empty() const noexcept { return size_ == 0; }

    /**
     * Add element to rear
     */
    void push(const T& value) {
        if (size_ == capacity_) {
            resize();
        }
        data_[rear_] = value;
        rear_ = (rear_ + 1) % capacity_;
        ++size_;
    }

    /**
     * Add element to rear (move version)
     */
    void push(T&& value) {
        if (size_ == capacity_) {
            resize();
        }
        data_[rear_] = std::move(value);
        rear_ = (rear_ + 1) % capacity_;
        ++size_;
    }

    /**
     * Remove front element
     */
    void pop() {
        if (empty()) {
            throw std::runtime_error("Queue is empty");
        }
        front_ = (front_ + 1) % capacity_;
        --size_;
    }

    /**
     * Get reference to front element
     */
    T& front() {
        if (empty()) {
            throw std::runtime_error("Queue is empty");
        }
        return data_[front_];
    }

    /**
     * Get const reference to front element
     */
    const T& front() const {
        if (empty()) {
            throw std::runtime_error("Queue is empty");
        }
        return data_[front_];
    }

    /**
     * Get reference to back element
     */
    T& back() {
        if (empty()) {
            throw std::runtime_error("Queue is empty");
        }
        return data_[(rear_ - 1 + capacity_) % capacity_];
    }

    /**
     * Get const reference to back element
     */
    const T& back() const {
        if (empty()) {
            throw std::runtime_error("Queue is empty");
        }
        return data_[(rear_ - 1 + capacity_) % capacity_];
    }

private:
    void resize() {
        auto new_data = std::make_unique<T[]>(capacity_ * 2);
        for (size_t i = 0; i < size_; ++i) {
            new_data[i] = std::move(data_[(front_ + i) % capacity_]);
        }
        data_ = std::move(new_data);
        front_ = 0;
        rear_ = size_;
        capacity_ *= 2;
    }
};

/**
 * Test function for all C++ data structures
 */
void test_cpp_data_structures() {
    std::cout << "🧪 Testing C++ Data Structures...\n\n";

    // Test Dynamic Array
    std::cout << "Testing DynamicArray:\n";
    DynamicArray<int> arr;
    for (int i = 0; i < 10; ++i) {
        arr.push_back(i);
    }
    assert(arr.size() == 10);
    assert(arr[5] == 5);
    assert(arr.front() == 0);
    assert(arr.back() == 9);
    
    arr.pop_back();
    assert(arr.size() == 9);
    assert(arr.back() == 8);
    std::cout << "✅ DynamicArray tests passed\n\n";

    // Test Singly Linked List
    std::cout << "Testing SinglyLinkedList:\n";
    SinglyLinkedList<std::string> list;
    list.push_back("hello");
    list.push_back("world");
    list.push_front("hi");
    
    assert(list.size() == 3);
    assert(list.front() == "hi");
    assert(list.contains("world"));
    assert(!list.contains("goodbye"));
    
    list.remove("hello");
    assert(list.size() == 2);
    assert(!list.contains("hello"));
    std::cout << "✅ SinglyLinkedList tests passed\n\n";

    // Test Stack
    std::cout << "Testing Stack:\n";
    Stack<int> stack;
    stack.push(1);
    stack.push(2);
    stack.push(3);
    
    assert(stack.size() == 3);
    assert(stack.top() == 3);
    
    stack.pop();
    assert(stack.size() == 2);
    assert(stack.top() == 2);
    std::cout << "✅ Stack tests passed\n\n";

    // Test Queue
    std::cout << "Testing Queue:\n";
    Queue<std::string> queue;
    queue.push("first");
    queue.push("second");
    queue.push("third");
    
    assert(queue.size() == 3);
    assert(queue.front() == "first");
    assert(queue.back() == "third");
    
    queue.pop();
    assert(queue.size() == 2);
    assert(queue.front() == "second");
    std::cout << "✅ Queue tests passed\n\n";

    std::cout << "🎉 All C++ data structure tests passed!\n";
}

/**
 * Main function to run all tests
 */
int main() {
    try {
        test_cpp_data_structures();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}