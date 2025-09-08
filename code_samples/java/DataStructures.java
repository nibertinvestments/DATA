/**
 * Comprehensive Java Data Structures for AI Training - Part 1
 * ===========================================================
 * 
 * This package contains fundamental data structures implemented with
 * comprehensive documentation, generics, and examples.
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

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 1. Dynamic Array with automatic resizing
 * Time Complexity: Access O(1), Insert O(1) amortized, Delete O(n)
 * Space Complexity: O(n)
 */
class DynamicArray<T> implements Iterable<T> {
    private static final int DEFAULT_CAPACITY = 4;
    private Object[] data;
    private int size;
    private int capacity;

    /**
     * Constructs a new DynamicArray with default capacity.
     */
    public DynamicArray() {
        this(DEFAULT_CAPACITY);
    }

    /**
     * Constructs a new DynamicArray with specified initial capacity.
     * @param initialCapacity the initial capacity
     */
    public DynamicArray(int initialCapacity) {
        if (initialCapacity < 1) {
            throw new IllegalArgumentException("Capacity must be at least 1");
        }
        this.capacity = initialCapacity;
        this.data = new Object[capacity];
        this.size = 0;
    }

    /**
     * Returns the number of elements in this array.
     * @return the number of elements
     */
    public int size() {
        return size;
    }

    /**
     * Returns true if this array contains no elements.
     * @return true if empty
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Returns the element at the specified position.
     * @param index the index of the element
     * @return the element at the specified position
     * @throws IndexOutOfBoundsException if index is out of range
     */
    @SuppressWarnings("unchecked")
    public T get(int index) {
        checkIndex(index);
        return (T) data[index];
    }

    /**
     * Replaces the element at the specified position.
     * @param index the index of the element to replace
     * @param element the element to be stored
     * @return the previous element at the specified position
     */
    @SuppressWarnings("unchecked")
    public T set(int index, T element) {
        checkIndex(index);
        T oldValue = (T) data[index];
        data[index] = element;
        return oldValue;
    }

    /**
     * Appends the specified element to the end of this array.
     * @param element the element to be appended
     */
    public void add(T element) {
        ensureCapacity();
        data[size++] = element;
    }

    /**
     * Inserts the specified element at the specified position.
     * @param index the index at which to insert
     * @param element the element to be inserted
     */
    public void add(int index, T element) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        ensureCapacity();
        System.arraycopy(data, index, data, index + 1, size - index);
        data[index] = element;
        size++;
    }

    /**
     * Removes the element at the specified position.
     * @param index the index of the element to be removed
     * @return the element that was removed
     */
    @SuppressWarnings("unchecked")
    public T remove(int index) {
        checkIndex(index);
        T oldValue = (T) data[index];
        int numMoved = size - index - 1;
        if (numMoved > 0) {
            System.arraycopy(data, index + 1, data, index, numMoved);
        }
        data[--size] = null; // Clear reference
        return oldValue;
    }

    /**
     * Returns the index of the first occurrence of the specified element.
     * @param element the element to search for
     * @return the index, or -1 if not found
     */
    public int indexOf(T element) {
        for (int i = 0; i < size; i++) {
            if (Objects.equals(element, data[i])) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns true if this array contains the specified element.
     * @param element the element to check for
     * @return true if found
     */
    public boolean contains(T element) {
        return indexOf(element) >= 0;
    }

    private void checkIndex(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }

    private void ensureCapacity() {
        if (size >= capacity) {
            capacity *= 2;
            data = Arrays.copyOf(data, capacity);
        }
    }

    @Override
    public Iterator<T> iterator() {
        return new ArrayIterator();
    }

    private class ArrayIterator implements Iterator<T> {
        private int currentIndex = 0;

        @Override
        public boolean hasNext() {
            return currentIndex < size;
        }

        @Override
        @SuppressWarnings("unchecked")
        public T next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return (T) data[currentIndex++];
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < size; i++) {
            sb.append(data[i]);
            if (i < size - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}

/**
 * 2. Singly Linked List Node
 */
class ListNode<T> {
    T data;
    ListNode<T> next;

    ListNode(T data) {
        this.data = data;
        this.next = null;
    }

    ListNode(T data, ListNode<T> next) {
        this.data = data;
        this.next = next;
    }
}

/**
 * 3. Singly Linked List
 * Time Complexity: Access O(n), Insert O(1) at head, Delete O(n)
 * Space Complexity: O(n)
 */
class SinglyLinkedList<T> implements Iterable<T> {
    private ListNode<T> head;
    private int size;

    /**
     * Constructs an empty linked list.
     */
    public SinglyLinkedList() {
        this.head = null;
        this.size = 0;
    }

    /**
     * Returns the number of elements in this list.
     * @return the number of elements
     */
    public int size() {
        return size;
    }

    /**
     * Returns true if this list contains no elements.
     * @return true if empty
     */
    public boolean isEmpty() {
        return head == null;
    }

    /**
     * Inserts the specified element at the beginning of this list.
     * @param data the element to add
     */
    public void addFirst(T data) {
        head = new ListNode<>(data, head);
        size++;
    }

    /**
     * Appends the specified element to the end of this list.
     * @param data the element to add
     */
    public void addLast(T data) {
        ListNode<T> newNode = new ListNode<>(data);
        if (head == null) {
            head = newNode;
        } else {
            ListNode<T> current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
        size++;
    }

    /**
     * Alias for addLast for convenience.
     * @param data the element to add
     */
    public void add(T data) {
        addLast(data);
    }

    /**
     * Returns the first element in this list.
     * @return the first element
     * @throws NoSuchElementException if list is empty
     */
    public T getFirst() {
        if (head == null) {
            throw new NoSuchElementException("List is empty");
        }
        return head.data;
    }

    /**
     * Returns the last element in this list.
     * @return the last element
     * @throws NoSuchElementException if list is empty
     */
    public T getLast() {
        if (head == null) {
            throw new NoSuchElementException("List is empty");
        }
        ListNode<T> current = head;
        while (current.next != null) {
            current = current.next;
        }
        return current.data;
    }

    /**
     * Removes and returns the first element from this list.
     * @return the first element
     * @throws NoSuchElementException if list is empty
     */
    public T removeFirst() {
        if (head == null) {
            throw new NoSuchElementException("List is empty");
        }
        T data = head.data;
        head = head.next;
        size--;
        return data;
    }

    /**
     * Removes the first occurrence of the specified element.
     * @param data the element to remove
     * @return true if the element was removed
     */
    public boolean remove(T data) {
        if (head == null) {
            return false;
        }

        if (Objects.equals(head.data, data)) {
            head = head.next;
            size--;
            return true;
        }

        ListNode<T> current = head;
        while (current.next != null) {
            if (Objects.equals(current.next.data, data)) {
                current.next = current.next.next;
                size--;
                return true;
            }
            current = current.next;
        }
        return false;
    }

    /**
     * Returns true if this list contains the specified element.
     * @param data the element to search for
     * @return true if found
     */
    public boolean contains(T data) {
        ListNode<T> current = head;
        while (current != null) {
            if (Objects.equals(current.data, data)) {
                return true;
            }
            current = current.next;
        }
        return false;
    }

    /**
     * Removes all elements from this list.
     */
    public void clear() {
        head = null;
        size = 0;
    }

    @Override
    public Iterator<T> iterator() {
        return new LinkedListIterator();
    }

    private class LinkedListIterator implements Iterator<T> {
        private ListNode<T> current = head;

        @Override
        public boolean hasNext() {
            return current != null;
        }

        @Override
        public T next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            T data = current.data;
            current = current.next;
            return data;
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        ListNode<T> current = head;
        while (current != null) {
            sb.append(current.data);
            if (current.next != null) sb.append(", ");
            current = current.next;
        }
        sb.append("]");
        return sb.toString();
    }
}

/**
 * 4. Stack implementation using dynamic array
 * Time Complexity: Push O(1) amortized, Pop O(1), Peek O(1)
 * Space Complexity: O(n)
 */
class Stack<T> implements Iterable<T> {
    private DynamicArray<T> array;

    /**
     * Constructs an empty stack.
     */
    public Stack() {
        this.array = new DynamicArray<>();
    }

    /**
     * Returns the number of elements in this stack.
     * @return the number of elements
     */
    public int size() {
        return array.size();
    }

    /**
     * Returns true if this stack contains no elements.
     * @return true if empty
     */
    public boolean isEmpty() {
        return array.isEmpty();
    }

    /**
     * Pushes an element onto the top of this stack.
     * @param item the element to push
     */
    public void push(T item) {
        array.add(item);
    }

    /**
     * Removes and returns the top element of this stack.
     * @return the top element
     * @throws EmptyStackException if stack is empty
     */
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return array.remove(array.size() - 1);
    }

    /**
     * Returns the top element without removing it.
     * @return the top element
     * @throws EmptyStackException if stack is empty
     */
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return array.get(array.size() - 1);
    }

    /**
     * Searches for an element in this stack.
     * @param item the element to search for
     * @return the 1-based position from the top of the stack, or -1 if not found
     */
    public int search(T item) {
        for (int i = array.size() - 1; i >= 0; i--) {
            if (Objects.equals(array.get(i), item)) {
                return array.size() - i;
            }
        }
        return -1;
    }

    @Override
    public Iterator<T> iterator() {
        return new StackIterator();
    }

    private class StackIterator implements Iterator<T> {
        private int currentIndex = array.size() - 1;

        @Override
        public boolean hasNext() {
            return currentIndex >= 0;
        }

        @Override
        public T next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return array.get(currentIndex--);
        }
    }

    @Override
    public String toString() {
        return array.toString();
    }
}

/**
 * 5. Queue implementation using circular array
 * Time Complexity: Enqueue O(1), Dequeue O(1), Peek O(1)
 * Space Complexity: O(n)
 */
class Queue<T> implements Iterable<T> {
    private Object[] data;
    private int front;
    private int rear;
    private int size;
    private int capacity;

    /**
     * Constructs a queue with default capacity.
     */
    public Queue() {
        this(16);
    }

    /**
     * Constructs a queue with specified initial capacity.
     * @param initialCapacity the initial capacity
     */
    public Queue(int initialCapacity) {
        if (initialCapacity < 1) {
            throw new IllegalArgumentException("Capacity must be at least 1");
        }
        this.capacity = initialCapacity;
        this.data = new Object[capacity];
        this.front = 0;
        this.rear = 0;
        this.size = 0;
    }

    /**
     * Returns the number of elements in this queue.
     * @return the number of elements
     */
    public int size() {
        return size;
    }

    /**
     * Returns true if this queue contains no elements.
     * @return true if empty
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Inserts the specified element into this queue.
     * @param item the element to add
     */
    public void enqueue(T item) {
        if (size == capacity) {
            resize();
        }
        data[rear] = item;
        rear = (rear + 1) % capacity;
        size++;
    }

    /**
     * Retrieves and removes the head of this queue.
     * @return the head of this queue
     * @throws NoSuchElementException if queue is empty
     */
    @SuppressWarnings("unchecked")
    public T dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue is empty");
        }
        T item = (T) data[front];
        data[front] = null; // Clear reference
        front = (front + 1) % capacity;
        size--;
        return item;
    }

    /**
     * Retrieves, but does not remove, the head of this queue.
     * @return the head of this queue
     * @throws NoSuchElementException if queue is empty
     */
    @SuppressWarnings("unchecked")
    public T peek() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue is empty");
        }
        return (T) data[front];
    }

    private void resize() {
        Object[] newData = new Object[capacity * 2];
        for (int i = 0; i < size; i++) {
            newData[i] = data[(front + i) % capacity];
        }
        data = newData;
        front = 0;
        rear = size;
        capacity *= 2;
    }

    @Override
    public Iterator<T> iterator() {
        return new QueueIterator();
    }

    private class QueueIterator implements Iterator<T> {
        private int currentIndex = 0;

        @Override
        public boolean hasNext() {
            return currentIndex < size;
        }

        @Override
        @SuppressWarnings("unchecked")
        public T next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            T item = (T) data[(front + currentIndex) % capacity];
            currentIndex++;
            return item;
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < size; i++) {
            sb.append(data[(front + i) % capacity]);
            if (i < size - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}

/**
 * Custom exception for empty stack operations
 */
class EmptyStackException extends RuntimeException {
    public EmptyStackException() {
        super("Stack is empty");
    }

    public EmptyStackException(String message) {
        super(message);
    }
}

/**
 * Test class for all Java data structures
 */
class DataStructuresTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing Java Data Structures...\n");

        testDynamicArray();
        testSinglyLinkedList();
        testStack();
        testQueue();

        System.out.println("🎉 All Java data structure tests passed!");
    }

    private static void testDynamicArray() {
        System.out.println("Testing DynamicArray:");
        DynamicArray<Integer> arr = new DynamicArray<>();
        
        // Test adding elements
        for (int i = 0; i < 10; i++) {
            arr.add(i);
        }
        
        assert arr.size() == 10 : "Size should be 10";
        assert arr.get(5).equals(5) : "Element at index 5 should be 5";
        assert arr.contains(7) : "Should contain 7";
        assert !arr.contains(15) : "Should not contain 15";
        
        // Test removal
        Integer removed = arr.remove(5);
        assert removed.equals(5) : "Removed element should be 5";
        assert arr.size() == 9 : "Size should be 9 after removal";
        
        System.out.println("✅ DynamicArray tests passed\n");
    }

    private static void testSinglyLinkedList() {
        System.out.println("Testing SinglyLinkedList:");
        SinglyLinkedList<String> list = new SinglyLinkedList<>();
        
        // Test adding elements
        list.add("hello");
        list.add("world");
        list.addFirst("hi");
        
        assert list.size() == 3 : "Size should be 3";
        assert list.getFirst().equals("hi") : "First element should be 'hi'";
        assert list.getLast().equals("world") : "Last element should be 'world'";
        assert list.contains("hello") : "Should contain 'hello'";
        
        // Test removal
        boolean removed = list.remove("hello");
        assert removed : "Should successfully remove 'hello'";
        assert list.size() == 2 : "Size should be 2 after removal";
        
        System.out.println("✅ SinglyLinkedList tests passed\n");
    }

    private static void testStack() {
        System.out.println("Testing Stack:");
        Stack<Integer> stack = new Stack<>();
        
        // Test pushing elements
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        assert stack.size() == 3 : "Size should be 3";
        assert stack.peek().equals(3) : "Top element should be 3";
        
        // Test popping elements
        Integer popped = stack.pop();
        assert popped.equals(3) : "Popped element should be 3";
        assert stack.size() == 2 : "Size should be 2 after pop";
        assert stack.peek().equals(2) : "Top element should now be 2";
        
        // Test search
        assert stack.search(1) == 2 : "Element 1 should be at position 2 from top";
        assert stack.search(5) == -1 : "Element 5 should not be found";
        
        System.out.println("✅ Stack tests passed\n");
    }

    private static void testQueue() {
        System.out.println("Testing Queue:");
        Queue<String> queue = new Queue<>();
        
        // Test enqueuing elements
        queue.enqueue("first");
        queue.enqueue("second");
        queue.enqueue("third");
        
        assert queue.size() == 3 : "Size should be 3";
        assert queue.peek().equals("first") : "Front element should be 'first'";
        
        // Test dequeuing elements
        String dequeued = queue.dequeue();
        assert dequeued.equals("first") : "Dequeued element should be 'first'";
        assert queue.size() == 2 : "Size should be 2 after dequeue";
        assert queue.peek().equals("second") : "Front element should now be 'second'";
        
        System.out.println("✅ Queue tests passed\n");
    }
}