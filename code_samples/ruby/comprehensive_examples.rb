# Comprehensive Ruby Examples
# Demonstrates Ruby idioms, metaprogramming, blocks, and object-oriented patterns

# ========== Classes and Modules ==========

class User
  attr_accessor :name, :email, :age
  attr_reader :id, :created_at
  
  @@user_count = 0
  
  def initialize(name, email, age = nil)
    @id = (@@user_count += 1)
    @name = name
    @email = email
    @age = age
    @created_at = Time.now
    validate_user
  end
  
  def adult?
    @age && @age >= 18
  end
  
  def display_name
    "#{@name} (#{@email})"
  end
  
  def update_attributes(attributes)
    attributes.each do |key, value|
      public_send("#{key}=", value) if respond_to?("#{key}=")
    end
    self
  end
  
  def to_h
    {
      id: @id,
      name: @name,
      email: @email,
      age: @age,
      created_at: @created_at
    }
  end
  
  def self.total_count
    @@user_count
  end
  
  def self.create_batch(user_data_array)
    user_data_array.map { |data| new(data[:name], data[:email], data[:age]) }
  end
  
  private
  
  def validate_user
    raise ArgumentError, "Name cannot be empty" if @name.nil? || @name.strip.empty?
    raise ArgumentError, "Invalid email" unless @email =~ /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i
    raise ArgumentError, "Age must be positive" if @age && @age < 0
  end
end

# Module for common functionality
module Auditable
  def self.included(base)
    base.extend(ClassMethods)
    base.class_eval do
      attr_accessor :audit_trail
      
      def initialize(*args)
        super
        @audit_trail = []
        log_action("created")
      end
    end
  end
  
  module ClassMethods
    def with_audit(&block)
      @audit_enabled = true
      result = block.call
      @audit_enabled = false
      result
    end
    
    def audit_enabled?
      @audit_enabled || false
    end
  end
  
  def log_action(action)
    @audit_trail << {
      action: action,
      timestamp: Time.now,
      class_name: self.class.name
    }
  end
  
  def audit_summary
    @audit_trail.map { |entry| "#{entry[:action]} at #{entry[:timestamp]}" }.join(", ")
  end
end

class Product
  include Auditable
  
  attr_reader :id, :name, :price, :category
  
  @@next_id = 1
  
  def initialize(name, price, category)
    @id = @@next_id
    @@next_id += 1
    @name = name
    @price = price
    @category = category
    super() # Call Auditable's initialize
  end
  
  def update_price(new_price)
    old_price = @price
    @price = new_price
    log_action("price updated from #{old_price} to #{new_price}")
    self
  end
  
  def discount(percentage)
    discount_amount = @price * (percentage / 100.0)
    @price -= discount_amount
    log_action("applied #{percentage}% discount")
    self
  end
  
  def to_s
    "#{@name} - $#{@price} (#{@category})"
  end
end

# ========== Metaprogramming Examples ==========

class DynamicAttributes
  def initialize(attributes = {})
    @attributes = {}
    attributes.each { |key, value| set_attribute(key, value) }
  end
  
  def method_missing(method_name, *args)
    if method_name.to_s.end_with?('=')
      attribute_name = method_name.to_s.chomp('=').to_sym
      set_attribute(attribute_name, args.first)
    elsif @attributes.key?(method_name.to_sym)
      @attributes[method_name.to_sym]
    else
      super
    end
  end
  
  def respond_to_missing?(method_name, include_private = false)
    method_name.to_s.end_with?('=') || @attributes.key?(method_name.to_sym) || super
  end
  
  def set_attribute(name, value)
    @attributes[name.to_sym] = value
    
    # Define getter method
    define_singleton_method(name) { @attributes[name.to_sym] }
    
    # Define setter method
    define_singleton_method("#{name}=") { |val| @attributes[name.to_sym] = val }
  end
  
  def attributes
    @attributes.dup
  end
end

# ========== Blocks, Procs, and Lambdas ==========

class DataProcessor
  def initialize(data)
    @data = data
  end
  
  def process(&block)
    if block_given?
      @data.map(&block)
    else
      @data
    end
  end
  
  def filter(&condition)
    @data.select(&condition)
  end
  
  def reduce(initial_value = 0, &operation)
    @data.reduce(initial_value, &operation)
  end
  
  def each_with_logging(&block)
    @data.each_with_index do |item, index|
      puts "Processing item #{index + 1}: #{item}"
      block.call(item, index) if block_given?
    end
  end
  
  # Class method demonstrating different callable types
  def self.demonstrate_callables
    # Proc
    square_proc = Proc.new { |x| x * x }
    
    # Lambda
    square_lambda = lambda { |x| x * x }
    
    # Method object
    def self.square_method(x)
      x * x
    end
    square_method_obj = method(:square_method)
    
    numbers = [1, 2, 3, 4, 5]
    
    {
      proc_results: numbers.map(&square_proc),
      lambda_results: numbers.map(&square_lambda),
      method_results: numbers.map(&square_method_obj)
    }
  end
end

# ========== Functional Programming Patterns ==========

module FunctionalUtils
  module_function
  
  def compose(*functions)
    lambda do |value|
      functions.reverse.reduce(value) { |acc, func| func.call(acc) }
    end
  end
  
  def curry(func, *partial_args)
    lambda do |*remaining_args|
      func.call(*(partial_args + remaining_args))
    end
  end
  
  def pipe(value, *functions)
    functions.reduce(value) { |acc, func| func.call(acc) }
  end
  
  def memoize(func)
    cache = {}
    lambda do |*args|
      cache[args] ||= func.call(*args)
    end
  end
end

# ========== Design Patterns ==========

# Singleton pattern
class DatabaseConnection
  @@instance = nil
  @@mutex = Mutex.new
  
  private_class_method :new
  
  def self.instance
    return @@instance if @@instance
    
    @@mutex.synchronize do
      @@instance ||= new
    end
  end
  
  def initialize
    @connection_string = "default://connection"
    @connected = false
  end
  
  def connect(connection_string = nil)
    @connection_string = connection_string if connection_string
    @connected = true
    puts "Connected to #{@connection_string}"
  end
  
  def disconnect
    @connected = false
    puts "Disconnected"
  end
  
  def connected?
    @connected
  end
end

# Observer pattern
module Observable
  def initialize
    super
    @observers = []
  end
  
  def add_observer(observer)
    @observers << observer unless @observers.include?(observer)
  end
  
  def remove_observer(observer)
    @observers.delete(observer)
  end
  
  def notify_observers(event, data = nil)
    @observers.each { |observer| observer.update(event, data) }
  end
end

class UserService
  include Observable
  
  def initialize
    super
    @users = []
  end
  
  def create_user(name, email, age = nil)
    user = User.new(name, email, age)
    @users << user
    notify_observers(:user_created, user)
    user
  end
  
  def find_user(id)
    @users.find { |user| user.id == id }
  end
  
  def all_users
    @users.dup
  end
  
  def users_count
    @users.size
  end
end

class UserObserver
  def update(event, data)
    case event
    when :user_created
      puts "Observer: New user created - #{data.display_name}"
    end
  end
end

# ========== Error Handling ==========

class CustomError < StandardError
  attr_reader :error_code, :context
  
  def initialize(message, error_code = nil, context = {})
    super(message)
    @error_code = error_code
    @context = context
  end
  
  def to_h
    {
      message: message,
      error_code: @error_code,
      context: @context,
      backtrace: backtrace&.first(5)
    }
  end
end

class ErrorHandler
  def self.with_retry(max_attempts: 3, delay: 1)
    attempts = 0
    begin
      attempts += 1
      yield
    rescue StandardError => e
      if attempts < max_attempts
        puts "Attempt #{attempts} failed: #{e.message}. Retrying in #{delay} seconds..."
        sleep(delay)
        retry
      else
        raise CustomError.new("Failed after #{max_attempts} attempts", "MAX_RETRIES_EXCEEDED", { original_error: e.message })
      end
    end
  end
  
  def self.safe_execute(default_value = nil)
    begin
      yield
    rescue StandardError => e
      puts "Error caught: #{e.message}"
      default_value
    end
  end
end

# ========== File and I/O Operations ==========

class FileManager
  def self.read_json(filename)
    require 'json'
    
    File.open(filename, 'r') do |file|
      JSON.parse(file.read, symbolize_names: true)
    end
  rescue Errno::ENOENT
    puts "File not found: #{filename}"
    nil
  rescue JSON::ParserError => e
    puts "Invalid JSON in #{filename}: #{e.message}"
    nil
  end
  
  def self.write_json(filename, data)
    require 'json'
    
    File.open(filename, 'w') do |file|
      file.write(JSON.pretty_generate(data))
    end
    
    puts "Data written to #{filename}"
  end
  
  def self.process_csv(filename)
    require 'csv'
    
    results = []
    CSV.foreach(filename, headers: true, header_converters: :symbol) do |row|
      yield(row) if block_given?
      results << row.to_h
    end
    results
  rescue Errno::ENOENT
    puts "CSV file not found: #{filename}"
    []
  end
  
  def self.backup_file(source, destination = nil)
    destination ||= "#{source}.backup.#{Time.now.to_i}"
    
    File.open(source, 'rb') do |src|
      File.open(destination, 'wb') do |dest|
        IO.copy_stream(src, dest)
      end
    end
    
    puts "Backup created: #{destination}"
    destination
  end
end

# ========== Threading and Concurrency ==========

class ThreadPoolWorker
  def initialize(pool_size = 4)
    @pool_size = pool_size
    @queue = Queue.new
    @workers = []
    @mutex = Mutex.new
    @running = false
  end
  
  def start
    @running = true
    
    @pool_size.times do |i|
      @workers << Thread.new do
        Thread.current[:name] = "Worker-#{i}"
        
        while @running
          begin
            job = @queue.pop(true) # Non-blocking pop
            job.call if job.respond_to?(:call)
          rescue ThreadError
            # Queue is empty, sleep briefly
            sleep(0.1)
          rescue StandardError => e
            puts "Worker #{Thread.current[:name]} error: #{e.message}"
          end
        end
      end
    end
    
    puts "Thread pool started with #{@pool_size} workers"
  end
  
  def submit(&block)
    @queue << block
  end
  
  def shutdown
    @running = false
    @workers.each(&:join)
    puts "Thread pool shutdown complete"
  end
  
  def stats
    {
      pool_size: @pool_size,
      queue_size: @queue.size,
      running: @running,
      worker_count: @workers.size
    }
  end
end

# ========== Regular Expressions and String Processing ==========

class TextProcessor
  EMAIL_REGEX = /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i
  PHONE_REGEX = /\A(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\z/
  URL_REGEX = /\Ahttps?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&=]*)\z/
  
  def self.extract_emails(text)
    text.scan(EMAIL_REGEX)
  end
  
  def self.extract_phone_numbers(text)
    text.scan(PHONE_REGEX).map { |match| match.join }
  end
  
  def self.extract_urls(text)
    text.scan(URL_REGEX)
  end
  
  def self.sanitize_filename(filename)
    # Remove or replace invalid characters
    filename.gsub(/[\/\\:*?"<>|]/, '_').strip
  end
  
  def self.word_frequency(text)
    words = text.downcase.scan(/\b\w+\b/)
    frequency = Hash.new(0)
    words.each { |word| frequency[word] += 1 }
    frequency.sort_by { |word, count| -count }.to_h
  end
  
  def self.truncate_words(text, word_limit)
    words = text.split
    if words.length > word_limit
      words.first(word_limit).join(' ') + '...'
    else
      text
    end
  end
end

# ========== Testing and Validation ==========

class Validator
  def self.validate_user(user_hash)
    errors = []
    
    errors << "Name is required" if user_hash[:name].nil? || user_hash[:name].strip.empty?
    errors << "Invalid email format" unless user_hash[:email] =~ TextProcessor::EMAIL_REGEX
    errors << "Age must be a positive number" if user_hash[:age] && (!user_hash[:age].is_a?(Integer) || user_hash[:age] < 0)
    
    errors
  end
  
  def self.validate_product(product_hash)
    errors = []
    
    errors << "Name is required" if product_hash[:name].nil? || product_hash[:name].strip.empty?
    errors << "Price must be a positive number" unless product_hash[:price].is_a?(Numeric) && product_hash[:price] > 0
    errors << "Category is required" if product_hash[:category].nil? || product_hash[:category].strip.empty?
    
    errors
  end
end

# ========== Demonstration ==========

class RubyExamplesDemo
  def self.run_all_demos
    puts "=== Comprehensive Ruby Examples Demo ==="
    puts
    
    demonstrate_basic_oop
    demonstrate_metaprogramming
    demonstrate_blocks_and_closures
    demonstrate_functional_programming
    demonstrate_design_patterns
    demonstrate_file_operations
    demonstrate_concurrency
    demonstrate_text_processing
    demonstrate_error_handling
    
    puts
    puts "=== Ruby Features Demonstrated ==="
    puts "- Object-oriented programming with classes and modules"
    puts "- Metaprogramming and dynamic method definition"
    puts "- Blocks, procs, and lambdas"
    puts "- Functional programming patterns"
    puts "- Design patterns (Singleton, Observer)"
    puts "- File I/O and JSON/CSV processing"
    puts "- Threading and concurrent programming"
    puts "- Regular expressions and text processing"
    puts "- Error handling and custom exceptions"
    puts "- Module inclusion and class extension"
    puts "- Method aliasing and method_missing"
    puts "- Duck typing and dynamic interfaces"
  end
  
  private
  
  def self.demonstrate_basic_oop
    puts "=== Basic OOP Demo ==="
    
    user1 = User.new("Alice", "alice@example.com", 25)
    user2 = User.new("Bob", "bob@example.com", 17)
    
    puts "Created users:"
    puts "- #{user1.display_name} (Adult: #{user1.adult?})"
    puts "- #{user2.display_name} (Adult: #{user2.adult?})"
    puts "Total users created: #{User.total_count}"
    
    # Update attributes
    user1.update_attributes(name: "Alice Smith", age: 26)
    puts "Updated user: #{user1.display_name}"
    puts
  end
  
  def self.demonstrate_metaprogramming
    puts "=== Metaprogramming Demo ==="
    
    dynamic_obj = DynamicAttributes.new(name: "Dynamic", type: "Object")
    dynamic_obj.description = "This is a dynamic object"
    dynamic_obj.count = 42
    
    puts "Dynamic object attributes:"
    dynamic_obj.attributes.each { |key, value| puts "  #{key}: #{value}" }
    puts "Description: #{dynamic_obj.description}"
    puts
  end
  
  def self.demonstrate_blocks_and_closures
    puts "=== Blocks and Closures Demo ==="
    
    processor = DataProcessor.new([1, 2, 3, 4, 5])
    
    # Using blocks
    squares = processor.process { |x| x * x }
    puts "Squares: #{squares}"
    
    evens = processor.filter { |x| x.even? }
    puts "Even numbers: #{evens}"
    
    sum = processor.reduce { |acc, x| acc + x }
    puts "Sum: #{sum}"
    
    # Demonstrate different callable types
    callable_results = DataProcessor.demonstrate_callables
    puts "Callable results: #{callable_results}"
    puts
  end
  
  def self.demonstrate_functional_programming
    puts "=== Functional Programming Demo ==="
    
    # Function composition
    add_one = lambda { |x| x + 1 }
    multiply_by_two = lambda { |x| x * 2 }
    
    composed = FunctionalUtils.compose(multiply_by_two, add_one)
    result = composed.call(5) # (5 + 1) * 2 = 12
    puts "Composed function result: #{result}"
    
    # Currying
    multiply = lambda { |x, y| x * y }
    double = FunctionalUtils.curry(multiply, 2)
    puts "Curried function result: #{double.call(6)}" # 2 * 6 = 12
    
    # Piping
    piped_result = FunctionalUtils.pipe(5, add_one, multiply_by_two)
    puts "Piped result: #{piped_result}"
    puts
  end
  
  def self.demonstrate_design_patterns
    puts "=== Design Patterns Demo ==="
    
    # Singleton
    db1 = DatabaseConnection.instance
    db2 = DatabaseConnection.instance
    puts "Same database instance: #{db1.object_id == db2.object_id}"
    db1.connect("postgres://localhost/mydb")
    
    # Observer
    user_service = UserService.new
    observer = UserObserver.new
    user_service.add_observer(observer)
    
    user = user_service.create_user("Charlie", "charlie@example.com", 30)
    puts "Users in service: #{user_service.users_count}"
    puts
  end
  
  def self.demonstrate_file_operations
    puts "=== File Operations Demo ==="
    
    # Create sample data
    sample_data = { users: [{ name: "Test", email: "test@example.com" }] }
    filename = "/tmp/sample_data.json"
    
    FileManager.write_json(filename, sample_data)
    loaded_data = FileManager.read_json(filename)
    puts "Loaded data: #{loaded_data}"
    puts
  end
  
  def self.demonstrate_concurrency
    puts "=== Concurrency Demo ==="
    
    thread_pool = ThreadPoolWorker.new(2)
    thread_pool.start
    
    # Submit some jobs
    5.times do |i|
      thread_pool.submit do
        puts "Processing job #{i + 1}"
        sleep(0.1) # Simulate work
      end
    end
    
    sleep(1) # Wait for jobs to complete
    puts "Thread pool stats: #{thread_pool.stats}"
    thread_pool.shutdown
    puts
  end
  
  def self.demonstrate_text_processing
    puts "=== Text Processing Demo ==="
    
    text = "Contact us at info@example.com or call 555-123-4567. Visit https://example.com"
    
    emails = TextProcessor.extract_emails(text)
    phones = TextProcessor.extract_phone_numbers(text)
    urls = TextProcessor.extract_urls(text)
    
    puts "Found emails: #{emails}"
    puts "Found phones: #{phones}"
    puts "Found URLs: #{urls}"
    
    word_freq = TextProcessor.word_frequency("hello world hello ruby world")
    puts "Word frequency: #{word_freq}"
    puts
  end
  
  def self.demonstrate_error_handling
    puts "=== Error Handling Demo ==="
    
    # Retry mechanism
    begin
      ErrorHandler.with_retry(max_attempts: 2, delay: 0.1) do
        rand > 0.7 ? "Success!" : raise("Random failure")
      end
      puts "Operation succeeded"
    rescue CustomError => e
      puts "Operation failed: #{e.to_h}"
    end
    
    # Safe execution
    result = ErrorHandler.safe_execute("default") do
      raise "This will be caught"
    end
    puts "Safe execution result: #{result}"
    puts
  end
end

# Run the demonstration
if __FILE__ == $0
  RubyExamplesDemo.run_all_demos
end