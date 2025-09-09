# Comprehensive Ruby Web Development Examples
# Demonstrates Ruby/Rails patterns, metaprogramming, and modern Ruby features

require 'date'
require 'json'
require 'digest'
require 'securerandom'
require 'logger'
require 'ostruct'
require 'forwardable'
require 'concurrent-ruby'

# ============ Modules and Mixins ============

# Concern pattern for shared functionality
module Timestampable
  def self.included(base)
    base.extend(ClassMethods)
    base.attr_accessor :created_at, :updated_at
  end

  module ClassMethods
    def with_timestamps
      include InstanceMethods
    end
  end

  module InstanceMethods
    def initialize(*args)
      super
      @created_at = Time.now
      @updated_at = Time.now
    end

    def touch
      @updated_at = Time.now
      self
    end

    def age
      Time.now - @created_at
    end
  end
end

module Validatable
  class ValidationError < StandardError
    attr_reader :field, :errors

    def initialize(field = nil, message = nil)
      if field.is_a?(Hash)
        @errors = field
        super("Validation failed: #{@errors.map { |k, v| "#{k}: #{v}" }.join(', ')}")
      else
        @field = field
        @errors = { field => [message] }
        super(message)
      end
    end
  end

  def self.included(base)
    base.extend(ClassMethods)
  end

  module ClassMethods
    def validates(field, **options)
      @validations ||= {}
      @validations[field] = options
    end

    def validations
      @validations || {}
    end
  end

  def validate!
    errors = {}
    
    self.class.validations.each do |field, rules|
      value = send(field)
      
      rules.each do |rule, constraint|
        case rule
        when :presence
          if constraint && (value.nil? || value.to_s.strip.empty?)
            errors[field] ||= []
            errors[field] << "can't be blank"
          end
        when :length
          if value.is_a?(String)
            if constraint[:minimum] && value.length < constraint[:minimum]
              errors[field] ||= []
              errors[field] << "must be at least #{constraint[:minimum]} characters"
            end
            if constraint[:maximum] && value.length > constraint[:maximum]
              errors[field] ||= []
              errors[field] << "can't exceed #{constraint[:maximum]} characters"
            end
          end
        when :format
          if value && !constraint.match?(value.to_s)
            errors[field] ||= []
            errors[field] << "has invalid format"
          end
        when :inclusion
          unless constraint.include?(value)
            errors[field] ||= []
            errors[field] << "must be one of: #{constraint.join(', ')}"
          end
        when :numericality
          if constraint && value && !value.is_a?(Numeric)
            errors[field] ||= []
            errors[field] << "must be a number"
          end
        end
      end
    end

    raise ValidationError.new(errors) unless errors.empty?
    true
  end

  def valid?
    validate!
    true
  rescue ValidationError
    false
  end
end

module Serializable
  def to_h
    instance_variables.each_with_object({}) do |var, hash|
      key = var.to_s.delete('@').to_sym
      value = instance_variable_get(var)
      
      hash[key] = case value
                  when Time, Date, DateTime
                    value.iso8601
                  when Array
                    value.map { |v| v.respond_to?(:to_h) ? v.to_h : v }
                  when Hash
                    value.transform_values { |v| v.respond_to?(:to_h) ? v.to_h : v }
                  else
                    value.respond_to?(:to_h) ? value.to_h : value
                  end
    end
  end

  def to_json(*args)
    to_h.to_json(*args)
  end
end

# ============ Value Objects and Domain Models ============

class Money
  include Comparable
  
  attr_reader :amount, :currency

  def initialize(amount, currency = 'USD')
    @amount = validate_amount(amount)
    @currency = validate_currency(currency)
    freeze
  end

  def self.zero(currency = 'USD')
    new(0, currency)
  end

  def +(other)
    ensure_same_currency(other)
    Money.new(@amount + other.amount, @currency)
  end

  def -(other)
    ensure_same_currency(other)
    result = @amount - other.amount
    raise ArgumentError, "Result cannot be negative" if result < 0
    Money.new(result, @currency)
  end

  def *(multiplier)
    raise ArgumentError, "Multiplier must be numeric" unless multiplier.is_a?(Numeric)
    raise ArgumentError, "Multiplier cannot be negative" if multiplier < 0
    Money.new(@amount * multiplier, @currency)
  end

  def /(divisor)
    raise ArgumentError, "Divisor must be numeric" unless divisor.is_a?(Numeric)
    raise ArgumentError, "Cannot divide by zero" if divisor.zero?
    Money.new(@amount / divisor, @currency)
  end

  def <=>(other)
    ensure_same_currency(other)
    @amount <=> other.amount
  end

  def zero?
    @amount.zero?
  end

  def positive?
    @amount > 0
  end

  def negative?
    @amount < 0
  end

  def to_s
    format("%.2f %s", @amount, @currency)
  end

  def to_f
    @amount.to_f
  end

  private

  def validate_amount(amount)
    raise ArgumentError, "Amount must be numeric" unless amount.is_a?(Numeric)
    (amount * 100).round / 100.0  # Round to 2 decimal places
  end

  def validate_currency(currency)
    raise ArgumentError, "Currency must be a string" unless currency.is_a?(String)
    currency.upcase
  end

  def ensure_same_currency(other)
    raise ArgumentError, "Currency mismatch: #{@currency} vs #{other.currency}" unless @currency == other.currency
  end
end

class Email
  EMAIL_REGEX = /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i.freeze

  attr_reader :address

  def initialize(address)
    @address = validate_address(address)
    freeze
  end

  def domain
    @address.split('@').last
  end

  def local_part
    @address.split('@').first
  end

  def ==(other)
    other.is_a?(Email) && @address == other.address
  end

  def to_s
    @address
  end

  private

  def validate_address(address)
    raise ArgumentError, "Email address is required" if address.nil? || address.empty?
    
    normalized = address.strip.downcase
    raise ArgumentError, "Invalid email format" unless EMAIL_REGEX.match?(normalized)
    
    normalized
  end
end

# Enum-like behavior with Ruby classes
class UserRole
  ROLES = %w[guest user moderator admin super_admin].freeze

  attr_reader :name

  def initialize(name)
    @name = name.to_s.downcase
    raise ArgumentError, "Invalid role: #{@name}" unless ROLES.include?(@name)
    freeze
  end

  ROLES.each do |role|
    define_method("#{role}?") { @name == role }
    
    # Class methods for creating instances
    define_singleton_method(role) { new(role) }
  end

  def permissions
    case @name
    when 'guest'
      ['read:public']
    when 'user'
      ['read:public', 'read:own', 'write:own']
    when 'moderator'
      ['read:public', 'read:own', 'write:own', 'moderate:content']
    when 'admin'
      ['read:all', 'write:all', 'delete:all', 'manage:users']
    when 'super_admin'
      ['*']
    else
      []
    end
  end

  def can?(permission)
    permissions.include?('*') || permissions.include?(permission)
  end

  def to_s
    @name
  end

  def ==(other)
    other.is_a?(UserRole) && @name == other.name
  end
end

class TaskStatus
  STATUSES = %w[draft active in_progress review completed cancelled archived].freeze

  attr_reader :name

  def initialize(name)
    @name = name.to_s.downcase
    raise ArgumentError, "Invalid status: #{@name}" unless STATUSES.include?(@name)
    freeze
  end

  STATUSES.each do |status|
    define_method("#{status}?") { @name == status }
    define_singleton_method(status) { new(status) }
  end

  def can_transition_to?(new_status)
    transitions = {
      'draft' => %w[active cancelled],
      'active' => %w[in_progress cancelled],
      'in_progress' => %w[review active cancelled],
      'review' => %w[completed in_progress cancelled],
      'completed' => %w[archived active],
      'cancelled' => %w[active],
      'archived' => []
    }

    transitions[@name]&.include?(new_status.to_s) || false
  end

  def to_s
    @name
  end

  def ==(other)
    other.is_a?(TaskStatus) && @name == other.name
  end
end

# ============ Domain Entities ============

class User
  include Timestampable
  include Validatable
  include Serializable

  attr_accessor :id, :first_name, :last_name, :email, :role, :active, :last_login_at, :preferences, :profile

  validates :first_name, presence: true, length: { minimum: 2, maximum: 50 }
  validates :last_name, presence: true, length: { minimum: 2, maximum: 50 }
  validates :email, presence: true, format: Email::EMAIL_REGEX

  def initialize(first_name:, last_name:, email:, role: UserRole.user, **options)
    @id = options[:id] || SecureRandom.uuid
    @first_name = first_name&.strip
    @last_name = last_name&.strip
    @email = Email.new(email)
    @role = role.is_a?(UserRole) ? role : UserRole.new(role)
    @active = options.fetch(:active, true)
    @last_login_at = options[:last_login_at]
    @preferences = options[:preferences] || default_preferences
    @profile = options[:profile] || {}
    
    super()
    validate!
  end

  def self.create(**attributes)
    new(**attributes)
  end

  def full_name
    "#{@first_name} #{@last_name}"
  end

  def display_name
    @profile[:display_name] || full_name
  end

  def age
    return nil unless @profile[:date_of_birth]
    
    Date.today.year - @profile[:date_of_birth].year
  end

  def adult?
    age && age >= 18
  end

  def can?(permission)
    @active && @role.can?(permission)
  end

  def update_profile(**updates)
    @profile.merge!(updates)
    touch
    self
  end

  def change_role(new_role)
    @role = new_role.is_a?(UserRole) ? new_role : UserRole.new(new_role)
    touch
    self
  end

  def deactivate!
    @active = false
    touch
    self
  end

  def activate!
    @active = true
    touch
    self
  end

  def record_login!
    @last_login_at = Time.now
    touch
    self
  end

  def to_h
    super.merge(
      email: @email.to_s,
      role: @role.to_s,
      full_name: full_name,
      display_name: display_name,
      age: age,
      adult: adult?
    )
  end

  private

  def default_preferences
    {
      theme: 'auto',
      language: 'en',
      timezone: 'UTC',
      notifications: {
        email: true,
        push: true,
        sms: false
      }
    }
  end
end

class Task
  include Timestampable
  include Validatable
  include Serializable

  attr_accessor :id, :title, :description, :status, :priority, :assignee_id, :creator_id,
                :due_date, :completed_at, :estimated_hours, :actual_hours, :tags,
                :attachments, :comments

  validates :title, presence: true, length: { minimum: 1, maximum: 200 }
  validates :description, length: { maximum: 2000 }
  validates :priority, inclusion: [1, 2, 3, 4]
  validates :estimated_hours, numericality: true

  def initialize(title:, description: '', priority: 2, assignee_id:, creator_id:, **options)
    @id = options[:id] || rand(1_000_000)
    @title = title&.strip
    @description = description&.strip
    @status = TaskStatus.new(options[:status] || 'draft')
    @priority = priority
    @assignee_id = assignee_id
    @creator_id = creator_id
    @due_date = options[:due_date]
    @completed_at = options[:completed_at]
    @estimated_hours = options[:estimated_hours]
    @actual_hours = options[:actual_hours] || 0
    @tags = options[:tags] || []
    @attachments = options[:attachments] || []
    @comments = options[:comments] || []
    
    super()
    validate!
  end

  def self.create(**attributes)
    new(**attributes)
  end

  def update(**updates)
    updates.each do |key, value|
      public_send("#{key}=", value) if respond_to?("#{key}=")
    end
    
    validate!
    touch
    self
  end

  def change_status(new_status)
    new_status = TaskStatus.new(new_status) unless new_status.is_a?(TaskStatus)
    
    unless @status.can_transition_to?(new_status)
      raise ArgumentError, "Cannot transition from #{@status} to #{new_status}"
    end

    @status = new_status
    
    if new_status.completed?
      @completed_at = Time.now
    elsif @status.completed?
      @completed_at = nil
    end
    
    touch
    self
  end

  def add_tag(tag)
    tag = tag.strip.downcase
    @tags << tag unless @tags.include?(tag)
    touch
    self
  end

  def remove_tag(tag)
    @tags.delete(tag.strip.downcase)
    touch
    self
  end

  def add_comment(content, author_id)
    comment = {
      id: SecureRandom.uuid,
      content: content.strip,
      author_id: author_id,
      created_at: Time.now,
      updated_at: Time.now
    }
    
    @comments << comment
    touch
    self
  end

  def record_work_time(hours)
    raise ArgumentError, "Hours must be positive" unless hours > 0
    
    @actual_hours += hours
    touch
    self
  end

  def overdue?
    @due_date && @due_date < Date.today && !@status.completed?
  end

  def completed?
    @status.completed?
  end

  def on_track?
    return true unless @estimated_hours && @actual_hours
    
    @actual_hours <= @estimated_hours * 1.1
  end

  def progress_percentage
    case @status.name
    when 'draft' then 0
    when 'active' then 10
    when 'in_progress' then 50
    when 'review' then 80
    when 'completed' then 100
    when 'cancelled' then 0
    when 'archived' then 100
    else 0
    end
  end

  def time_to_completion
    return nil unless @completed_at && @created_at
    
    @completed_at - @created_at
  end

  def to_h
    super.merge(
      status: @status.to_s,
      overdue: overdue?,
      completed: completed?,
      on_track: on_track?,
      progress_percentage: progress_percentage,
      time_to_completion: time_to_completion
    )
  end
end

# ============ Repository Pattern ============

class Repository
  def initialize
    @storage = {}
    @next_id = 1
  end

  def find(id)
    @storage[id]
  end

  def find_all
    @storage.values
  end

  def save(entity)
    if entity.id.nil?
      entity.id = generate_id
    end
    
    @storage[entity.id] = entity
    entity
  end

  def delete(id)
    @storage.delete(id)
  end

  def count
    @storage.size
  end

  def exists?(id)
    @storage.key?(id)
  end

  def find_by(**criteria)
    find_all.find do |entity|
      criteria.all? { |key, value| entity.public_send(key) == value }
    end
  end

  def where(**criteria)
    find_all.select do |entity|
      criteria.all? { |key, value| entity.public_send(key) == value }
    end
  end

  private

  def generate_id
    @next_id += 1
  end
end

class UserRepository < Repository
  def find_by_email(email)
    email_obj = email.is_a?(Email) ? email : Email.new(email)
    find_all.find { |user| user.email == email_obj }
  end

  def find_by_role(role)
    role_obj = role.is_a?(UserRole) ? role : UserRole.new(role)
    where(role: role_obj)
  end

  def active_users
    where(active: true)
  end

  def search(query)
    query = query.downcase
    find_all.select do |user|
      user.full_name.downcase.include?(query) ||
        user.email.to_s.downcase.include?(query)
    end
  end
end

class TaskRepository < Repository
  def find_by_assignee(assignee_id)
    where(assignee_id: assignee_id)
  end

  def find_by_creator(creator_id)
    where(creator_id: creator_id)
  end

  def find_by_status(status)
    status_obj = status.is_a?(TaskStatus) ? status : TaskStatus.new(status)
    find_all.select { |task| task.status == status_obj }
  end

  def find_by_priority(priority)
    where(priority: priority)
  end

  def overdue_tasks
    find_all.select(&:overdue?)
  end

  def completed_tasks
    find_all.select(&:completed?)
  end

  def statistics
    tasks = find_all
    
    {
      total: tasks.size,
      by_status: group_by_status(tasks),
      by_priority: group_by_priority(tasks),
      overdue: tasks.count(&:overdue?),
      completed: tasks.count(&:completed?),
      completion_rate: tasks.empty? ? 0 : (tasks.count(&:completed?).to_f / tasks.size * 100).round(2)
    }
  end

  private

  def group_by_status(tasks)
    tasks.group_by { |task| task.status.to_s }
         .transform_values(&:size)
  end

  def group_by_priority(tasks)
    tasks.group_by(&:priority)
         .transform_values(&:size)
  end
end

# ============ Service Layer ============

class UserService
  class UserNotFoundError < StandardError; end
  class DuplicateEmailError < StandardError; end
  class AuthenticationError < StandardError; end

  def initialize(user_repository, logger = nil)
    @user_repository = user_repository
    @logger = logger || Logger.new($stdout)
  end

  def create_user(**attributes)
    @logger.info "Creating user with email: #{attributes[:email]}"
    
    # Check for duplicate email
    if @user_repository.find_by_email(attributes[:email])
      raise DuplicateEmailError, "User with email #{attributes[:email]} already exists"
    end

    user = User.create(**attributes)
    saved_user = @user_repository.save(user)
    
    @logger.info "User created successfully with ID: #{saved_user.id}"
    saved_user
  rescue Validatable::ValidationError => e
    @logger.error "User validation failed: #{e.message}"
    raise
  end

  def find_user(id)
    user = @user_repository.find(id)
    raise UserNotFoundError, "User with ID #{id} not found" unless user
    
    user
  end

  def update_user(id, **updates)
    user = find_user(id)
    
    # Check for email conflicts
    if updates[:email] && updates[:email] != user.email.to_s
      existing_user = @user_repository.find_by_email(updates[:email])
      if existing_user && existing_user.id != id
        raise DuplicateEmailError, "Email #{updates[:email]} is already taken"
      end
    end

    user.update(**updates) if updates.any?
    @user_repository.save(user)
    
    @logger.info "User #{id} updated successfully"
    user
  end

  def delete_user(id)
    user = find_user(id)
    @user_repository.delete(id)
    
    @logger.info "User #{id} deleted successfully"
    true
  end

  def authenticate_user(email, password)
    user = @user_repository.find_by_email(email)
    
    unless user&.active
      raise AuthenticationError, "Invalid credentials"
    end

    # In a real application, you would verify the password hash
    # For demo purposes, we'll assume authentication succeeds
    if verify_password(password, user)
      user.record_login!
      @user_repository.save(user)
      
      @logger.info "User #{user.id} authenticated successfully"
      user
    else
      raise AuthenticationError, "Invalid credentials"
    end
  end

  def search_users(query)
    @user_repository.search(query)
  end

  def get_active_users
    @user_repository.active_users
  end

  def get_users_by_role(role)
    @user_repository.find_by_role(role)
  end

  def change_user_role(id, new_role, current_user)
    unless current_user.can?('manage:users')
      raise ArgumentError, "Insufficient permissions to change user roles"
    end

    user = find_user(id)
    user.change_role(new_role)
    @user_repository.save(user)
    
    @logger.info "User #{id} role changed to #{new_role} by #{current_user.id}"
    user
  end

  private

  def verify_password(password, user)
    # Simplified password verification for demo
    # In real application, use bcrypt or similar
    !password.nil? && !password.empty?
  end
end

class TaskService
  class TaskNotFoundError < StandardError; end
  class InsufficientPermissionsError < StandardError; end

  def initialize(task_repository, user_repository, logger = nil)
    @task_repository = task_repository
    @user_repository = user_repository
    @logger = logger || Logger.new($stdout)
  end

  def create_task(**attributes)
    @logger.info "Creating task: #{attributes[:title]}"
    
    # Validate that assignee and creator exist
    assignee = @user_repository.find(attributes[:assignee_id])
    creator = @user_repository.find(attributes[:creator_id])
    
    raise UserService::UserNotFoundError, "Assignee not found" unless assignee
    raise UserService::UserNotFoundError, "Creator not found" unless creator

    task = Task.create(**attributes)
    saved_task = @task_repository.save(task)
    
    @logger.info "Task created successfully with ID: #{saved_task.id}"
    saved_task
  rescue Validatable::ValidationError => e
    @logger.error "Task validation failed: #{e.message}"
    raise
  end

  def find_task(id)
    task = @task_repository.find(id)
    raise TaskNotFoundError, "Task with ID #{id} not found" unless task
    
    task
  end

  def update_task(id, current_user, **updates)
    task = find_task(id)
    
    unless can_modify_task?(task, current_user)
      raise InsufficientPermissionsError, "You don't have permission to modify this task"
    end

    task.update(**updates) if updates.any?
    @task_repository.save(task)
    
    @logger.info "Task #{id} updated successfully by user #{current_user.id}"
    task
  end

  def change_task_status(id, new_status, current_user)
    task = find_task(id)
    
    unless can_modify_task?(task, current_user)
      raise InsufficientPermissionsError, "You don't have permission to modify this task"
    end

    task.change_status(new_status)
    @task_repository.save(task)
    
    @logger.info "Task #{id} status changed to #{new_status} by user #{current_user.id}"
    task
  end

  def delete_task(id, current_user)
    task = find_task(id)
    
    unless can_delete_task?(task, current_user)
      raise InsufficientPermissionsError, "You don't have permission to delete this task"
    end

    @task_repository.delete(id)
    
    @logger.info "Task #{id} deleted successfully by user #{current_user.id}"
    true
  end

  def get_tasks_by_user(user_id)
    @task_repository.find_by_assignee(user_id)
  end

  def get_tasks_by_status(status)
    @task_repository.find_by_status(status)
  end

  def get_overdue_tasks
    @task_repository.overdue_tasks
  end

  def get_task_statistics
    @task_repository.statistics
  end

  def add_work_time(task_id, hours, current_user)
    task = find_task(task_id)
    
    unless can_modify_task?(task, current_user)
      raise InsufficientPermissionsError, "You don't have permission to modify this task"
    end

    task.record_work_time(hours)
    @task_repository.save(task)
    
    @logger.info "#{hours} hours recorded for task #{task_id} by user #{current_user.id}"
    task
  end

  private

  def can_modify_task?(task, user)
    task.assignee_id == user.id ||
      task.creator_id == user.id ||
      user.can?('write:all')
  end

  def can_delete_task?(task, user)
    task.creator_id == user.id ||
      user.can?('delete:all')
  end
end

# ============ Metaprogramming and DSL Examples ============

# Simple DSL for configuration
class Configuration
  def initialize(&block)
    @settings = {}
    instance_eval(&block) if block_given?
  end

  def method_missing(name, *args)
    if args.length == 1
      @settings[name] = args.first
    elsif args.empty?
      @settings[name]
    else
      super
    end
  end

  def respond_to_missing?(name, include_private = false)
    true
  end

  def to_h
    @settings.dup
  end
end

# Query builder DSL
class QueryBuilder
  def initialize(repository)
    @repository = repository
    @conditions = {}
    @limit_value = nil
    @offset_value = nil
    @order_by = nil
  end

  def where(**conditions)
    @conditions.merge!(conditions)
    self
  end

  def limit(value)
    @limit_value = value
    self
  end

  def offset(value)
    @offset_value = value
    self
  end

  def order(field)
    @order_by = field
    self
  end

  def all
    results = @repository.find_all

    # Apply conditions
    unless @conditions.empty?
      results = results.select do |entity|
        @conditions.all? { |key, value| entity.public_send(key) == value }
      end
    end

    # Apply ordering
    if @order_by
      results = results.sort_by { |entity| entity.public_send(@order_by) }
    end

    # Apply offset
    if @offset_value
      results = results.drop(@offset_value)
    end

    # Apply limit
    if @limit_value
      results = results.take(@limit_value)
    end

    results
  end

  def first
    limit(1).all.first
  end

  def count
    # Apply conditions only for count
    results = @repository.find_all

    unless @conditions.empty?
      results = results.select do |entity|
        @conditions.all? { |key, value| entity.public_send(key) == value }
      end
    end

    results.size
  end
end

# Module for adding query methods to repositories
module Queryable
  def self.included(base)
    base.extend(ClassMethods)
  end

  module ClassMethods
    def scope(name, lambda)
      define_method(name) do |*args|
        QueryBuilder.new(self).instance_exec(*args, &lambda)
      end
    end
  end

  def query
    QueryBuilder.new(self)
  end
end

# Extend repositories with query capabilities
class UserRepository
  include Queryable

  scope :active, -> { where(active: true) }
  scope :by_role, ->(role) { where(role: role) }
  scope :adults, -> { 
    find_all.select { |user| user.adult? }
  }
end

class TaskRepository
  include Queryable

  scope :active, -> { where(status: TaskStatus.active) }
  scope :completed, -> { where(status: TaskStatus.completed) }
  scope :high_priority, -> { where(priority: 4) }
  scope :overdue, -> { 
    find_all.select(&:overdue?)
  }
end

# ============ Concurrent Programming ============

class AsyncTaskProcessor
  def initialize(thread_pool_size: 5)
    @executor = Concurrent::ThreadPoolExecutor.new(
      min_threads: 1,
      max_threads: thread_pool_size,
      max_queue: 100
    )
  end

  def process_tasks_async(tasks, &block)
    futures = tasks.map do |task|
      Concurrent::Future.execute(executor: @executor) do
        block.call(task)
      end
    end

    # Return a future that completes when all tasks are done
    Concurrent::Future.new(executor: @executor) do
      futures.map(&:value)
    end
  end

  def process_with_timeout(task, timeout_seconds, &block)
    future = Concurrent::Future.execute(executor: @executor) do
      block.call(task)
    end

    begin
      future.value(timeout_seconds)
    rescue Concurrent::TimeoutError
      future.cancel
      raise "Task processing timed out after #{timeout_seconds} seconds"
    end
  end

  def shutdown
    @executor.shutdown
    @executor.wait_for_termination(10)
  end
end

# ============ Functional Programming Utilities ============

module FunctionalUtils
  def self.curry(method)
    proc do |*args|
      if args.length >= method.arity
        method.call(*args)
      else
        curry(proc { |*more_args| method.call(*(args + more_args)) })
      end
    end
  end

  def self.compose(*functions)
    proc do |value|
      functions.reverse.reduce(value) { |acc, func| func.call(acc) }
    end
  end

  def self.pipe(value, *functions)
    functions.reduce(value) { |acc, func| func.call(acc) }
  end

  def self.partial(method, *partial_args)
    proc { |*remaining_args| method.call(*(partial_args + remaining_args)) }
  end
end

# Add functional methods to Array
class Array
  def fmap(&block)
    map(&block)
  end

  def flat_map(&block)
    map(&block).flatten(1)
  end

  def fold_left(initial, &block)
    reduce(initial, &block)
  end

  def fold_right(initial, &block)
    reverse.reduce(initial) { |acc, item| block.call(item, acc) }
  end

  def take_while(&block)
    result = []
    each do |item|
      break unless block.call(item)
      result << item
    end
    result
  end

  def drop_while(&block)
    drop_count = 0
    each do |item|
      break unless block.call(item)
      drop_count += 1
    end
    drop(drop_count)
  end

  def group_by_count(&block)
    group_by(&block).transform_values(&:size)
  end
end

# ============ Decorator Pattern ============

class LoggingDecorator
  def initialize(target, logger = Logger.new($stdout))
    @target = target
    @logger = logger
  end

  def method_missing(method_name, *args, &block)
    @logger.info "Calling #{@target.class}##{method_name} with args: #{args.inspect}"
    
    start_time = Time.now
    result = @target.public_send(method_name, *args, &block)
    end_time = Time.now
    
    @logger.info "#{@target.class}##{method_name} completed in #{(end_time - start_time).round(3)}s"
    result
  rescue => e
    @logger.error "#{@target.class}##{method_name} failed: #{e.message}"
    raise
  end

  def respond_to_missing?(method_name, include_private = false)
    @target.respond_to?(method_name, include_private)
  end
end

class CachingDecorator
  def initialize(target, ttl_seconds = 300)
    @target = target
    @cache = {}
    @ttl_seconds = ttl_seconds
  end

  def method_missing(method_name, *args, &block)
    cache_key = [method_name, args].hash
    
    if cached_result = @cache[cache_key]
      if Time.now - cached_result[:timestamp] < @ttl_seconds
        return cached_result[:value]
      end
    end

    result = @target.public_send(method_name, *args, &block)
    
    @cache[cache_key] = {
      value: result,
      timestamp: Time.now
    }
    
    result
  end

  def respond_to_missing?(method_name, include_private = false)
    @target.respond_to?(method_name, include_private)
  end

  def clear_cache
    @cache.clear
  end
end

# ============ Demo Application ============

def run_demo
  puts "=== Ruby Comprehensive Examples Demo ===\n\n"

  # Setup
  user_repo = UserRepository.new
  task_repo = TaskRepository.new
  logger = Logger.new($stdout)
  logger.level = Logger::INFO

  user_service = LoggingDecorator.new(
    CachingDecorator.new(
      UserService.new(user_repo, logger)
    )
  )
  
  task_service = TaskService.new(task_repo, user_repo, logger)

  begin
    # Create users
    puts "Creating users..."
    
    admin = user_service.create_user(
      first_name: 'Admin',
      last_name: 'User',
      email: 'admin@example.com',
      role: UserRole.admin
    )

    alice = user_service.create_user(
      first_name: 'Alice',
      last_name: 'Johnson',
      email: 'alice@example.com',
      role: UserRole.user,
      profile: { date_of_birth: Date.new(1990, 5, 15) }
    )

    bob = user_service.create_user(
      first_name: 'Bob',
      last_name: 'Smith',
      email: 'bob@example.com',
      role: UserRole.user,
      profile: { date_of_birth: Date.new(1985, 10, 20) }
    )

    puts "✅ Created #{user_service.get_active_users.size} users\n\n"

    # Create tasks
    puts "Creating tasks..."
    
    task1 = task_service.create_task(
      title: 'Implement user authentication',
      description: 'Add secure login system with JWT tokens',
      priority: 4,
      assignee_id: alice.id,
      creator_id: admin.id,
      due_date: Date.today + 7
    )

    task2 = task_service.create_task(
      title: 'Design database schema',
      description: 'Create comprehensive database design',
      priority: 3,
      assignee_id: bob.id,
      creator_id: admin.id,
      estimated_hours: 8
    )

    task3 = task_service.create_task(
      title: 'Write API documentation',
      description: 'Document all REST API endpoints',
      priority: 2,
      assignee_id: alice.id,
      creator_id: admin.id,
      due_date: Date.today + 14
    )

    puts "✅ Created 3 tasks\n\n"

    # Update task statuses
    puts "Updating task statuses..."
    task_service.change_task_status(task1.id, TaskStatus.in_progress, alice)
    task_service.change_task_status(task2.id, TaskStatus.completed, bob)

    # Add work time
    task_service.add_work_time(task1.id, 4.5, alice)
    task_service.add_work_time(task2.id, 6.0, bob)

    # Get statistics
    puts "\n📊 Task Statistics:"
    stats = task_service.get_task_statistics
    stats.each { |key, value| puts "  #{key}: #{value}" }

    # Test search and queries
    puts "\n🔍 Testing search and queries..."
    search_results = user_service.search_users('alice')
    puts "Found #{search_results.size} users matching 'alice'"

    # Test query DSL
    active_users = user_repo.active.all
    puts "Active users: #{active_users.size}"

    high_priority_tasks = task_repo.high_priority.all
    puts "High priority tasks: #{high_priority_tasks.size}"

    # Test value objects
    puts "\n💰 Testing value objects..."
    money1 = Money.new(100.50)
    money2 = Money.new(25.25)
    total = money1 + money2
    puts "Money calculation: #{money1} + #{money2} = #{total}"

    email = Email.new('test@example.com')
    puts "Email domain: #{email.domain}"
    puts "Email local part: #{email.local_part}"

    # Test functional programming
    puts "\n🔧 Testing functional programming..."
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    result = FunctionalUtils.pipe(
      numbers,
      ->(arr) { arr.select(&:even?) },
      ->(arr) { arr.map { |n| n * 2 } },
      ->(arr) { arr.reduce(:+) }
    )
    puts "Functional pipeline result: #{result}"

    grouped_tasks = task_repo.find_all.group_by_count { |task| task.status.to_s }
    puts "Tasks grouped by status: #{grouped_tasks}"

    # Test concurrent processing
    puts "\n⚡ Testing concurrent processing..."
    processor = AsyncTaskProcessor.new(thread_pool_size: 3)
    
    tasks = task_repo.find_all
    future = processor.process_tasks_async(tasks) do |task|
      # Simulate some processing
      sleep(0.1)
      "Processed task: #{task.title}"
    end

    results = future.value(5) # Wait up to 5 seconds
    puts "Processed #{results.size} tasks concurrently"
    
    processor.shutdown

    # Test configuration DSL
    puts "\n⚙️  Testing configuration DSL..."
    config = Configuration.new do
      app_name 'Task Manager'
      database_url 'postgresql://localhost/taskmanager'
      redis_url 'redis://localhost:6379'
      log_level 'info'
      max_connections 100
    end

    puts "Configuration: #{config.to_h}"

    # Test metaprogramming
    puts "\n🪄 Testing metaprogramming..."
    puts "User role permissions:"
    UserRole::ROLES.each do |role|
      role_obj = UserRole.new(role)
      puts "  #{role}: #{role_obj.permissions}"
    end

    # Test serialization
    puts "\n📄 Testing serialization..."
    user_hash = alice.to_h
    puts "User as hash: #{user_hash.keys.join(', ')}"

    task_json = task1.to_json
    puts "Task as JSON (length): #{task_json.length} characters"

  rescue => e
    puts "❌ Demo error: #{e.message}"
    puts e.backtrace.first(5)
  end

  puts "\n=== Ruby Features Demonstrated ==="
  puts "🔥 Modules and mixins for code reuse"
  puts "💎 Ruby's object model and metaprogramming"
  puts "🎭 Method missing and dynamic method definition"
  puts "🏗️  Domain-specific languages (DSLs)"
  puts "🔧 Functional programming utilities"
  puts "⚡ Concurrent programming with concurrent-ruby"
  puts "🎨 Decorator pattern with method_missing"
  puts "📊 Repository pattern with query capabilities"
  puts "✅ Validation with custom error handling"
  puts "💰 Value objects with immutability"
  puts "🚀 Service layer with business logic"
  puts "📄 Serialization and data transformation"
  puts "🎯 Ruby idioms and best practices"
  puts "🔍 Advanced enumerable operations"
end

# Run the demo
if __FILE__ == $0
  run_demo
end