/*
 * Advanced C# Programming Examples for AI Coding Agents
 * =====================================================
 * 
 * This module demonstrates advanced C# features including:
 * - Modern C# 12 features and patterns
 * - LINQ and functional programming
 * - Async/await and Task-based programming
 * - Dependency injection and IoC containers
 * - Entity Framework and database patterns
 * - ASP.NET Core web development patterns
 * - Memory management and performance optimization
 * - Reflection and dynamic programming
 * - Exception handling and error patterns
 * - Testing patterns and frameworks
 * 
 * Author: AI Dataset Creation Team
 * License: MIT
 * Created: 2024
 */

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.ComponentModel.DataAnnotations;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Diagnostics;
using System.IO;
using System.Net.Http;

namespace AdvancedCSharpExamples
{
    // =============================================================================
    // Modern C# 12 Features and Patterns
    // =============================================================================

    /// <summary>
    /// Record types for immutable data structures with value equality
    /// </summary>
    public sealed record PersonRecord(
        string FirstName,
        string LastName,
        DateTime BirthDate,
        string Email)
    {
        /// <summary>
        /// Computed property using C# expression syntax
        /// </summary>
        public string FullName => $"{FirstName} {LastName}";
        
        /// <summary>
        /// Age calculation with null-conditional operator
        /// </summary>
        public int Age => DateTime.Now.Year - BirthDate.Year;
        
        /// <summary>
        /// Pattern matching with when clause
        /// </summary>
        public string AgeCategory => Age switch
        {
            < 18 => "Minor",
            >= 18 and < 65 => "Adult",
            >= 65 => "Senior",
            _ => "Unknown"
        };
        
        /// <summary>
        /// Validation using data annotations pattern
        /// </summary>
        public bool IsValid()
        {
            var context = new ValidationContext(this);
            var results = new List<ValidationResult>();
            return Validator.TryValidateObject(this, context, results, true);
        }
    }

    /// <summary>
    /// Generic repository pattern with constraints and covariance
    /// </summary>
    public interface IRepository<T> where T : class, IEntity
    {
        Task<T?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default);
        Task<IEnumerable<T>> GetAllAsync(CancellationToken cancellationToken = default);
        Task<T> AddAsync(T entity, CancellationToken cancellationToken = default);
        Task<T> UpdateAsync(T entity, CancellationToken cancellationToken = default);
        Task<bool> DeleteAsync(Guid id, CancellationToken cancellationToken = default);
        Task<IEnumerable<T>> FindAsync(
            System.Linq.Expressions.Expression<Func<T, bool>> predicate,
            CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Base entity interface for repository pattern
    /// </summary>
    public interface IEntity
    {
        Guid Id { get; set; }
        DateTime CreatedAt { get; set; }
        DateTime? UpdatedAt { get; set; }
    }

    /// <summary>
    /// In-memory repository implementation with thread safety
    /// </summary>
    public class InMemoryRepository<T> : IRepository<T> where T : class, IEntity
    {
        private readonly ConcurrentDictionary<Guid, T> _entities = new();
        private readonly object _lock = new();

        public Task<T?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            _entities.TryGetValue(id, out var entity);
            return Task.FromResult(entity);
        }

        public Task<IEnumerable<T>> GetAllAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            return Task.FromResult(_entities.Values.AsEnumerable());
        }

        public Task<T> AddAsync(T entity, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            ArgumentNullException.ThrowIfNull(entity);
            
            lock (_lock)
            {
                if (entity.Id == Guid.Empty)
                    entity.Id = Guid.NewGuid();
                
                entity.CreatedAt = DateTime.UtcNow;
                _entities.TryAdd(entity.Id, entity);
            }
            
            return Task.FromResult(entity);
        }

        public Task<T> UpdateAsync(T entity, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            ArgumentNullException.ThrowIfNull(entity);
            
            lock (_lock)
            {
                if (_entities.ContainsKey(entity.Id))
                {
                    entity.UpdatedAt = DateTime.UtcNow;
                    _entities.TryUpdate(entity.Id, entity, _entities[entity.Id]);
                }
                else
                {
                    throw new InvalidOperationException($"Entity with ID {entity.Id} not found");
                }
            }
            
            return Task.FromResult(entity);
        }

        public Task<bool> DeleteAsync(Guid id, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            return Task.FromResult(_entities.TryRemove(id, out _));
        }

        public Task<IEnumerable<T>> FindAsync(
            System.Linq.Expressions.Expression<Func<T, bool>> predicate,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            ArgumentNullException.ThrowIfNull(predicate);
            
            var compiled = predicate.Compile();
            var results = _entities.Values.Where(compiled);
            
            return Task.FromResult(results);
        }
    }

    // =============================================================================
    // LINQ and Functional Programming Patterns
    // =============================================================================

    /// <summary>
    /// Extension methods for functional programming patterns
    /// </summary>
    public static class FunctionalExtensions
    {
        /// <summary>
        /// Maybe monad for null-safe operations
        /// </summary>
        public static Maybe<T> ToMaybe<T>(this T? value) where T : class
        {
            return value is null ? Maybe<T>.None() : Maybe<T>.Some(value);
        }

        /// <summary>
        /// Safe dictionary access
        /// </summary>
        public static Maybe<TValue> TryGetValue<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary, 
            TKey key) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) 
                ? Maybe<TValue>.Some(value) 
                : Maybe<TValue>.None();
        }

        /// <summary>
        /// Batch processing for IEnumerable
        /// </summary>
        public static IEnumerable<IEnumerable<T>> Batch<T>(
            this IEnumerable<T> source, 
            int batchSize)
        {
            ArgumentNullException.ThrowIfNull(source);
            
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive", nameof(batchSize));

            var batch = new List<T>(batchSize);
            
            foreach (var item in source)
            {
                batch.Add(item);
                
                if (batch.Count == batchSize)
                {
                    yield return batch.ToList(); // Create defensive copy
                    batch.Clear();
                }
            }
            
            if (batch.Count > 0)
                yield return batch;
        }

        /// <summary>
        /// Pipe operation for functional composition
        /// </summary>
        public static TResult Pipe<T, TResult>(this T value, Func<T, TResult> func)
        {
            ArgumentNullException.ThrowIfNull(func);
            return func(value);
        }

        /// <summary>
        /// Tap operation for side effects without changing the pipeline
        /// </summary>
        public static T Tap<T>(this T value, Action<T> action)
        {
            ArgumentNullException.ThrowIfNull(action);
            action(value);
            return value;
        }
    }

    /// <summary>
    /// Maybe monad implementation for null-safe operations
    /// </summary>
    public readonly struct Maybe<T> : IEquatable<Maybe<T>>
    {
        private readonly T? _value;
        private readonly bool _hasValue;

        private Maybe(T value)
        {
            _value = value;
            _hasValue = true;
        }

        public static Maybe<T> Some(T value) => new(value);
        public static Maybe<T> None() => default;

        public bool HasValue => _hasValue;
        public T Value => _hasValue ? _value! : throw new InvalidOperationException("Maybe has no value");

        /// <summary>
        /// Map operation for transforming the contained value
        /// </summary>
        public Maybe<TResult> Map<TResult>(Func<T, TResult> mapper)
        {
            ArgumentNullException.ThrowIfNull(mapper);
            
            return _hasValue 
                ? Maybe<TResult>.Some(mapper(_value!)) 
                : Maybe<TResult>.None();
        }

        /// <summary>
        /// Bind operation for chaining Maybe operations
        /// </summary>
        public Maybe<TResult> Bind<TResult>(Func<T, Maybe<TResult>> binder)
        {
            ArgumentNullException.ThrowIfNull(binder);
            
            return _hasValue ? binder(_value!) : Maybe<TResult>.None();
        }

        /// <summary>
        /// Filter operation based on predicate
        /// </summary>
        public Maybe<T> Filter(Predicate<T> predicate)
        {
            ArgumentNullException.ThrowIfNull(predicate);
            
            return _hasValue && predicate(_value!) ? this : None();
        }

        /// <summary>
        /// Get value or default
        /// </summary>
        public T GetValueOrDefault(T defaultValue = default!) =>
            _hasValue ? _value! : defaultValue;

        /// <summary>
        /// Execute action if value exists
        /// </summary>
        public Maybe<T> IfSome(Action<T> action)
        {
            if (_hasValue && action is not null)
                action(_value!);
            
            return this;
        }

        public bool Equals(Maybe<T> other) =>
            _hasValue == other._hasValue && 
            (!_hasValue || EqualityComparer<T>.Default.Equals(_value, other._value));

        public override bool Equals(object? obj) =>
            obj is Maybe<T> other && Equals(other);

        public override int GetHashCode() =>
            _hasValue ? EqualityComparer<T>.Default.GetHashCode(_value!) : 0;

        public static bool operator ==(Maybe<T> left, Maybe<T> right) => left.Equals(right);
        public static bool operator !=(Maybe<T> left, Maybe<T> right) => !left.Equals(right);
    }

    /// <summary>
    /// Advanced LINQ query builder with fluent interface
    /// </summary>
    public class QueryBuilder<T>
    {
        private IQueryable<T> _query;

        public QueryBuilder(IQueryable<T> source)
        {
            _query = source ?? throw new ArgumentNullException(nameof(source));
        }

        public QueryBuilder<T> Where(System.Linq.Expressions.Expression<Func<T, bool>> predicate)
        {
            _query = _query.Where(predicate);
            return this;
        }

        public QueryBuilder<TResult> Select<TResult>(
            System.Linq.Expressions.Expression<Func<T, TResult>> selector)
        {
            return new QueryBuilder<TResult>(_query.Select(selector));
        }

        public QueryBuilder<T> OrderBy<TKey>(
            System.Linq.Expressions.Expression<Func<T, TKey>> keySelector)
        {
            _query = _query.OrderBy(keySelector);
            return this;
        }

        public QueryBuilder<T> OrderByDescending<TKey>(
            System.Linq.Expressions.Expression<Func<T, TKey>> keySelector)
        {
            _query = _query.OrderByDescending(keySelector);
            return this;
        }

        public QueryBuilder<T> Take(int count)
        {
            _query = _query.Take(count);
            return this;
        }

        public QueryBuilder<T> Skip(int count)
        {
            _query = _query.Skip(count);
            return this;
        }

        public async Task<List<T>> ToListAsync(CancellationToken cancellationToken = default)
        {
            // In a real implementation, this would use Entity Framework's ToListAsync
            return await Task.FromResult(_query.ToList());
        }

        public async Task<T?> FirstOrDefaultAsync(CancellationToken cancellationToken = default)
        {
            return await Task.FromResult(_query.FirstOrDefault());
        }

        public async Task<int> CountAsync(CancellationToken cancellationToken = default)
        {
            return await Task.FromResult(_query.Count());
        }

        public IQueryable<T> Build() => _query;
    }

    // =============================================================================
    // Async/Await and Task-Based Programming
    // =============================================================================

    /// <summary>
    /// Advanced HTTP client with retry policies and circuit breaker
    /// </summary>
    public class ResilientHttpClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly SemaphoreSlim _circuitBreakerSemaphore;
        private readonly TimeSpan _circuitBreakerTimeout;
        private DateTime _circuitBreakerLastFailure;
        private int _circuitBreakerFailureCount;
        private bool _disposed;

        public ResilientHttpClient(HttpClient httpClient, TimeSpan circuitBreakerTimeout)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _circuitBreakerTimeout = circuitBreakerTimeout;
            _circuitBreakerSemaphore = new SemaphoreSlim(1, 1);
        }

        /// <summary>
        /// GET request with exponential backoff retry
        /// </summary>
        public async Task<HttpResponseMessage> GetWithRetryAsync(
            string requestUri,
            int maxRetries = 3,
            CancellationToken cancellationToken = default)
        {
            var retryCount = 0;
            
            while (retryCount <= maxRetries)
            {
                try
                {
                    if (await IsCircuitOpenAsync())
                    {
                        throw new InvalidOperationException("Circuit breaker is open");
                    }

                    var response = await _httpClient.GetAsync(requestUri, cancellationToken);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        await ResetCircuitBreakerAsync();
                        return response;
                    }
                    
                    if (retryCount == maxRetries)
                    {
                        await RecordFailureAsync();
                        return response;
                    }
                }
                catch (HttpRequestException) when (retryCount < maxRetries)
                {
                    await RecordFailureAsync();
                }
                catch (TaskCanceledException) when (retryCount < maxRetries)
                {
                    await RecordFailureAsync();
                }

                retryCount++;
                var delay = TimeSpan.FromSeconds(Math.Pow(2, retryCount - 1));
                await Task.Delay(delay, cancellationToken);
            }

            throw new InvalidOperationException($"Request failed after {maxRetries} retries");
        }

        /// <summary>
        /// POST request with JSON serialization
        /// </summary>
        public async Task<TResponse?> PostJsonAsync<TRequest, TResponse>(
            string requestUri,
            TRequest request,
            CancellationToken cancellationToken = default)
            where TRequest : class
            where TResponse : class
        {
            ArgumentNullException.ThrowIfNull(request);

            var json = JsonSerializer.Serialize(request, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = false
            });

            using var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            
            var response = await _httpClient.PostAsync(requestUri, content, cancellationToken);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            
            return JsonSerializer.Deserialize<TResponse>(responseJson, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
        }

        /// <summary>
        /// Parallel batch processing with concurrency limit
        /// </summary>
        public async Task<IEnumerable<TResult>> ProcessBatchAsync<TInput, TResult>(
            IEnumerable<TInput> inputs,
            Func<TInput, CancellationToken, Task<TResult>> processor,
            int maxConcurrency = 5,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            ArgumentNullException.ThrowIfNull(processor);

            using var semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
            var tasks = inputs.Select(async input =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    return await processor(input, cancellationToken);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            return await Task.WhenAll(tasks);
        }

        private async Task<bool> IsCircuitOpenAsync()
        {
            await _circuitBreakerSemaphore.WaitAsync();
            try
            {
                return _circuitBreakerFailureCount >= 5 &&
                       DateTime.UtcNow - _circuitBreakerLastFailure < _circuitBreakerTimeout;
            }
            finally
            {
                _circuitBreakerSemaphore.Release();
            }
        }

        private async Task RecordFailureAsync()
        {
            await _circuitBreakerSemaphore.WaitAsync();
            try
            {
                _circuitBreakerFailureCount++;
                _circuitBreakerLastFailure = DateTime.UtcNow;
            }
            finally
            {
                _circuitBreakerSemaphore.Release();
            }
        }

        private async Task ResetCircuitBreakerAsync()
        {
            await _circuitBreakerSemaphore.WaitAsync();
            try
            {
                _circuitBreakerFailureCount = 0;
            }
            finally
            {
                _circuitBreakerSemaphore.Release();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _circuitBreakerSemaphore?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Task coordination and workflow management
    /// </summary>
    public class TaskCoordinator
    {
        /// <summary>
        /// Execute tasks with timeout and cancellation
        /// </summary>
        public static async Task<T> WithTimeout<T>(
            Task<T> task,
            TimeSpan timeout,
            CancellationToken cancellationToken = default)
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var combinedCts = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, timeoutCts.Token);

            try
            {
                return await task.WaitAsync(combinedCts.Token);
            }
            catch (OperationCanceledException) when (timeoutCts.Token.IsCancellationRequested)
            {
                throw new TimeoutException($"Task timed out after {timeout}");
            }
        }

        /// <summary>
        /// Fan-out/fan-in pattern for parallel processing
        /// </summary>
        public static async Task<TResult> FanOutFanIn<TInput, TIntermediate, TResult>(
            TInput input,
            IEnumerable<Func<TInput, Task<TIntermediate>>> processors,
            Func<IEnumerable<TIntermediate>, TResult> aggregator,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(processors);
            ArgumentNullException.ThrowIfNull(aggregator);

            var tasks = processors.Select(processor => processor(input));
            var results = await Task.WhenAll(tasks);
            
            return aggregator(results);
        }

        /// <summary>
        /// Producer-consumer pattern with async enumerable
        /// </summary>
        public static async IAsyncEnumerable<T> ProduceAsync<T>(
            IEnumerable<Func<Task<T>>> producers,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var channel = System.Threading.Channels.Channel.CreateUnbounded<T>();
            var writer = channel.Writer;

            // Start all producers
            var producerTasks = producers.Select(async producer =>
            {
                try
                {
                    var result = await producer();
                    await writer.WriteAsync(result, cancellationToken);
                }
                catch (Exception ex)
                {
                    // Log error in real implementation
                    Console.WriteLine($"Producer error: {ex.Message}");
                }
            });

            // Complete the writer when all producers finish
            _ = Task.Run(async () =>
            {
                await Task.WhenAll(producerTasks);
                writer.Complete();
            }, cancellationToken);

            // Yield results as they become available
            await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken))
            {
                yield return item;
            }
        }
    }

    // =============================================================================
    // Dependency Injection and IoC Patterns
    // =============================================================================

    /// <summary>
    /// Simple dependency injection container
    /// </summary>
    public class SimpleContainer
    {
        private readonly ConcurrentDictionary<Type, Func<SimpleContainer, object>> _services = new();
        private readonly ConcurrentDictionary<Type, object> _singletons = new();

        /// <summary>
        /// Register a transient service
        /// </summary>
        public void RegisterTransient<TInterface, TImplementation>()
            where TImplementation : class, TInterface
        {
            _services.TryAdd(typeof(TInterface), container => CreateInstance<TImplementation>(container));
        }

        /// <summary>
        /// Register a singleton service
        /// </summary>
        public void RegisterSingleton<TInterface, TImplementation>()
            where TImplementation : class, TInterface
        {
            _services.TryAdd(typeof(TInterface), container =>
            {
                return _singletons.GetOrAdd(typeof(TInterface), 
                    _ => CreateInstance<TImplementation>(container));
            });
        }

        /// <summary>
        /// Register a singleton instance
        /// </summary>
        public void RegisterInstance<TInterface>(TInterface instance)
        {
            ArgumentNullException.ThrowIfNull(instance);
            _singletons.TryAdd(typeof(TInterface), instance);
            _services.TryAdd(typeof(TInterface), _ => instance);
        }

        /// <summary>
        /// Resolve a service from the container
        /// </summary>
        public T Resolve<T>()
        {
            return (T)Resolve(typeof(T));
        }

        /// <summary>
        /// Resolve a service by type
        /// </summary>
        public object Resolve(Type serviceType)
        {
            if (_services.TryGetValue(serviceType, out var factory))
            {
                return factory(this);
            }

            throw new InvalidOperationException($"Service of type {serviceType.Name} is not registered");
        }

        private T CreateInstance<T>(SimpleContainer container)
        {
            var type = typeof(T);
            var constructors = type.GetConstructors();
            var constructor = constructors.OrderByDescending(c => c.GetParameters().Length).First();

            var parameters = constructor.GetParameters()
                .Select(p => container.Resolve(p.ParameterType))
                .ToArray();

            return (T)Activator.CreateInstance(type, parameters)!;
        }
    }

    // =============================================================================
    // Example Domain Models and Services
    // =============================================================================

    /// <summary>
    /// Example entity for demonstrating patterns
    /// </summary>
    public class Customer : IEntity
    {
        public Guid Id { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
        
        [Required]
        [StringLength(100)]
        public string FirstName { get; set; } = string.Empty;
        
        [Required]
        [StringLength(100)]
        public string LastName { get; set; } = string.Empty;
        
        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;
        
        public DateTime? DateOfBirth { get; set; }
        
        public List<Order> Orders { get; set; } = new();
    }

    /// <summary>
    /// Related entity demonstrating navigation properties
    /// </summary>
    public class Order : IEntity
    {
        public Guid Id { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
        
        public Guid CustomerId { get; set; }
        public Customer Customer { get; set; } = null!;
        
        public decimal TotalAmount { get; set; }
        public DateTime OrderDate { get; set; }
        public OrderStatus Status { get; set; }
        
        public List<OrderItem> Items { get; set; } = new();
    }

    public class OrderItem : IEntity
    {
        public Guid Id { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
        
        public Guid OrderId { get; set; }
        public Order Order { get; set; } = null!;
        
        public string ProductName { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal UnitPrice { get; set; }
        public decimal Total => Quantity * UnitPrice;
    }

    public enum OrderStatus
    {
        Pending,
        Processing,
        Shipped,
        Delivered,
        Cancelled
    }

    /// <summary>
    /// Service interface for business logic
    /// </summary>
    public interface ICustomerService
    {
        Task<Customer?> GetCustomerByIdAsync(Guid customerId, CancellationToken cancellationToken = default);
        Task<IEnumerable<Customer>> GetCustomersAsync(CancellationToken cancellationToken = default);
        Task<Customer> CreateCustomerAsync(Customer customer, CancellationToken cancellationToken = default);
        Task<Customer> UpdateCustomerAsync(Customer customer, CancellationToken cancellationToken = default);
        Task<bool> DeleteCustomerAsync(Guid customerId, CancellationToken cancellationToken = default);
        Task<IEnumerable<Customer>> FindCustomersByEmailDomainAsync(string domain, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Customer service implementation with business logic
    /// </summary>
    public class CustomerService : ICustomerService
    {
        private readonly IRepository<Customer> _customerRepository;
        private readonly IRepository<Order> _orderRepository;

        public CustomerService(IRepository<Customer> customerRepository, IRepository<Order> orderRepository)
        {
            _customerRepository = customerRepository ?? throw new ArgumentNullException(nameof(customerRepository));
            _orderRepository = orderRepository ?? throw new ArgumentNullException(nameof(orderRepository));
        }

        public async Task<Customer?> GetCustomerByIdAsync(Guid customerId, CancellationToken cancellationToken = default)
        {
            var customer = await _customerRepository.GetByIdAsync(customerId, cancellationToken);
            
            if (customer is not null)
            {
                // Load related orders
                var orders = await _orderRepository.FindAsync(o => o.CustomerId == customerId, cancellationToken);
                customer.Orders = orders.ToList();
            }
            
            return customer;
        }

        public async Task<IEnumerable<Customer>> GetCustomersAsync(CancellationToken cancellationToken = default)
        {
            return await _customerRepository.GetAllAsync(cancellationToken);
        }

        public async Task<Customer> CreateCustomerAsync(Customer customer, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(customer);
            
            // Validate business rules
            if (string.IsNullOrWhiteSpace(customer.Email))
                throw new ArgumentException("Email is required", nameof(customer));
            
            // Check for duplicate email
            var existingCustomers = await _customerRepository.FindAsync(
                c => c.Email == customer.Email, cancellationToken);
            
            if (existingCustomers.Any())
                throw new InvalidOperationException($"Customer with email {customer.Email} already exists");
            
            return await _customerRepository.AddAsync(customer, cancellationToken);
        }

        public async Task<Customer> UpdateCustomerAsync(Customer customer, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(customer);
            
            var existingCustomer = await _customerRepository.GetByIdAsync(customer.Id, cancellationToken);
            if (existingCustomer is null)
                throw new InvalidOperationException($"Customer with ID {customer.Id} not found");
            
            return await _customerRepository.UpdateAsync(customer, cancellationToken);
        }

        public async Task<bool> DeleteCustomerAsync(Guid customerId, CancellationToken cancellationToken = default)
        {
            // Check for existing orders
            var orders = await _orderRepository.FindAsync(o => o.CustomerId == customerId, cancellationToken);
            if (orders.Any())
                throw new InvalidOperationException("Cannot delete customer with existing orders");
            
            return await _customerRepository.DeleteAsync(customerId, cancellationToken);
        }

        public async Task<IEnumerable<Customer>> FindCustomersByEmailDomainAsync(string domain, CancellationToken cancellationToken = default)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(domain);
            
            return await _customerRepository.FindAsync(
                c => c.Email.Contains($"@{domain}"), cancellationToken);
        }
    }

    // =============================================================================
    // Performance Optimization and Memory Management
    // =============================================================================

    /// <summary>
    /// Object pool for expensive-to-create objects
    /// </summary>
    public class ObjectPool<T> where T : class, new()
    {
        private readonly ConcurrentQueue<T> _objects = new();
        private readonly Func<T> _objectGenerator;
        private readonly int _maxSize;
        private int _currentCount;

        public ObjectPool(int maxSize = 100, Func<T>? objectGenerator = null)
        {
            _maxSize = maxSize;
            _objectGenerator = objectGenerator ?? (() => new T());
        }

        public T Get()
        {
            if (_objects.TryDequeue(out var item))
            {
                Interlocked.Decrement(ref _currentCount);
                return item;
            }

            return _objectGenerator();
        }

        public void Return(T item)
        {
            ArgumentNullException.ThrowIfNull(item);
            
            if (_currentCount < _maxSize)
            {
                _objects.Enqueue(item);
                Interlocked.Increment(ref _currentCount);
            }
        }
    }

    /// <summary>
    /// Memory-efficient string builder pool
    /// </summary>
    public static class StringBuilderPool
    {
        private static readonly ObjectPool<System.Text.StringBuilder> Pool = 
            new(objectGenerator: () => new System.Text.StringBuilder(1024));

        public static System.Text.StringBuilder Get()
        {
            var sb = Pool.Get();
            sb.Clear();
            return sb;
        }

        public static void Return(System.Text.StringBuilder stringBuilder)
        {
            if (stringBuilder.Capacity <= 8192) // Don't pool very large builders
            {
                Pool.Return(stringBuilder);
            }
        }

        public static string BuildString(Action<System.Text.StringBuilder> buildAction)
        {
            ArgumentNullException.ThrowIfNull(buildAction);
            
            var sb = Get();
            try
            {
                buildAction(sb);
                return sb.ToString();
            }
            finally
            {
                Return(sb);
            }
        }
    }

    /// <summary>
    /// Performance measurement utilities
    /// </summary>
    public static class PerformanceProfiler
    {
        /// <summary>
        /// Measure execution time of an operation
        /// </summary>
        public static async Task<(T Result, TimeSpan Duration)> MeasureAsync<T>(Func<Task<T>> operation)
        {
            ArgumentNullException.ThrowIfNull(operation);
            
            var stopwatch = Stopwatch.StartNew();
            var result = await operation();
            stopwatch.Stop();
            
            return (result, stopwatch.Elapsed);
        }

        /// <summary>
        /// Measure execution time with memory usage
        /// </summary>
        public static (T Result, TimeSpan Duration, long MemoryBefore, long MemoryAfter) MeasureWithMemory<T>(Func<T> operation)
        {
            ArgumentNullException.ThrowIfNull(operation);
            
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            
            var memoryBefore = GC.GetTotalMemory(false);
            var stopwatch = Stopwatch.StartNew();
            
            var result = operation();
            
            stopwatch.Stop();
            var memoryAfter = GC.GetTotalMemory(false);
            
            return (result, stopwatch.Elapsed, memoryBefore, memoryAfter);
        }

        /// <summary>
        /// Benchmark an operation with multiple iterations
        /// </summary>
        public static BenchmarkResult Benchmark(Action operation, int iterations = 1000)
        {
            ArgumentNullException.ThrowIfNull(operation);
            
            // Warmup
            for (int i = 0; i < Math.Min(iterations / 10, 100); i++)
            {
                operation();
            }
            
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            
            var times = new List<TimeSpan>(iterations);
            var memoryBefore = GC.GetTotalMemory(false);
            
            for (int i = 0; i < iterations; i++)
            {
                var stopwatch = Stopwatch.StartNew();
                operation();
                stopwatch.Stop();
                times.Add(stopwatch.Elapsed);
            }
            
            var memoryAfter = GC.GetTotalMemory(true);
            
            return new BenchmarkResult
            {
                Iterations = iterations,
                TotalTime = times.Sum(t => t.Ticks),
                MinTime = times.Min(),
                MaxTime = times.Max(),
                AverageTime = TimeSpan.FromTicks((long)times.Average(t => t.Ticks)),
                MemoryAllocated = memoryAfter - memoryBefore
            };
        }
    }

    public class BenchmarkResult
    {
        public int Iterations { get; set; }
        public long TotalTime { get; set; }
        public TimeSpan MinTime { get; set; }
        public TimeSpan MaxTime { get; set; }
        public TimeSpan AverageTime { get; set; }
        public long MemoryAllocated { get; set; }
        
        public double OperationsPerSecond => 
            AverageTime.TotalSeconds > 0 ? 1.0 / AverageTime.TotalSeconds : 0;
    }

    // =============================================================================
    // Example Usage and Demonstrations
    // =============================================================================

    /// <summary>
    /// Main program demonstrating all advanced C# features
    /// </summary>
    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 Advanced C# Programming Demonstration");
            Console.WriteLine(new string('=', 50));
            
            await DemonstrateRecordTypes();
            await DemonstrateFunctionalProgramming();
            await DemonstrateAsyncPatterns();
            await DemonstrateDependencyInjection();
            await DemonstratePerformanceOptimization();
            
            Console.WriteLine("\n✅ All C# demonstrations completed!");
        }

        private static async Task DemonstrateRecordTypes()
        {
            Console.WriteLine("\n📝 Record Types and Pattern Matching:");
            
            var person = new PersonRecord(
                "John", 
                "Doe", 
                new DateTime(1990, 5, 15), 
                "john.doe@example.com");
            
            Console.WriteLine($"Person: {person.FullName}");
            Console.WriteLine($"Age: {person.Age}");
            Console.WriteLine($"Category: {person.AgeCategory}");
            Console.WriteLine($"Is Valid: {person.IsValid()}");
            
            // Demonstrate with expressions
            var updatedPerson = person with { Email = "john.doe@company.com" };
            Console.WriteLine($"Updated email: {updatedPerson.Email}");
            Console.WriteLine($"Original unchanged: {person.Email}");
        }

        private static async Task DemonstrateFunctionalProgramming()
        {
            Console.WriteLine("\n🔧 Functional Programming Patterns:");
            
            // Maybe monad demonstration
            var dictionary = new Dictionary<string, string>
            {
                ["key1"] = "value1",
                ["key2"] = "value2"
            };
            
            var result = dictionary.TryGetValue("key1")
                .Map(value => value.ToUpper())
                .Filter(value => value.Length > 3)
                .GetValueOrDefault("Not found");
            
            Console.WriteLine($"Maybe result: {result}");
            
            // Pipeline demonstration
            var numbers = Enumerable.Range(1, 100);
            var processedResult = numbers
                .Where(n => n % 2 == 0)
                .Take(10)
                .Pipe(evenNumbers => evenNumbers.Sum())
                .Tap(sum => Console.WriteLine($"Sum of first 10 even numbers: {sum}"));
            
            // Batch processing
            var batches = numbers.Batch(10).ToList();
            Console.WriteLine($"Created {batches.Count} batches of 10 numbers");
        }

        private static async Task DemonstrateAsyncPatterns()
        {
            Console.WriteLine("\n⚡ Async Programming Patterns:");
            
            using var httpClient = new HttpClient();
            using var resilientClient = new ResilientHttpClient(httpClient, TimeSpan.FromMinutes(1));
            
            try
            {
                // Demonstrate timeout handling
                var timeoutTask = Task.Delay(TimeSpan.FromSeconds(2));
                var result = await TaskCoordinator.WithTimeout(timeoutTask, TimeSpan.FromSeconds(1));
            }
            catch (TimeoutException)
            {
                Console.WriteLine("Operation timed out as expected");
            }
            
            // Demonstrate fan-out/fan-in
            var input = 42;
            var processors = new List<Func<int, Task<int>>>
            {
                async x => { await Task.Delay(10); return x * 2; },
                async x => { await Task.Delay(15); return x + 10; },
                async x => { await Task.Delay(5); return x - 5; }
            };
            
            var fanOutResult = await TaskCoordinator.FanOutFanIn(
                input,
                processors,
                results => results.Sum());
            
            Console.WriteLine($"Fan-out/fan-in result: {fanOutResult}");
            
            // Demonstrate async enumerable
            var producers = new List<Func<Task<string>>>
            {
                async () => { await Task.Delay(10); return "First"; },
                async () => { await Task.Delay(20); return "Second"; },
                async () => { await Task.Delay(5); return "Third"; }
            };
            
            Console.WriteLine("Async enumerable results:");
            await foreach (var item in TaskCoordinator.ProduceAsync(producers))
            {
                Console.WriteLine($"  Received: {item}");
            }
        }

        private static async Task DemonstrateDependencyInjection()
        {
            Console.WriteLine("\n🏗️ Dependency Injection:");
            
            // Setup container
            var container = new SimpleContainer();
            container.RegisterTransient<IRepository<Customer>, InMemoryRepository<Customer>>();
            container.RegisterTransient<IRepository<Order>, InMemoryRepository<Order>>();
            container.RegisterTransient<ICustomerService, CustomerService>();
            
            // Use services
            var customerService = container.Resolve<ICustomerService>();
            
            var customer = new Customer
            {
                FirstName = "Jane",
                LastName = "Smith",
                Email = "jane.smith@example.com",
                DateOfBirth = new DateTime(1985, 3, 20)
            };
            
            var createdCustomer = await customerService.CreateCustomerAsync(customer);
            Console.WriteLine($"Created customer: {createdCustomer.FirstName} {createdCustomer.LastName}");
            
            var retrievedCustomer = await customerService.GetCustomerByIdAsync(createdCustomer.Id);
            Console.WriteLine($"Retrieved customer: {retrievedCustomer?.Email}");
            
            // Demonstrate LINQ query builder
            var customerRepo = container.Resolve<IRepository<Customer>>();
            var queryBuilder = new QueryBuilder<Customer>(customerRepo.GetAllAsync().Result.AsQueryable());
            
            var customers = await queryBuilder
                .Where(c => c.Email.Contains("example.com"))
                .OrderBy(c => c.LastName)
                .Take(10)
                .ToListAsync();
            
            Console.WriteLine($"Found {customers.Count} customers with example.com email");
        }

        private static async Task DemonstratePerformanceOptimization()
        {
            Console.WriteLine("\n⚡ Performance Optimization:");
            
            // Object pool demonstration
            var stringBuilderPool = new ObjectPool<System.Text.StringBuilder>(
                objectGenerator: () => new System.Text.StringBuilder(1024));
            
            var sb = stringBuilderPool.Get();
            sb.Append("Hello, ");
            sb.Append("World!");
            var result = sb.ToString();
            stringBuilderPool.Return(sb);
            
            Console.WriteLine($"Pooled StringBuilder result: {result}");
            
            // String builder pool utility
            var builtString = StringBuilderPool.BuildString(sb =>
            {
                sb.Append("Built with ");
                sb.Append("pooled ");
                sb.Append("StringBuilder");
            });
            
            Console.WriteLine($"Built string: {builtString}");
            
            // Performance benchmarking
            Console.WriteLine("\nBenchmark Results:");
            
            var benchmarkResult = PerformanceProfiler.Benchmark(() =>
            {
                var list = new List<int>();
                for (int i = 0; i < 1000; i++)
                {
                    list.Add(i * i);
                }
            }, 1000);
            
            Console.WriteLine($"Iterations: {benchmarkResult.Iterations}");
            Console.WriteLine($"Average time: {benchmarkResult.AverageTime.TotalMicroseconds:F2} μs");
            Console.WriteLine($"Operations/sec: {benchmarkResult.OperationsPerSecond:F0}");
            Console.WriteLine($"Memory allocated: {benchmarkResult.MemoryAllocated} bytes");
            
            // Memory measurement
            var (memResult, duration, memBefore, memAfter) = PerformanceProfiler.MeasureWithMemory(() =>
            {
                return Enumerable.Range(1, 10000).ToArray();
            });
            
            Console.WriteLine($"\nMemory measurement:");
            Console.WriteLine($"Duration: {duration.TotalMilliseconds:F2} ms");
            Console.WriteLine($"Memory before: {memBefore} bytes");
            Console.WriteLine($"Memory after: {memAfter} bytes");
            Console.WriteLine($"Memory allocated: {memAfter - memBefore} bytes");
            Console.WriteLine($"Array length: {memResult.Length}");
        }
    }
}