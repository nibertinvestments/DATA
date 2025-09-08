/**
 * Comprehensive C# Examples
 * Demonstrates modern C# features, LINQ, async/await, generics, and .NET patterns
 */

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using System.Text.Json;
using System.ComponentModel.DataAnnotations;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Runtime.Serialization;

// ========== Domain Models and Records ==========

/// <summary>
/// User record demonstrating modern C# record types
/// </summary>
public record User(
    int Id,
    string Username,
    string Email,
    int? Age = null,
    bool Active = true,
    DateTime CreatedAt = default,
    List<Role> Roles = null
)
{
    // Additional properties with validation
    [Required]
    [StringLength(50, MinimumLength = 3)]
    public string Username { get; init; } = Username;

    [Required]
    [EmailAddress]
    public string Email { get; init; } = Email;

    [Range(0, 150)]
    public int? Age { get; init; } = Age;

    public DateTime CreatedAt { get; init; } = CreatedAt == default ? DateTime.UtcNow : CreatedAt;
    
    public List<Role> Roles { get; init; } = Roles ?? new List<Role>();

    // Business logic methods
    public bool IsAdult => Age >= 18;
    public bool HasRole(Role role) => Roles.Contains(role);
    
    // Deconstruction
    public void Deconstruct(out string username, out string email, out bool active)
    {
        username = Username;
        email = Email;
        active = Active;
    }

    // Validation method
    public ValidationResult Validate()
    {
        var context = new ValidationContext(this);
        var results = new List<ValidationResult>();
        
        if (!Validator.TryValidateObject(this, context, results, true))
        {
            return new ValidationResult(string.Join("; ", results.Select(r => r.ErrorMessage)));
        }
        
        return ValidationResult.Success;
    }

    // ToString override for records
    public override string ToString() => 
        $"User {{ Id = {Id}, Username = {Username}, Email = {Email}, Age = {Age}, Active = {Active} }}";
}

/// <summary>
/// Product base class demonstrating inheritance and polymorphism
/// </summary>
public abstract class Product
{
    public int Id { get; init; }
    public string Name { get; init; }
    public string Description { get; init; }
    public decimal BasePrice { get; init; }
    public Category Category { get; init; }
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    public bool Available { get; set; } = true;

    protected Product(int id, string name, string description, decimal basePrice, Category category)
    {
        Id = id;
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Description = description ?? throw new ArgumentNullException(nameof(description));
        BasePrice = basePrice;
        Category = category;
    }

    // Abstract methods for polymorphism
    public abstract decimal CalculatePrice();
    public abstract decimal CalculateShippingCost();
    public abstract string GetProductType();

    // Virtual method with default implementation
    public virtual string GetDisplayName() => $"{Name} ({Category})";

    // Template method pattern
    public decimal GetTotalCost() => CalculatePrice() + CalculateShippingCost();

    public override string ToString() => 
        $"{GetProductType()}: {Name} - ${CalculatePrice():F2}";
}

/// <summary>
/// Physical product implementation
/// </summary>
public sealed class PhysicalProduct : Product
{
    public double Weight { get; init; }
    public Dimensions Dimensions { get; init; }

    public PhysicalProduct(int id, string name, string description, decimal basePrice, 
                          Category category, double weight, Dimensions dimensions)
        : base(id, name, description, basePrice, category)
    {
        Weight = weight;
        Dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));
    }

    public override decimal CalculatePrice() => BasePrice + (decimal)(Weight * 0.5m);

    public override decimal CalculateShippingCost()
    {
        var weightCost = (decimal)(Weight * 2.0);
        var sizeCost = (decimal)(Dimensions.Volume * 0.1);
        return weightCost + sizeCost;
    }

    public override string GetProductType() => "Physical";

    public override string GetDisplayName() => 
        $"{base.GetDisplayName()} - {Weight}kg, {Dimensions}";
}

/// <summary>
/// Digital product implementation
/// </summary>
public sealed class DigitalProduct : Product
{
    public long FileSizeBytes { get; init; }
    public string DownloadUrl { get; set; }
    public int DownloadLimit { get; init; }

    public DigitalProduct(int id, string name, string description, decimal basePrice,
                         Category category, long fileSizeBytes, int downloadLimit)
        : base(id, name, description, basePrice, category)
    {
        FileSizeBytes = fileSizeBytes;
        DownloadLimit = downloadLimit;
    }

    public override decimal CalculatePrice() => BasePrice; // No additional cost

    public override decimal CalculateShippingCost() => 0; // No shipping for digital

    public override string GetProductType() => "Digital";

    public string GetFileSizeFormatted()
    {
        var sizes = new[] { "B", "KB", "MB", "GB", "TB" };
        var order = 0;
        double size = FileSizeBytes;

        while (size >= 1024 && order < sizes.Length - 1)
        {
            order++;
            size /= 1024;
        }

        return $"{size:0.##} {sizes[order]}";
    }
}

/// <summary>
/// Value object for dimensions
/// </summary>
public readonly record struct Dimensions(double Length, double Width, double Height)
{
    public double Volume => Length * Width * Height;
    public double SurfaceArea => 2 * (Length * Width + Length * Height + Width * Height);
    
    public override string ToString() => $"{Length:F1} x {Width:F1} x {Height:F1}";
}

/// <summary>
/// Enumerations
/// </summary>
public enum Role
{
    Guest,
    User,
    Moderator,
    Admin
}

public enum Category
{
    Electronics,
    Books,
    Clothing,
    Home,
    Sports,
    Software,
    Media
}

// ========== Generic Repository Pattern ==========

/// <summary>
/// Generic repository interface
/// </summary>
public interface IRepository<T, TKey> where T : class
{
    Task<T> GetByIdAsync(TKey id);
    Task<IEnumerable<T>> GetAllAsync();
    Task<T> AddAsync(T entity);
    Task<T> UpdateAsync(T entity);
    Task<bool> DeleteAsync(TKey id);
    Task<bool> ExistsAsync(TKey id);
    Task<int> CountAsync();
}

/// <summary>
/// Searchable repository interface
/// </summary>
public interface ISearchableRepository<T> where T : class
{
    Task<IEnumerable<T>> SearchAsync(string query);
    Task<IEnumerable<T>> FindByConditionAsync(Func<T, bool> predicate);
}

/// <summary>
/// In-memory repository implementation
/// </summary>
public class InMemoryRepository<T, TKey> : IRepository<T, TKey>, ISearchableRepository<T>
    where T : class
    where TKey : IEquatable<TKey>
{
    private readonly ConcurrentDictionary<TKey, T> _storage = new();
    private readonly Func<T, TKey> _keySelector;
    private readonly Action<T, TKey> _keySetter;
    private readonly ReaderWriterLockSlim _lock = new();
    private int _nextId = 1;

    public InMemoryRepository(Func<T, TKey> keySelector, Action<T, TKey> keySetter = null)
    {
        _keySelector = keySelector ?? throw new ArgumentNullException(nameof(keySelector));
        _keySetter = keySetter;
    }

    public Task<T> GetByIdAsync(TKey id)
    {
        _lock.EnterReadLock();
        try
        {
            _storage.TryGetValue(id, out var entity);
            return Task.FromResult(entity);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public Task<IEnumerable<T>> GetAllAsync()
    {
        _lock.EnterReadLock();
        try
        {
            return Task.FromResult(_storage.Values.AsEnumerable());
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public Task<T> AddAsync(T entity)
    {
        if (entity == null) throw new ArgumentNullException(nameof(entity));

        _lock.EnterWriteLock();
        try
        {
            var key = _keySelector(entity);
            
            // Generate new key if needed
            if (key.Equals(default(TKey)) && _keySetter != null && typeof(TKey) == typeof(int))
            {
                var newKey = (TKey)(object)Interlocked.Increment(ref _nextId);
                _keySetter(entity, newKey);
                key = newKey;
            }

            _storage[key] = entity;
            return Task.FromResult(entity);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public Task<T> UpdateAsync(T entity)
    {
        if (entity == null) throw new ArgumentNullException(nameof(entity));

        _lock.EnterWriteLock();
        try
        {
            var key = _keySelector(entity);
            if (_storage.ContainsKey(key))
            {
                _storage[key] = entity;
                return Task.FromResult(entity);
            }
            
            throw new InvalidOperationException($"Entity with key {key} not found");
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public Task<bool> DeleteAsync(TKey id)
    {
        _lock.EnterWriteLock();
        try
        {
            return Task.FromResult(_storage.TryRemove(id, out _));
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public Task<bool> ExistsAsync(TKey id)
    {
        _lock.EnterReadLock();
        try
        {
            return Task.FromResult(_storage.ContainsKey(id));
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public Task<int> CountAsync()
    {
        _lock.EnterReadLock();
        try
        {
            return Task.FromResult(_storage.Count);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public Task<IEnumerable<T>> SearchAsync(string query)
    {
        _lock.EnterReadLock();
        try
        {
            var results = _storage.Values
                .Where(entity => entity.ToString().Contains(query, StringComparison.OrdinalIgnoreCase));
            return Task.FromResult(results);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public Task<IEnumerable<T>> FindByConditionAsync(Func<T, bool> predicate)
    {
        _lock.EnterReadLock();
        try
        {
            var results = _storage.Values.Where(predicate);
            return Task.FromResult(results);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _lock?.Dispose();
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
}

// ========== Service Layer with Dependency Injection ==========

/// <summary>
/// User service interface
/// </summary>
public interface IUserService
{
    Task<User> CreateUserAsync(string username, string email, int? age = null);
    Task<User> GetUserAsync(int id);
    Task<User> UpdateUserAsync(int id, string username = null, string email = null, int? age = null);
    Task<bool> DeleteUserAsync(int id);
    Task<IEnumerable<User>> GetAllUsersAsync();
    Task<IEnumerable<User>> SearchUsersAsync(string query);
    Task<IEnumerable<User>> GetActiveUsersAsync();
    Task<IEnumerable<User>> GetUsersByRoleAsync(Role role);
}

/// <summary>
/// User service implementation
/// </summary>
public class UserService : IUserService
{
    private readonly IRepository<User, int> _userRepository;
    private readonly ISearchableRepository<User> _searchableRepository;
    private readonly ILogger _logger;

    public UserService(IRepository<User, int> userRepository, ILogger logger = null)
    {
        _userRepository = userRepository ?? throw new ArgumentNullException(nameof(userRepository));
        _searchableRepository = userRepository as ISearchableRepository<User>;
        _logger = logger ?? new ConsoleLogger();
    }

    public async Task<User> CreateUserAsync(string username, string email, int? age = null)
    {
        _logger.LogInfo($"Creating user: {username}");

        // Validation
        if (string.IsNullOrWhiteSpace(username))
            throw new ArgumentException("Username cannot be null or empty", nameof(username));
        
        if (string.IsNullOrWhiteSpace(email))
            throw new ArgumentException("Email cannot be null or empty", nameof(email));

        // Check for duplicates
        var existingUsers = await _userRepository.GetAllAsync();
        if (existingUsers.Any(u => u.Username.Equals(username, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException($"Username '{username}' already exists");

        if (existingUsers.Any(u => u.Email.Equals(email, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException($"Email '{email}' already exists");

        // Create user
        var user = new User(0, username, email, age, Roles: new List<Role> { Role.User });
        
        // Validate
        var validationResult = user.Validate();
        if (validationResult != ValidationResult.Success)
            throw new ValidationException(validationResult.ErrorMessage);

        var createdUser = await _userRepository.AddAsync(user);
        
        _logger.LogInfo($"User created successfully: {createdUser.Id}");
        return createdUser;
    }

    public async Task<User> GetUserAsync(int id)
    {
        var user = await _userRepository.GetByIdAsync(id);
        if (user == null)
            throw new KeyNotFoundException($"User with ID {id} not found");
        
        return user;
    }

    public async Task<User> UpdateUserAsync(int id, string username = null, string email = null, int? age = null)
    {
        var existingUser = await GetUserAsync(id);
        
        var updatedUser = existingUser with
        {
            Username = username ?? existingUser.Username,
            Email = email ?? existingUser.Email,
            Age = age ?? existingUser.Age
        };

        var validationResult = updatedUser.Validate();
        if (validationResult != ValidationResult.Success)
            throw new ValidationException(validationResult.ErrorMessage);

        return await _userRepository.UpdateAsync(updatedUser);
    }

    public async Task<bool> DeleteUserAsync(int id)
    {
        var exists = await _userRepository.ExistsAsync(id);
        if (!exists)
            throw new KeyNotFoundException($"User with ID {id} not found");

        return await _userRepository.DeleteAsync(id);
    }

    public async Task<IEnumerable<User>> GetAllUsersAsync()
    {
        return await _userRepository.GetAllAsync();
    }

    public async Task<IEnumerable<User>> SearchUsersAsync(string query)
    {
        if (_searchableRepository == null)
            throw new NotSupportedException("Repository doesn't support search operations");

        return await _searchableRepository.SearchAsync(query);
    }

    public async Task<IEnumerable<User>> GetActiveUsersAsync()
    {
        var users = await _userRepository.GetAllAsync();
        return users.Where(u => u.Active);
    }

    public async Task<IEnumerable<User>> GetUsersByRoleAsync(Role role)
    {
        var users = await _userRepository.GetAllAsync();
        return users.Where(u => u.HasRole(role));
    }
}

// ========== LINQ and Functional Programming ==========

/// <summary>
/// Data analytics service demonstrating LINQ and functional programming
/// </summary>
public class AnalyticsService
{
    private readonly IUserService _userService;

    public AnalyticsService(IUserService userService)
    {
        _userService = userService ?? throw new ArgumentNullException(nameof(userService));
    }

    /// <summary>
    /// Advanced user analytics using LINQ
    /// </summary>
    public async Task<UserAnalytics> GetUserAnalyticsAsync()
    {
        var users = await _userService.GetAllUsersAsync();
        var userList = users.ToList();

        return new UserAnalytics
        {
            TotalUsers = userList.Count,
            ActiveUsers = userList.Count(u => u.Active),
            AdultUsers = userList.Count(u => u.IsAdult),
            
            AverageAge = userList
                .Where(u => u.Age.HasValue)
                .Select(u => u.Age.Value)
                .DefaultIfEmpty(0)
                .Average(),

            AgeDistribution = userList
                .Where(u => u.Age.HasValue)
                .GroupBy(u => u.Age.Value / 10 * 10) // Group by decade
                .ToDictionary(g => $"{g.Key}-{g.Key + 9}", g => g.Count()),

            RoleDistribution = userList
                .SelectMany(u => u.Roles)
                .GroupBy(r => r)
                .ToDictionary(g => g.Key.ToString(), g => g.Count()),

            TopEmailDomains = userList
                .Select(u => u.Email.Split('@')[1])
                .GroupBy(domain => domain)
                .OrderByDescending(g => g.Count())
                .Take(10)
                .ToDictionary(g => g.Key, g => g.Count()),

            RecentUsers = userList
                .Where(u => u.CreatedAt > DateTime.UtcNow.AddDays(-30))
                .Count(),

            UserCreationTrend = userList
                .GroupBy(u => u.CreatedAt.Date)
                .OrderBy(g => g.Key)
                .ToDictionary(g => g.Key, g => g.Count())
        };
    }

    /// <summary>
    /// Complex LINQ queries demonstration
    /// </summary>
    public async Task<IEnumerable<UserSummary>> GetUserSummariesAsync()
    {
        var users = await _userService.GetAllUsersAsync();

        return users
            .Where(u => u.Active)
            .OrderBy(u => u.Username)
            .Select(u => new UserSummary
            {
                Id = u.Id,
                Username = u.Username,
                Domain = u.Email.Split('@')[1],
                AgeGroup = u.Age switch
                {
                    null => "Unknown",
                    < 18 => "Minor",
                    >= 18 and < 65 => "Adult",
                    >= 65 => "Senior",
                    _ => "Unknown"
                },
                RoleNames = u.Roles.Select(r => r.ToString()).ToList(),
                AccountAge = (int)(DateTime.UtcNow - u.CreatedAt).TotalDays
            })
            .ToList();
    }

    /// <summary>
    /// Parallel LINQ demonstration
    /// </summary>
    public async Task<Dictionary<string, object>> ProcessUsersInParallelAsync()
    {
        var users = (await _userService.GetAllUsersAsync()).ToList();

        var stopwatch = Stopwatch.StartNew();

        // Parallel processing of complex operations
        var emailDomains = users
            .AsParallel()
            .Where(u => u.Active)
            .Select(u => u.Email.Split('@')[1])
            .GroupBy(domain => domain)
            .Select(g => new { Domain = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .Take(10)
            .ToDictionary(x => x.Domain, x => x.Count);

        var averageAge = users
            .AsParallel()
            .Where(u => u.Age.HasValue)
            .Select(u => u.Age.Value)
            .Average();

        stopwatch.Stop();

        return new Dictionary<string, object>
        {
            ["EmailDomains"] = emailDomains,
            ["AverageAge"] = averageAge,
            ["ProcessingTimeMs"] = stopwatch.ElapsedMilliseconds,
            ["ProcessedUsers"] = users.Count
        };
    }
}

/// <summary>
/// User analytics result
/// </summary>
public record UserAnalytics
{
    public int TotalUsers { get; init; }
    public int ActiveUsers { get; init; }
    public int AdultUsers { get; init; }
    public double AverageAge { get; init; }
    public Dictionary<string, int> AgeDistribution { get; init; } = new();
    public Dictionary<string, int> RoleDistribution { get; init; } = new();
    public Dictionary<string, int> TopEmailDomains { get; init; } = new();
    public int RecentUsers { get; init; }
    public Dictionary<DateTime, int> UserCreationTrend { get; init; } = new();
}

/// <summary>
/// User summary for reporting
/// </summary>
public record UserSummary
{
    public int Id { get; init; }
    public string Username { get; init; }
    public string Domain { get; init; }
    public string AgeGroup { get; init; }
    public List<string> RoleNames { get; init; } = new();
    public int AccountAge { get; init; }
}

// ========== Async/Await and Concurrency ==========

/// <summary>
/// Asynchronous user service with concurrency
/// </summary>
public class AsyncUserService
{
    private readonly IUserService _userService;
    private readonly SemaphoreSlim _semaphore;

    public AsyncUserService(IUserService userService, int maxConcurrency = 10)
    {
        _userService = userService ?? throw new ArgumentNullException(nameof(userService));
        _semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
    }

    /// <summary>
    /// Create multiple users concurrently
    /// </summary>
    public async Task<List<User>> CreateUsersAsync(IEnumerable<(string username, string email, int? age)> userData)
    {
        var tasks = userData.Select(async data =>
        {
            await _semaphore.WaitAsync();
            try
            {
                await Task.Delay(50); // Simulate network delay
                return await _userService.CreateUserAsync(data.username, data.email, data.age);
            }
            finally
            {
                _semaphore.Release();
            }
        });

        return (await Task.WhenAll(tasks)).ToList();
    }

    /// <summary>
    /// Validate users asynchronously with timeout
    /// </summary>
    public async Task<Dictionary<int, bool>> ValidateUsersAsync(IEnumerable<int> userIds, TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        var results = new ConcurrentDictionary<int, bool>();

        var tasks = userIds.Select(async id =>
        {
            try
            {
                var user = await _userService.GetUserAsync(id);
                var validationResult = user.Validate();
                results[id] = validationResult == ValidationResult.Success;
            }
            catch (Exception)
            {
                results[id] = false;
            }
        });

        try
        {
            await Task.WhenAll(tasks);
        }
        catch (OperationCanceledException)
        {
            // Some validations may have timed out
        }

        return new Dictionary<int, bool>(results);
    }

    /// <summary>
    /// Process users with progress reporting
    /// </summary>
    public async Task ProcessUsersWithProgressAsync(IProgress<ProgressReport> progress = null)
    {
        var users = await _userService.GetAllUsersAsync();
        var userList = users.ToList();
        var totalUsers = userList.Count;

        for (int i = 0; i < totalUsers; i++)
        {
            var user = userList[i];
            
            // Simulate processing
            await Task.Delay(100);
            
            // Report progress
            progress?.Report(new ProgressReport
            {
                ProcessedCount = i + 1,
                TotalCount = totalUsers,
                CurrentItem = user.Username,
                PercentComplete = (double)(i + 1) / totalUsers * 100
            });
        }
    }

    /// <summary>
    /// Retry pattern with exponential backoff
    /// </summary>
    public async Task<T> RetryAsync<T>(Func<Task<T>> operation, int maxRetries = 3)
    {
        var delay = TimeSpan.FromMilliseconds(100);
        
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                return await operation();
            }
            catch (Exception ex) when (attempt < maxRetries)
            {
                Console.WriteLine($"Attempt {attempt} failed: {ex.Message}. Retrying in {delay.TotalMilliseconds}ms...");
                await Task.Delay(delay);
                delay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds * 2); // Exponential backoff
            }
        }
        
        // Final attempt without catching exceptions
        return await operation();
    }
}

/// <summary>
/// Progress report for async operations
/// </summary>
public record ProgressReport
{
    public int ProcessedCount { get; init; }
    public int TotalCount { get; init; }
    public string CurrentItem { get; init; }
    public double PercentComplete { get; init; }
    
    public override string ToString() => 
        $"Progress: {ProcessedCount}/{TotalCount} ({PercentComplete:F1}%) - Current: {CurrentItem}";
}

// ========== File I/O and Serialization ==========

/// <summary>
/// File service for data persistence
/// </summary>
public class FileService
{
    private readonly JsonSerializerOptions _jsonOptions;

    public FileService()
    {
        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Save users to JSON file
    /// </summary>
    public async Task SaveUsersToJsonAsync(IEnumerable<User> users, string filePath)
    {
        var json = JsonSerializer.Serialize(users, _jsonOptions);
        await File.WriteAllTextAsync(filePath, json);
    }

    /// <summary>
    /// Load users from JSON file
    /// </summary>
    public async Task<List<User>> LoadUsersFromJsonAsync(string filePath)
    {
        if (!File.Exists(filePath))
            return new List<User>();

        var json = await File.ReadAllTextAsync(filePath);
        return JsonSerializer.Deserialize<List<User>>(json, _jsonOptions) ?? new List<User>();
    }

    /// <summary>
    /// Save users to CSV file
    /// </summary>
    public async Task SaveUsersToCsvAsync(IEnumerable<User> users, string filePath)
    {
        using var writer = new StreamWriter(filePath);
        
        // Write header
        await writer.WriteLineAsync("Id,Username,Email,Age,Active,CreatedAt,Roles");
        
        // Write data
        foreach (var user in users)
        {
            var roles = string.Join(";", user.Roles);
            var line = $"{user.Id},{EscapeCsv(user.Username)},{EscapeCsv(user.Email)}," +
                      $"{user.Age},{user.Active},{user.CreatedAt:yyyy-MM-dd HH:mm:ss},{roles}";
            await writer.WriteLineAsync(line);
        }
    }

    /// <summary>
    /// Process large files with streaming
    /// </summary>
    public async Task<int> ProcessLargeFileAsync(string filePath, Func<string, bool> lineProcessor)
    {
        var processedLines = 0;
        
        using var reader = new StreamReader(filePath);
        
        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (line != null && lineProcessor(line))
            {
                processedLines++;
            }
        }
        
        return processedLines;
    }

    /// <summary>
    /// Create backup with compression
    /// </summary>
    public async Task CreateBackupAsync(IEnumerable<User> users, string backupDirectory)
    {
        Directory.CreateDirectory(backupDirectory);
        
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        var backupFile = Path.Combine(backupDirectory, $"users_backup_{timestamp}.json");
        
        await SaveUsersToJsonAsync(users, backupFile);
        
        // Compress backup (simplified - in real scenario, use System.IO.Compression)
        var compressedFile = backupFile + ".gz";
        File.Move(backupFile, compressedFile);
        
        Console.WriteLine($"Backup created: {compressedFile}");
    }

    private static string EscapeCsv(string value)
    {
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
        {
            return $"\"{value.Replace("\"", "\"\"")}\"";
        }
        return value;
    }
}

// ========== Logging Interface ==========

/// <summary>
/// Simple logging interface
/// </summary>
public interface ILogger
{
    void LogInfo(string message);
    void LogWarning(string message);
    void LogError(string message, Exception exception = null);
}

/// <summary>
/// Console logger implementation
/// </summary>
public class ConsoleLogger : ILogger
{
    public void LogInfo(string message)
    {
        Console.WriteLine($"[INFO] {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} - {message}");
    }

    public void LogWarning(string message)
    {
        Console.WriteLine($"[WARN] {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} - {message}");
    }

    public void LogError(string message, Exception exception = null)
    {
        Console.WriteLine($"[ERROR] {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} - {message}");
        if (exception != null)
        {
            Console.WriteLine($"Exception: {exception}");
        }
    }
}

// ========== Extension Methods ==========

/// <summary>
/// Extension methods for collections
/// </summary>
public static class CollectionExtensions
{
    public static IEnumerable<T> WhereNotNull<T>(this IEnumerable<T?> source) where T : class
    {
        return source.Where(item => item != null)!;
    }

    public static void ForEach<T>(this IEnumerable<T> source, Action<T> action)
    {
        foreach (var item in source)
        {
            action(item);
        }
    }

    public static Dictionary<TKey, TValue> SafeToDictionary<TSource, TKey, TValue>(
        this IEnumerable<TSource> source,
        Func<TSource, TKey> keySelector,
        Func<TSource, TValue> valueSelector) where TKey : notnull
    {
        var result = new Dictionary<TKey, TValue>();
        foreach (var item in source)
        {
            var key = keySelector(item);
            if (!result.ContainsKey(key))
            {
                result[key] = valueSelector(item);
            }
        }
        return result;
    }

    public static async Task<IEnumerable<TResult>> SelectAsync<TSource, TResult>(
        this IEnumerable<TSource> source,
        Func<TSource, Task<TResult>> selector)
    {
        var tasks = source.Select(selector);
        return await Task.WhenAll(tasks);
    }
}

// ========== Demonstration Class ==========

/// <summary>
/// Main demonstration class
/// </summary>
public class CSharpExamplesDemo
{
    public static async Task Main(string[] args)
    {
        var demo = new CSharpExamplesDemo();
        
        try
        {
            await demo.RunAllDemosAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Demo failed: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    public async Task RunAllDemosAsync()
    {
        Console.WriteLine("=== Comprehensive C# Examples ===\n");

        await DemonstrateBasicFeaturesAsync();
        await DemonstrateLinqAndFunctionalAsync();
        await DemonstrateAsyncProgrammingAsync();
        await DemonstrateFileOperationsAsync();
        await DemonstrateGenericRepositoryAsync();
        DemonstrateExtensionMethods();
        PrintFeaturesSummary();
    }

    private async Task DemonstrateBasicFeaturesAsync()
    {
        Console.WriteLine("=== Basic C# Features Demo ===");

        // Records and deconstruction
        var user1 = new User(1, "alice", "alice@example.com", 25, Roles: new List<Role> { Role.User, Role.Moderator });
        var user2 = new User(2, "bob", "bob@example.com", 17);

        Console.WriteLine($"User 1: {user1}");
        Console.WriteLine($"User 2: {user2}");
        Console.WriteLine($"User 1 is adult: {user1.IsAdult}");

        // Deconstruction
        var (username, email, active) = user1;
        Console.WriteLine($"Deconstructed: {username}, {email}, {active}");

        // Pattern matching and switch expressions
        var category = user1.Age switch
        {
            null => "Unknown",
            < 18 => "Minor",
            >= 18 and < 65 => "Adult",
            >= 65 => "Senior",
            _ => "Invalid"
        };
        Console.WriteLine($"Age category: {category}");

        // Polymorphism
        var products = new List<Product>
        {
            new PhysicalProduct(1, "Laptop", "Gaming laptop", 999.99m, Category.Electronics, 2.5, new Dimensions(35, 25, 3)),
            new DigitalProduct(2, "Software", "Development tools", 199.99m, Category.Software, 1024 * 1024 * 500, 5)
        };

        Console.WriteLine("\nProducts:");
        foreach (var product in products)
        {
            Console.WriteLine($"- {product} (Total: ${product.GetTotalCost():F2})");
        }

        Console.WriteLine();
    }

    private async Task DemonstrateLinqAndFunctionalAsync()
    {
        Console.WriteLine("=== LINQ and Functional Programming Demo ===");

        // Setup
        var userRepo = new InMemoryRepository<User, int>(u => u.Id, (u, id) => u = u with { Id = id });
        var userService = new UserService(userRepo);
        var analyticsService = new AnalyticsService(userService);

        // Create sample data
        var userData = new[]
        {
            ("alice", "alice@example.com", (int?)25),
            ("bob", "bob@gmail.com", (int?)17),
            ("charlie", "charlie@yahoo.com", (int?)30),
            ("diana", "diana@example.com", (int?)22),
            ("eve", "eve@gmail.com", (int?)35)
        };

        foreach (var (username, email, age) in userData)
        {
            await userService.CreateUserAsync(username, email, age);
        }

        // Analytics
        var analytics = await analyticsService.GetUserAnalyticsAsync();
        Console.WriteLine($"Total users: {analytics.TotalUsers}");
        Console.WriteLine($"Active users: {analytics.ActiveUsers}");
        Console.WriteLine($"Adult users: {analytics.AdultUsers}");
        Console.WriteLine($"Average age: {analytics.AverageAge:F1}");

        Console.WriteLine("Top email domains:");
        foreach (var (domain, count) in analytics.TopEmailDomains.Take(3))
        {
            Console.WriteLine($"  - {domain}: {count}");
        }

        // User summaries
        var summaries = await analyticsService.GetUserSummariesAsync();
        Console.WriteLine("\nUser summaries:");
        foreach (var summary in summaries.Take(3))
        {
            Console.WriteLine($"  - {summary.Username} ({summary.AgeGroup}, {summary.Domain})");
        }

        // Parallel processing
        var parallelResults = await analyticsService.ProcessUsersInParallelAsync();
        Console.WriteLine($"\nParallel processing completed in {parallelResults["ProcessingTimeMs"]}ms");

        Console.WriteLine();
    }

    private async Task DemonstrateAsyncProgrammingAsync()
    {
        Console.WriteLine("=== Async Programming Demo ===");

        var userRepo = new InMemoryRepository<User, int>(u => u.Id, (u, id) => u = u with { Id = id });
        var userService = new UserService(userRepo);
        var asyncService = new AsyncUserService(userService);

        // Create users concurrently
        var userData = new[]
        {
            ("async1", "async1@example.com", (int?)25),
            ("async2", "async2@example.com", (int?)30),
            ("async3", "async3@example.com", (int?)28)
        };

        var stopwatch = Stopwatch.StartNew();
        var users = await asyncService.CreateUsersAsync(userData);
        stopwatch.Stop();

        Console.WriteLine($"Created {users.Count} users concurrently in {stopwatch.ElapsedMilliseconds}ms");

        // Progress reporting
        var progress = new Progress<ProgressReport>(report =>
            Console.WriteLine($"Processing: {report}"));

        await asyncService.ProcessUsersWithProgressAsync(progress);

        // Retry pattern
        var result = await asyncService.RetryAsync(async () =>
        {
            await Task.Delay(50);
            return "Success after retry";
        });

        Console.WriteLine($"Retry result: {result}");

        Console.WriteLine();
    }

    private async Task DemonstrateFileOperationsAsync()
    {
        Console.WriteLine("=== File Operations Demo ===");

        var fileService = new FileService();
        var userRepo = new InMemoryRepository<User, int>(u => u.Id, (u, id) => u = u with { Id = id });
        var userService = new UserService(userRepo);

        // Create test users
        var users = new List<User>();
        for (int i = 1; i <= 5; i++)
        {
            var user = await userService.CreateUserAsync($"fileuser{i}", $"fileuser{i}@example.com", 20 + i);
            users.Add(user);
        }

        // Save to JSON
        var jsonFile = "/tmp/users.json";
        await fileService.SaveUsersToJsonAsync(users, jsonFile);
        Console.WriteLine($"Saved {users.Count} users to JSON file");

        // Load from JSON
        var loadedUsers = await fileService.LoadUsersFromJsonAsync(jsonFile);
        Console.WriteLine($"Loaded {loadedUsers.Count} users from JSON file");

        // Save to CSV
        var csvFile = "/tmp/users.csv";
        await fileService.SaveUsersToCsvAsync(users, csvFile);
        Console.WriteLine($"Saved users to CSV file");

        // Create backup
        await fileService.CreateBackupAsync(users, "/tmp/backups");

        Console.WriteLine();
    }

    private async Task DemonstrateGenericRepositoryAsync()
    {
        Console.WriteLine("=== Generic Repository Demo ===");

        var userRepo = new InMemoryRepository<User, int>(u => u.Id, (u, id) => u = u with { Id = id });

        // Add users
        var user1 = await userRepo.AddAsync(new User(0, "repo_user1", "repo1@example.com", 25));
        var user2 = await userRepo.AddAsync(new User(0, "repo_user2", "repo2@example.com", 30));

        Console.WriteLine($"Added users with IDs: {user1.Id}, {user2.Id}");

        // Search and filter
        var searchResults = await userRepo.SearchAsync("repo_user");
        Console.WriteLine($"Search results: {searchResults.Count()}");

        var adults = await userRepo.FindByConditionAsync(u => u.IsAdult);
        Console.WriteLine($"Adult users: {adults.Count()}");

        // Count and existence checks
        var totalCount = await userRepo.CountAsync();
        var user1Exists = await userRepo.ExistsAsync(user1.Id);

        Console.WriteLine($"Total users: {totalCount}");
        Console.WriteLine($"User 1 exists: {user1Exists}");

        Console.WriteLine();
    }

    private void DemonstrateExtensionMethods()
    {
        Console.WriteLine("=== Extension Methods Demo ===");

        var numbers = new List<int?> { 1, null, 3, null, 5 };
        var validNumbers = numbers.WhereNotNull().ToList();
        Console.WriteLine($"Valid numbers: [{string.Join(", ", validNumbers)}]");

        var users = new List<User>
        {
            new User(1, "ext_user1", "ext1@example.com"),
            new User(2, "ext_user2", "ext2@example.com")
        };

        users.ForEach(u => Console.WriteLine($"  Processing: {u.Username}"));

        var userDict = users.SafeToDictionary(u => u.Id, u => u.Username);
        Console.WriteLine($"User dictionary: {userDict.Count} entries");

        Console.WriteLine();
    }

    private void PrintFeaturesSummary()
    {
        Console.WriteLine("=== C# Features Demonstrated ===");
        Console.WriteLine("- Records and init-only properties");
        Console.WriteLine("- Pattern matching and switch expressions");
        Console.WriteLine("- Nullable reference types");
        Console.WriteLine("- Generic repository pattern");
        Console.WriteLine("- LINQ and functional programming");
        Console.WriteLine("- Async/await and Task-based operations");
        Console.WriteLine("- Parallel LINQ (PLINQ)");
        Console.WriteLine("- Extension methods");
        Console.WriteLine("- Dependency injection patterns");
        Console.WriteLine("- File I/O with async operations");
        Console.WriteLine("- JSON serialization");
        Console.WriteLine("- Exception handling and validation");
        Console.WriteLine("- Progress reporting");
        Console.WriteLine("- Retry patterns with exponential backoff");
        Console.WriteLine("- Polymorphism and inheritance");
        Console.WriteLine("- Value types and reference types");
        Console.WriteLine("- Resource management and disposal");
    }
}