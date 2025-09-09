// Comprehensive C# .NET Development Examples
// Demonstrates modern C# features, .NET ecosystem, and enterprise patterns

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Data;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace ComprehensiveCSharpExamples
{
    // ============ Modern C# Records and Value Types ============
    
    /// <summary>
    /// Record types for immutable data structures (C# 9.0+)
    /// </summary>
    public record PersonId(int Value)
    {
        public static implicit operator int(PersonId id) => id.Value;
        public static implicit operator PersonId(int value) => new(value);
        
        public override string ToString() => Value.ToString();
    }
    
    public record Email(string Value)
    {
        public Email : this(Value)
        {
            if (string.IsNullOrWhiteSpace(Value) || !Value.Contains('@'))
                throw new ArgumentException("Invalid email format", nameof(Value));
        }
        
        public string Domain => Value.Split('@')[1];
        public string LocalPart => Value.Split('@')[0];
        
        public static implicit operator string(Email email) => email.Value;
        public static implicit operator Email(string value) => new(value);
    }
    
    public record Address(
        string Street,
        string City,
        string State,
        string ZipCode,
        string Country = "USA"
    )
    {
        public string FullAddress => $"{Street}, {City}, {State} {ZipCode}, {Country}";
    }
    
    public record Person(
        PersonId Id,
        string FirstName,
        string LastName,
        Email Email,
        DateTime DateOfBirth,
        Address Address,
        UserRole Role = UserRole.User
    )
    {
        public string FullName => $"{FirstName} {LastName}";
        public int Age => DateTime.Now.Year - DateOfBirth.Year;
        public bool IsAdult => Age >= 18;
        
        // Records can have methods and properties
        public Person WithUpdatedEmail(Email newEmail) => this with { Email = newEmail };
        public Person WithUpdatedAddress(Address newAddress) => this with { Address = newAddress };
    }
    
    // ============ Enums and Pattern Matching ============
    
    public enum UserRole
    {
        Guest = 0,
        User = 1,
        Moderator = 2,
        Admin = 3,
        SuperAdmin = 4
    }
    
    public enum TaskStatus
    {
        Draft,
        Active,
        InProgress,
        Review,
        Completed,
        Cancelled,
        Archived
    }
    
    public enum Priority
    {
        Low = 1,
        Medium = 2,
        High = 3,
        Critical = 4
    }
    
    // ============ Custom Exceptions ============
    
    public class ValidationException : Exception
    {
        public IDictionary<string, string[]> Errors { get; }
        
        public ValidationException(IDictionary<string, string[]> errors) 
            : base("Validation failed")
        {
            Errors = errors;
        }
        
        public ValidationException(string propertyName, string error) 
            : this(new Dictionary<string, string[]> { [propertyName] = new[] { error } })
        {
        }
    }
    
    public class EntityNotFoundException : Exception
    {
        public string EntityType { get; }
        public object Id { get; }
        
        public EntityNotFoundException(string entityType, object id)
            : base($"{entityType} with ID '{id}' was not found")
        {
            EntityType = entityType;
            Id = id;
        }
    }
    
    public class BusinessRuleException : Exception
    {
        public string RuleName { get; }
        
        public BusinessRuleException(string ruleName, string message) : base(message)
        {
            RuleName = ruleName;
        }
    }
    
    // ============ Domain Models with Rich Behavior ============
    
    public class User
    {
        private readonly List<string> _permissions = new();
        private readonly List<Task> _assignedTasks = new();
        
        public PersonId Id { get; private set; }
        public string FirstName { get; private set; }
        public string LastName { get; private set; }
        public Email Email { get; private set; }
        public DateTime DateOfBirth { get; private set; }
        public Address Address { get; private set; }
        public UserRole Role { get; private set; }
        public bool IsActive { get; private set; } = true;
        public DateTime CreatedAt { get; private set; }
        public DateTime? LastLoginAt { get; private set; }
        public string? PasswordHash { get; private set; }
        
        // Navigation properties
        public IReadOnlyList<string> Permissions => _permissions.AsReadOnly();
        public IReadOnlyList<Task> AssignedTasks => _assignedTasks.AsReadOnly();
        
        // Factory method
        public static User Create(string firstName, string lastName, Email email, 
                                DateTime dateOfBirth, Address address, UserRole role = UserRole.User)
        {
            var user = new User
            {
                Id = PersonId.NewId(),
                FirstName = firstName,
                LastName = lastName,
                Email = email,
                DateOfBirth = dateOfBirth,
                Address = address,
                Role = role,
                CreatedAt = DateTime.UtcNow
            };
            
            user.ValidateInvariant();
            user.InitializePermissions();
            
            return user;
        }
        
        // Business methods
        public void UpdateProfile(string firstName, string lastName, Email email, Address address)
        {
            FirstName = firstName;
            LastName = lastName;
            Email = email;
            Address = address;
            
            ValidateInvariant();
        }
        
        public void ChangeRole(UserRole newRole)
        {
            if (Role == newRole) return;
            
            Role = newRole;
            InitializePermissions();
        }
        
        public void RecordLogin()
        {
            if (!IsActive)
                throw new BusinessRuleException("InactiveUser", "Cannot login with inactive account");
                
            LastLoginAt = DateTime.UtcNow;
        }
        
        public void Deactivate()
        {
            IsActive = false;
        }
        
        public void Activate()
        {
            IsActive = true;
        }
        
        public void SetPassword(string passwordHash)
        {
            if (string.IsNullOrWhiteSpace(passwordHash))
                throw new ArgumentException("Password hash cannot be empty", nameof(passwordHash));
                
            PasswordHash = passwordHash;
        }
        
        public bool HasPermission(string permission)
        {
            return _permissions.Contains(permission, StringComparer.OrdinalIgnoreCase);
        }
        
        public void AssignTask(Task task)
        {
            if (task.AssignedToId != Id)
                throw new BusinessRuleException("TaskAssignment", "Task must be assigned to this user");
                
            if (!_assignedTasks.Contains(task))
                _assignedTasks.Add(task);
        }
        
        // Properties
        public string FullName => $"{FirstName} {LastName}";
        public int Age => DateTime.UtcNow.Year - DateOfBirth.Year;
        public bool IsAdult => Age >= 18;
        
        private void ValidateInvariant()
        {
            var errors = new Dictionary<string, List<string>>();
            
            if (string.IsNullOrWhiteSpace(FirstName))
                errors.GetOrAdd("FirstName").Add("First name is required");
                
            if (string.IsNullOrWhiteSpace(LastName))
                errors.GetOrAdd("LastName").Add("Last name is required");
                
            if (DateOfBirth > DateTime.Today)
                errors.GetOrAdd("DateOfBirth").Add("Date of birth cannot be in the future");
                
            if (Age > 150)
                errors.GetOrAdd("DateOfBirth").Add("Age cannot exceed 150 years");
            
            if (errors.Any())
                throw new ValidationException(errors.ToDictionary(k => k.Key, v => v.Value.ToArray()));
        }
        
        private void InitializePermissions()
        {
            _permissions.Clear();
            
            var permissions = Role switch
            {
                UserRole.Guest => new[] { "read:public" },
                UserRole.User => new[] { "read:public", "read:own", "write:own" },
                UserRole.Moderator => new[] { "read:public", "read:own", "write:own", "moderate:content" },
                UserRole.Admin => new[] { "read:all", "write:all", "delete:all", "manage:users" },
                UserRole.SuperAdmin => new[] { "read:all", "write:all", "delete:all", "manage:users", "system:admin" },
                _ => Array.Empty<string>()
            };
            
            _permissions.AddRange(permissions);
        }
    }
    
    public class Task
    {
        private readonly List<string> _tags = new();
        private readonly List<TaskComment> _comments = new();
        private readonly List<TaskAttachment> _attachments = new();
        
        public int Id { get; private set; }
        public string Title { get; private set; }
        public string Description { get; private set; }
        public Priority Priority { get; private set; }
        public TaskStatus Status { get; private set; }
        public PersonId AssignedToId { get; private set; }
        public PersonId CreatedById { get; private set; }
        public DateTime CreatedAt { get; private set; }
        public DateTime UpdatedAt { get; private set; }
        public DateTime? DueDate { get; private set; }
        public DateTime? CompletedAt { get; private set; }
        public decimal? EstimatedHours { get; private set; }
        public decimal? ActualHours { get; private set; }
        
        // Navigation properties
        public User? AssignedTo { get; private set; }
        public User? CreatedBy { get; private set; }
        public IReadOnlyList<string> Tags => _tags.AsReadOnly();
        public IReadOnlyList<TaskComment> Comments => _comments.AsReadOnly();
        public IReadOnlyList<TaskAttachment> Attachments => _attachments.AsReadOnly();
        
        // Factory method
        public static Task Create(string title, string description, Priority priority, 
                                PersonId assignedToId, PersonId createdById)
        {
            var task = new Task
            {
                Title = title,
                Description = description,
                Priority = priority,
                Status = TaskStatus.Draft,
                AssignedToId = assignedToId,
                CreatedById = createdById,
                CreatedAt = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow
            };
            
            task.ValidateInvariant();
            return task;
        }
        
        // Business methods
        public void UpdateDetails(string title, string description, Priority priority, decimal? estimatedHours = null)
        {
            Title = title;
            Description = description;
            Priority = priority;
            EstimatedHours = estimatedHours;
            UpdatedAt = DateTime.UtcNow;
            
            ValidateInvariant();
        }
        
        public void ChangeStatus(TaskStatus newStatus)
        {
            if (!IsValidStatusTransition(Status, newStatus))
                throw new BusinessRuleException("InvalidStatusTransition", 
                    $"Cannot change status from {Status} to {newStatus}");
            
            var oldStatus = Status;
            Status = newStatus;
            UpdatedAt = DateTime.UtcNow;
            
            // Handle status-specific logic
            switch (newStatus)
            {
                case TaskStatus.Completed:
                    CompletedAt = DateTime.UtcNow;
                    break;
                case TaskStatus.Active when oldStatus == TaskStatus.Completed:
                    CompletedAt = null;
                    break;
            }
        }
        
        public void SetDueDate(DateTime? dueDate)
        {
            if (dueDate.HasValue && dueDate.Value < DateTime.Today)
                throw new BusinessRuleException("PastDueDate", "Due date cannot be in the past");
                
            DueDate = dueDate;
            UpdatedAt = DateTime.UtcNow;
        }
        
        public void RecordWorkTime(decimal hours)
        {
            if (hours <= 0)
                throw new ArgumentException("Hours must be positive", nameof(hours));
                
            ActualHours = (ActualHours ?? 0) + hours;
            UpdatedAt = DateTime.UtcNow;
        }
        
        public void AddTag(string tag)
        {
            if (string.IsNullOrWhiteSpace(tag))
                throw new ArgumentException("Tag cannot be empty", nameof(tag));
                
            tag = tag.Trim().ToLowerInvariant();
            if (!_tags.Contains(tag))
            {
                _tags.Add(tag);
                UpdatedAt = DateTime.UtcNow;
            }
        }
        
        public void RemoveTag(string tag)
        {
            if (_tags.Remove(tag.Trim().ToLowerInvariant()))
                UpdatedAt = DateTime.UtcNow;
        }
        
        public void AddComment(string content, PersonId authorId)
        {
            var comment = TaskComment.Create(content, authorId);
            _comments.Add(comment);
            UpdatedAt = DateTime.UtcNow;
        }
        
        public void AddAttachment(string fileName, string filePath, long fileSize)
        {
            var attachment = TaskAttachment.Create(fileName, filePath, fileSize);
            _attachments.Add(attachment);
            UpdatedAt = DateTime.UtcNow;
        }
        
        // Properties
        public bool IsOverdue => DueDate.HasValue && DueDate.Value < DateTime.Today && Status != TaskStatus.Completed;
        public bool IsCompleted => Status == TaskStatus.Completed;
        public TimeSpan? TimeToCompletion => CompletedAt - CreatedAt;
        public bool IsOnTrack => !EstimatedHours.HasValue || !ActualHours.HasValue || ActualHours <= EstimatedHours * 1.1m;
        
        private static bool IsValidStatusTransition(TaskStatus from, TaskStatus to)
        {
            return (from, to) switch
            {
                (TaskStatus.Draft, TaskStatus.Active) => true,
                (TaskStatus.Active, TaskStatus.InProgress) => true,
                (TaskStatus.InProgress, TaskStatus.Review) => true,
                (TaskStatus.InProgress, TaskStatus.Active) => true,
                (TaskStatus.Review, TaskStatus.Completed) => true,
                (TaskStatus.Review, TaskStatus.InProgress) => true,
                (_, TaskStatus.Cancelled) => true,
                (TaskStatus.Cancelled, TaskStatus.Active) => true,
                (TaskStatus.Completed, TaskStatus.Archived) => true,
                _ => false
            };
        }
        
        private void ValidateInvariant()
        {
            var errors = new Dictionary<string, List<string>>();
            
            if (string.IsNullOrWhiteSpace(Title))
                errors.GetOrAdd("Title").Add("Title is required");
            else if (Title.Length > 200)
                errors.GetOrAdd("Title").Add("Title cannot exceed 200 characters");
                
            if (Description?.Length > 2000)
                errors.GetOrAdd("Description").Add("Description cannot exceed 2000 characters");
                
            if (EstimatedHours.HasValue && EstimatedHours.Value <= 0)
                errors.GetOrAdd("EstimatedHours").Add("Estimated hours must be positive");
            
            if (errors.Any())
                throw new ValidationException(errors.ToDictionary(k => k.Key, v => v.Value.ToArray()));
        }
    }
    
    public class TaskComment
    {
        public int Id { get; private set; }
        public string Content { get; private set; }
        public PersonId AuthorId { get; private set; }
        public DateTime CreatedAt { get; private set; }
        public DateTime? UpdatedAt { get; private set; }
        
        public User? Author { get; private set; }
        
        public static TaskComment Create(string content, PersonId authorId)
        {
            if (string.IsNullOrWhiteSpace(content))
                throw new ArgumentException("Content is required", nameof(content));
                
            return new TaskComment
            {
                Content = content,
                AuthorId = authorId,
                CreatedAt = DateTime.UtcNow
            };
        }
        
        public void UpdateContent(string newContent)
        {
            if (string.IsNullOrWhiteSpace(newContent))
                throw new ArgumentException("Content is required", nameof(newContent));
                
            Content = newContent;
            UpdatedAt = DateTime.UtcNow;
        }
    }
    
    public class TaskAttachment
    {
        public int Id { get; private set; }
        public string FileName { get; private set; }
        public string FilePath { get; private set; }
        public long FileSize { get; private set; }
        public string ContentType { get; private set; }
        public DateTime UploadedAt { get; private set; }
        
        public static TaskAttachment Create(string fileName, string filePath, long fileSize)
        {
            if (string.IsNullOrWhiteSpace(fileName))
                throw new ArgumentException("File name is required", nameof(fileName));
                
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path is required", nameof(filePath));
                
            if (fileSize <= 0)
                throw new ArgumentException("File size must be positive", nameof(fileSize));
            
            return new TaskAttachment
            {
                FileName = fileName,
                FilePath = filePath,
                FileSize = fileSize,
                ContentType = GetContentType(fileName),
                UploadedAt = DateTime.UtcNow
            };
        }
        
        private static string GetContentType(string fileName)
        {
            var extension = Path.GetExtension(fileName).ToLowerInvariant();
            return extension switch
            {
                ".txt" => "text/plain",
                ".pdf" => "application/pdf",
                ".doc" => "application/msword",
                ".docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".jpg" or ".jpeg" => "image/jpeg",
                ".png" => "image/png",
                ".gif" => "image/gif",
                _ => "application/octet-stream"
            };
        }
    }
    
    // ============ Repository Pattern with Async/Await ============
    
    public interface IRepository<TEntity, TKey> where TEntity : class
    {
        Task<TEntity?> GetByIdAsync(TKey id, CancellationToken cancellationToken = default);
        Task<IEnumerable<TEntity>> GetAllAsync(CancellationToken cancellationToken = default);
        Task<TEntity> AddAsync(TEntity entity, CancellationToken cancellationToken = default);
        Task<TEntity> UpdateAsync(TEntity entity, CancellationToken cancellationToken = default);
        Task<bool> DeleteAsync(TKey id, CancellationToken cancellationToken = default);
        Task<int> CountAsync(CancellationToken cancellationToken = default);
    }
    
    public interface IUserRepository : IRepository<User, PersonId>
    {
        Task<User?> GetByEmailAsync(Email email, CancellationToken cancellationToken = default);
        Task<IEnumerable<User>> GetByRoleAsync(UserRole role, CancellationToken cancellationToken = default);
        Task<IEnumerable<User>> SearchAsync(string searchTerm, CancellationToken cancellationToken = default);
        Task<IEnumerable<User>> GetActiveUsersAsync(CancellationToken cancellationToken = default);
    }
    
    public interface ITaskRepository : IRepository<Task, int>
    {
        Task<IEnumerable<Task>> GetByAssigneeAsync(PersonId assigneeId, CancellationToken cancellationToken = default);
        Task<IEnumerable<Task>> GetByStatusAsync(TaskStatus status, CancellationToken cancellationToken = default);
        Task<IEnumerable<Task>> GetOverdueTasksAsync(CancellationToken cancellationToken = default);
        Task<IEnumerable<Task>> GetByPriorityAsync(Priority priority, CancellationToken cancellationToken = default);
        Task<Dictionary<TaskStatus, int>> GetTaskCountByStatusAsync(CancellationToken cancellationToken = default);
    }
    
    // In-memory implementations for demo
    public class InMemoryUserRepository : IUserRepository
    {
        private readonly ConcurrentDictionary<PersonId, User> _users = new();
        private int _nextId = 1;
        
        public Task<User?> GetByIdAsync(PersonId id, CancellationToken cancellationToken = default)
        {
            _users.TryGetValue(id, out var user);
            return Task.FromResult(user);
        }
        
        public Task<IEnumerable<User>> GetAllAsync(CancellationToken cancellationToken = default)
        {
            return Task.FromResult(_users.Values.AsEnumerable());
        }
        
        public Task<User> AddAsync(User entity, CancellationToken cancellationToken = default)
        {
            if (entity.Id.Value == 0)
            {
                // Simulate auto-generated ID
                var id = new PersonId(Interlocked.Increment(ref _nextId));
                entity = entity with { Id = id };
            }
            
            _users.TryAdd(entity.Id, entity);
            return Task.FromResult(entity);
        }
        
        public Task<User> UpdateAsync(User entity, CancellationToken cancellationToken = default)
        {
            _users.AddOrUpdate(entity.Id, entity, (key, oldValue) => entity);
            return Task.FromResult(entity);
        }
        
        public Task<bool> DeleteAsync(PersonId id, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(_users.TryRemove(id, out _));
        }
        
        public Task<int> CountAsync(CancellationToken cancellationToken = default)
        {
            return Task.FromResult(_users.Count);
        }
        
        public Task<User?> GetByEmailAsync(Email email, CancellationToken cancellationToken = default)
        {
            var user = _users.Values.FirstOrDefault(u => u.Email == email);
            return Task.FromResult(user);
        }
        
        public Task<IEnumerable<User>> GetByRoleAsync(UserRole role, CancellationToken cancellationToken = default)
        {
            var users = _users.Values.Where(u => u.Role == role);
            return Task.FromResult(users);
        }
        
        public Task<IEnumerable<User>> SearchAsync(string searchTerm, CancellationToken cancellationToken = default)
        {
            var users = _users.Values.Where(u =>
                u.FullName.Contains(searchTerm, StringComparison.OrdinalIgnoreCase) ||
                u.Email.Value.Contains(searchTerm, StringComparison.OrdinalIgnoreCase));
            return Task.FromResult(users);
        }
        
        public Task<IEnumerable<User>> GetActiveUsersAsync(CancellationToken cancellationToken = default)
        {
            var users = _users.Values.Where(u => u.IsActive);
            return Task.FromResult(users);
        }
    }
    
    public class InMemoryTaskRepository : ITaskRepository
    {
        private readonly ConcurrentDictionary<int, Task> _tasks = new();
        private int _nextId = 1;
        
        public Task<Task?> GetByIdAsync(int id, CancellationToken cancellationToken = default)
        {
            _tasks.TryGetValue(id, out var task);
            return System.Threading.Tasks.Task.FromResult(task);
        }
        
        public Task<IEnumerable<Task>> GetAllAsync(CancellationToken cancellationToken = default)
        {
            return System.Threading.Tasks.Task.FromResult(_tasks.Values.AsEnumerable());
        }
        
        public Task<Task> AddAsync(Task entity, CancellationToken cancellationToken = default)
        {
            if (entity.Id == 0)
            {
                // Simulate auto-generated ID using reflection
                var idProperty = typeof(Task).GetProperty("Id", BindingFlags.Public | BindingFlags.Instance);
                idProperty?.SetValue(entity, Interlocked.Increment(ref _nextId));
            }
            
            _tasks.TryAdd(entity.Id, entity);
            return System.Threading.Tasks.Task.FromResult(entity);
        }
        
        public Task<Task> UpdateAsync(Task entity, CancellationToken cancellationToken = default)
        {
            _tasks.AddOrUpdate(entity.Id, entity, (key, oldValue) => entity);
            return System.Threading.Tasks.Task.FromResult(entity);
        }
        
        public Task<bool> DeleteAsync(int id, CancellationToken cancellationToken = default)
        {
            return System.Threading.Tasks.Task.FromResult(_tasks.TryRemove(id, out _));
        }
        
        public Task<int> CountAsync(CancellationToken cancellationToken = default)
        {
            return System.Threading.Tasks.Task.FromResult(_tasks.Count);
        }
        
        public Task<IEnumerable<Task>> GetByAssigneeAsync(PersonId assigneeId, CancellationToken cancellationToken = default)
        {
            var tasks = _tasks.Values.Where(t => t.AssignedToId == assigneeId);
            return System.Threading.Tasks.Task.FromResult(tasks);
        }
        
        public Task<IEnumerable<Task>> GetByStatusAsync(TaskStatus status, CancellationToken cancellationToken = default)
        {
            var tasks = _tasks.Values.Where(t => t.Status == status);
            return System.Threading.Tasks.Task.FromResult(tasks);
        }
        
        public Task<IEnumerable<Task>> GetOverdueTasksAsync(CancellationToken cancellationToken = default)
        {
            var tasks = _tasks.Values.Where(t => t.IsOverdue);
            return System.Threading.Tasks.Task.FromResult(tasks);
        }
        
        public Task<IEnumerable<Task>> GetByPriorityAsync(Priority priority, CancellationToken cancellationToken = default)
        {
            var tasks = _tasks.Values.Where(t => t.Priority == priority);
            return System.Threading.Tasks.Task.FromResult(tasks);
        }
        
        public Task<Dictionary<TaskStatus, int>> GetTaskCountByStatusAsync(CancellationToken cancellationToken = default)
        {
            var counts = _tasks.Values
                .GroupBy(t => t.Status)
                .ToDictionary(g => g.Key, g => g.Count());
            return System.Threading.Tasks.Task.FromResult(counts);
        }
    }
    
    // ============ Service Layer with Business Logic ============
    
    public interface IUserService
    {
        Task<User> CreateUserAsync(string firstName, string lastName, Email email, 
                                 DateTime dateOfBirth, Address address, UserRole role = UserRole.User);
        Task<User> GetUserByIdAsync(PersonId id);
        Task<User?> GetUserByEmailAsync(Email email);
        Task<User> UpdateUserAsync(PersonId id, string firstName, string lastName, Email email, Address address);
        Task<bool> DeleteUserAsync(PersonId id);
        Task<IEnumerable<User>> SearchUsersAsync(string searchTerm);
        Task<IEnumerable<User>> GetActiveUsersAsync();
        Task<User> AuthenticateUserAsync(Email email, string password);
        Task<User> ChangeUserRoleAsync(PersonId id, UserRole newRole, User currentUser);
    }
    
    public interface ITaskService
    {
        Task<Task> CreateTaskAsync(string title, string description, Priority priority, 
                                 PersonId assignedToId, PersonId createdById, DateTime? dueDate = null);
        Task<Task> GetTaskByIdAsync(int id);
        Task<Task> UpdateTaskAsync(int id, string title, string description, Priority priority, User currentUser);
        Task<Task> ChangeTaskStatusAsync(int id, TaskStatus newStatus, User currentUser);
        Task<bool> DeleteTaskAsync(int id, User currentUser);
        Task<IEnumerable<Task>> GetTasksByUserAsync(PersonId userId);
        Task<IEnumerable<Task>> GetOverdueTasksAsync();
        Task<Dictionary<string, object>> GetTaskStatisticsAsync();
    }
    
    public class UserService : IUserService
    {
        private readonly IUserRepository _userRepository;
        private readonly IPasswordHasher _passwordHasher;
        private readonly ILogger<UserService> _logger;
        
        public UserService(IUserRepository userRepository, IPasswordHasher passwordHasher, ILogger<UserService> logger)
        {
            _userRepository = userRepository;
            _passwordHasher = passwordHasher;
            _logger = logger;
        }
        
        public async Task<User> CreateUserAsync(string firstName, string lastName, Email email, 
                                               DateTime dateOfBirth, Address address, UserRole role = UserRole.User)
        {
            _logger.LogInformation("Creating user with email {Email}", email);
            
            // Check if user already exists
            var existingUser = await _userRepository.GetByEmailAsync(email);
            if (existingUser != null)
                throw new ValidationException("Email", "A user with this email already exists");
            
            var user = User.Create(firstName, lastName, email, dateOfBirth, address, role);
            var savedUser = await _userRepository.AddAsync(user);
            
            _logger.LogInformation("User created successfully with ID {UserId}", savedUser.Id);
            return savedUser;
        }
        
        public async Task<User> GetUserByIdAsync(PersonId id)
        {
            var user = await _userRepository.GetByIdAsync(id);
            if (user == null)
                throw new EntityNotFoundException(nameof(User), id);
            
            return user;
        }
        
        public async Task<User?> GetUserByEmailAsync(Email email)
        {
            return await _userRepository.GetByEmailAsync(email);
        }
        
        public async Task<User> UpdateUserAsync(PersonId id, string firstName, string lastName, Email email, Address address)
        {
            _logger.LogInformation("Updating user {UserId}", id);
            
            var user = await GetUserByIdAsync(id);
            
            // Check if email is taken by another user
            if (user.Email != email)
            {
                var existingUser = await _userRepository.GetByEmailAsync(email);
                if (existingUser != null && existingUser.Id != id)
                    throw new ValidationException("Email", "A user with this email already exists");
            }
            
            user.UpdateProfile(firstName, lastName, email, address);
            var updatedUser = await _userRepository.UpdateAsync(user);
            
            _logger.LogInformation("User {UserId} updated successfully", id);
            return updatedUser;
        }
        
        public async Task<bool> DeleteUserAsync(PersonId id)
        {
            _logger.LogInformation("Deleting user {UserId}", id);
            
            var user = await GetUserByIdAsync(id);
            var result = await _userRepository.DeleteAsync(id);
            
            if (result)
                _logger.LogInformation("User {UserId} deleted successfully", id);
            
            return result;
        }
        
        public async Task<IEnumerable<User>> SearchUsersAsync(string searchTerm)
        {
            return await _userRepository.SearchAsync(searchTerm);
        }
        
        public async Task<IEnumerable<User>> GetActiveUsersAsync()
        {
            return await _userRepository.GetActiveUsersAsync();
        }
        
        public async Task<User> AuthenticateUserAsync(Email email, string password)
        {
            _logger.LogInformation("Authenticating user with email {Email}", email);
            
            var user = await _userRepository.GetByEmailAsync(email);
            if (user == null || !_passwordHasher.VerifyPassword(password, user.PasswordHash))
                throw new BusinessRuleException("Authentication", "Invalid email or password");
            
            if (!user.IsActive)
                throw new BusinessRuleException("Authentication", "Account is deactivated");
            
            user.RecordLogin();
            await _userRepository.UpdateAsync(user);
            
            _logger.LogInformation("User {UserId} authenticated successfully", user.Id);
            return user;
        }
        
        public async Task<User> ChangeUserRoleAsync(PersonId id, UserRole newRole, User currentUser)
        {
            if (!currentUser.HasPermission("manage:users"))
                throw new BusinessRuleException("Authorization", "Insufficient permissions to change user roles");
            
            var user = await GetUserByIdAsync(id);
            user.ChangeRole(newRole);
            
            var updatedUser = await _userRepository.UpdateAsync(user);
            _logger.LogInformation("User {UserId} role changed to {Role} by {CurrentUserId}", id, newRole, currentUser.Id);
            
            return updatedUser;
        }
    }
    
    public class TaskService : ITaskService
    {
        private readonly ITaskRepository _taskRepository;
        private readonly IUserRepository _userRepository;
        private readonly ILogger<TaskService> _logger;
        
        public TaskService(ITaskRepository taskRepository, IUserRepository userRepository, ILogger<TaskService> logger)
        {
            _taskRepository = taskRepository;
            _userRepository = userRepository;
            _logger = logger;
        }
        
        public async Task<Task> CreateTaskAsync(string title, string description, Priority priority, 
                                               PersonId assignedToId, PersonId createdById, DateTime? dueDate = null)
        {
            _logger.LogInformation("Creating task '{Title}' assigned to {AssignedToId}", title, assignedToId);
            
            // Validate that users exist
            await GetUserByIdAsync(assignedToId);
            await GetUserByIdAsync(createdById);
            
            var task = Task.Create(title, description, priority, assignedToId, createdById);
            if (dueDate.HasValue)
                task.SetDueDate(dueDate);
            
            var savedTask = await _taskRepository.AddAsync(task);
            
            _logger.LogInformation("Task {TaskId} created successfully", savedTask.Id);
            return savedTask;
        }
        
        public async Task<Task> GetTaskByIdAsync(int id)
        {
            var task = await _taskRepository.GetByIdAsync(id);
            if (task == null)
                throw new EntityNotFoundException(nameof(Task), id);
            
            return task;
        }
        
        public async Task<Task> UpdateTaskAsync(int id, string title, string description, Priority priority, User currentUser)
        {
            _logger.LogInformation("Updating task {TaskId}", id);
            
            var task = await GetTaskByIdAsync(id);
            
            if (!CanModifyTask(task, currentUser))
                throw new BusinessRuleException("Authorization", "You do not have permission to modify this task");
            
            task.UpdateDetails(title, description, priority);
            var updatedTask = await _taskRepository.UpdateAsync(task);
            
            _logger.LogInformation("Task {TaskId} updated successfully", id);
            return updatedTask;
        }
        
        public async Task<Task> ChangeTaskStatusAsync(int id, TaskStatus newStatus, User currentUser)
        {
            _logger.LogInformation("Changing task {TaskId} status to {Status}", id, newStatus);
            
            var task = await GetTaskByIdAsync(id);
            
            if (!CanModifyTask(task, currentUser))
                throw new BusinessRuleException("Authorization", "You do not have permission to modify this task");
            
            task.ChangeStatus(newStatus);
            var updatedTask = await _taskRepository.UpdateAsync(task);
            
            _logger.LogInformation("Task {TaskId} status changed to {Status}", id, newStatus);
            return updatedTask;
        }
        
        public async Task<bool> DeleteTaskAsync(int id, User currentUser)
        {
            _logger.LogInformation("Deleting task {TaskId}", id);
            
            var task = await GetTaskByIdAsync(id);
            
            if (!CanDeleteTask(task, currentUser))
                throw new BusinessRuleException("Authorization", "You do not have permission to delete this task");
            
            var result = await _taskRepository.DeleteAsync(id);
            
            if (result)
                _logger.LogInformation("Task {TaskId} deleted successfully", id);
            
            return result;
        }
        
        public async Task<IEnumerable<Task>> GetTasksByUserAsync(PersonId userId)
        {
            return await _taskRepository.GetByAssigneeAsync(userId);
        }
        
        public async Task<IEnumerable<Task>> GetOverdueTasksAsync()
        {
            return await _taskRepository.GetOverdueTasksAsync();
        }
        
        public async Task<Dictionary<string, object>> GetTaskStatisticsAsync()
        {
            var statusCounts = await _taskRepository.GetTaskCountByStatusAsync();
            var totalTasks = await _taskRepository.CountAsync();
            var overdueTasks = await GetOverdueTasksAsync();
            
            return new Dictionary<string, object>
            {
                ["totalTasks"] = totalTasks,
                ["tasksByStatus"] = statusCounts,
                ["overdueTasks"] = overdueTasks.Count(),
                ["completionRate"] = totalTasks > 0 ? 
                    (double)statusCounts.GetValueOrDefault(TaskStatus.Completed, 0) / totalTasks * 100 : 0
            };
        }
        
        private async Task<User> GetUserByIdAsync(PersonId id)
        {
            var user = await _userRepository.GetByIdAsync(id);
            if (user == null)
                throw new EntityNotFoundException(nameof(User), id);
            
            return user;
        }
        
        private static bool CanModifyTask(Task task, User user)
        {
            return task.AssignedToId == user.Id || 
                   task.CreatedById == user.Id || 
                   user.HasPermission("write:all");
        }
        
        private static bool CanDeleteTask(Task task, User user)
        {
            return task.CreatedById == user.Id || 
                   user.HasPermission("delete:all");
        }
    }
    
    // ============ Supporting Services ============
    
    public interface IPasswordHasher
    {
        string HashPassword(string password);
        bool VerifyPassword(string password, string? hash);
    }
    
    public class BCryptPasswordHasher : IPasswordHasher
    {
        public string HashPassword(string password)
        {
            return BCrypt.Net.BCrypt.HashPassword(password);
        }
        
        public bool VerifyPassword(string password, string? hash)
        {
            return !string.IsNullOrEmpty(hash) && BCrypt.Net.BCrypt.Verify(password, hash);
        }
    }
    
    // ============ Extension Methods ============
    
    public static class Extensions
    {
        public static void GetOrAdd<TKey, TValue>(this Dictionary<TKey, List<TValue>> dictionary, TKey key) 
            where TKey : notnull
        {
            if (!dictionary.ContainsKey(key))
                dictionary[key] = new List<TValue>();
        }
        
        public static PersonId NewId(this PersonId _)
        {
            return new PersonId(Random.Shared.Next(1, int.MaxValue));
        }
        
        public static async Task<IEnumerable<TResult>> SelectAsync<TSource, TResult>(
            this IEnumerable<TSource> source, 
            Func<TSource, Task<TResult>> selector)
        {
            var tasks = source.Select(selector);
            return await System.Threading.Tasks.Task.WhenAll(tasks);
        }
        
        public static async Task<IEnumerable<TSource>> WhereAsync<TSource>(
            this IEnumerable<TSource> source,
            Func<TSource, Task<bool>> predicate)
        {
            var results = new List<TSource>();
            foreach (var item in source)
            {
                if (await predicate(item))
                    results.Add(item);
            }
            return results;
        }
    }
    
    // ============ Configuration and Dependency Injection ============
    
    public class Startup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            // Repository registrations
            services.AddSingleton<IUserRepository, InMemoryUserRepository>();
            services.AddSingleton<ITaskRepository, InMemoryTaskRepository>();
            
            // Service registrations
            services.AddScoped<IUserService, UserService>();
            services.AddScoped<ITaskService, TaskService>();
            services.AddScoped<IPasswordHasher, BCryptPasswordHasher>();
            
            // Logging
            services.AddLogging(builder => builder.AddConsole());
            
            // HTTP client
            services.AddHttpClient();
            
            // Background services
            services.AddHostedService<TaskReminderService>();
        }
    }
    
    // ============ Background Services ============
    
    public class TaskReminderService : BackgroundService
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<TaskReminderService> _logger;
        private readonly TimeSpan _interval = TimeSpan.FromHours(1);
        
        public TaskReminderService(IServiceProvider serviceProvider, ILogger<TaskReminderService> logger)
        {
            _serviceProvider = serviceProvider;
            _logger = logger;
        }
        
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await ProcessOverdueTasksAsync();
                    await System.Threading.Tasks.Task.Delay(_interval, stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error occurred while processing overdue tasks");
                    await System.Threading.Tasks.Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
                }
            }
        }
        
        private async Task ProcessOverdueTasksAsync()
        {
            using var scope = _serviceProvider.CreateScope();
            var taskService = scope.ServiceProvider.GetRequiredService<ITaskService>();
            
            var overdueTasks = await taskService.GetOverdueTasksAsync();
            var overdueCount = overdueTasks.Count();
            
            if (overdueCount > 0)
            {
                _logger.LogInformation("Found {Count} overdue tasks", overdueCount);
                
                // Here you could send notifications, create alerts, etc.
                foreach (var task in overdueTasks)
                {
                    _logger.LogWarning("Task {TaskId} '{Title}' is overdue (due: {DueDate})", 
                        task.Id, task.Title, task.DueDate);
                }
            }
        }
    }
    
    // ============ Web API Controllers (Simplified) ============
    
    [ApiController]
    [Route("api/[controller]")]
    public class UsersController : ControllerBase
    {
        private readonly IUserService _userService;
        private readonly ILogger<UsersController> _logger;
        
        public UsersController(IUserService userService, ILogger<UsersController> logger)
        {
            _userService = userService;
            _logger = logger;
        }
        
        [HttpGet]
        public async Task<ActionResult<IEnumerable<UserDto>>> GetUsers([FromQuery] string? search = null)
        {
            try
            {
                var users = string.IsNullOrWhiteSpace(search) 
                    ? await _userService.GetActiveUsersAsync()
                    : await _userService.SearchUsersAsync(search);
                
                var userDtos = users.Select(UserDto.FromDomain);
                return Ok(userDtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving users");
                return StatusCode(500, "An error occurred while retrieving users");
            }
        }
        
        [HttpGet("{id:int}")]
        public async Task<ActionResult<UserDto>> GetUser(int id)
        {
            try
            {
                var user = await _userService.GetUserByIdAsync(new PersonId(id));
                return Ok(UserDto.FromDomain(user));
            }
            catch (EntityNotFoundException)
            {
                return NotFound();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving user {UserId}", id);
                return StatusCode(500, "An error occurred while retrieving the user");
            }
        }
        
        [HttpPost]
        public async Task<ActionResult<UserDto>> CreateUser([FromBody] CreateUserRequest request)
        {
            try
            {
                var address = new Address(request.Street, request.City, request.State, request.ZipCode, request.Country);
                var user = await _userService.CreateUserAsync(
                    request.FirstName, 
                    request.LastName, 
                    new Email(request.Email), 
                    request.DateOfBirth, 
                    address, 
                    request.Role);
                
                var userDto = UserDto.FromDomain(user);
                return CreatedAtAction(nameof(GetUser), new { id = user.Id.Value }, userDto);
            }
            catch (ValidationException ex)
            {
                return BadRequest(ex.Errors);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating user");
                return StatusCode(500, "An error occurred while creating the user");
            }
        }
    }
    
    // ============ DTOs and Request/Response Models ============
    
    public class UserDto
    {
        public int Id { get; set; }
        public string FirstName { get; set; } = string.Empty;
        public string LastName { get; set; } = string.Empty;
        public string FullName { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public DateTime DateOfBirth { get; set; }
        public AddressDto Address { get; set; } = new();
        public UserRole Role { get; set; }
        public bool IsActive { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? LastLoginAt { get; set; }
        public int Age { get; set; }
        public bool IsAdult { get; set; }
        
        public static UserDto FromDomain(User user)
        {
            return new UserDto
            {
                Id = user.Id.Value,
                FirstName = user.FirstName,
                LastName = user.LastName,
                FullName = user.FullName,
                Email = user.Email.Value,
                DateOfBirth = user.DateOfBirth,
                Address = AddressDto.FromDomain(user.Address),
                Role = user.Role,
                IsActive = user.IsActive,
                CreatedAt = user.CreatedAt,
                LastLoginAt = user.LastLoginAt,
                Age = user.Age,
                IsAdult = user.IsAdult
            };
        }
    }
    
    public class AddressDto
    {
        public string Street { get; set; } = string.Empty;
        public string City { get; set; } = string.Empty;
        public string State { get; set; } = string.Empty;
        public string ZipCode { get; set; } = string.Empty;
        public string Country { get; set; } = string.Empty;
        public string FullAddress { get; set; } = string.Empty;
        
        public static AddressDto FromDomain(Address address)
        {
            return new AddressDto
            {
                Street = address.Street,
                City = address.City,
                State = address.State,
                ZipCode = address.ZipCode,
                Country = address.Country,
                FullAddress = address.FullAddress
            };
        }
    }
    
    public class CreateUserRequest
    {
        [Required]
        [StringLength(50, MinimumLength = 2)]
        public string FirstName { get; set; } = string.Empty;
        
        [Required]
        [StringLength(50, MinimumLength = 2)]
        public string LastName { get; set; } = string.Empty;
        
        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;
        
        [Required]
        public DateTime DateOfBirth { get; set; }
        
        [Required]
        public string Street { get; set; } = string.Empty;
        
        [Required]
        public string City { get; set; } = string.Empty;
        
        [Required]
        public string State { get; set; } = string.Empty;
        
        [Required]
        public string ZipCode { get; set; } = string.Empty;
        
        public string Country { get; set; } = "USA";
        
        public UserRole Role { get; set; } = UserRole.User;
    }
    
    // ============ Demo Application ============
    
    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("=== C# Comprehensive Examples Demo ===\n");
            
            // Setup dependency injection
            var services = new ServiceCollection();
            var startup = new Startup();
            startup.ConfigureServices(services);
            
            var serviceProvider = services.BuildServiceProvider();
            
            try
            {
                await RunDemoAsync(serviceProvider);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Demo error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            finally
            {
                await serviceProvider.DisposeAsync();
            }
            
            Console.WriteLine("\n=== C# Features Demonstrated ===");
            Console.WriteLine("- Records and value types (C# 9.0+)");
            Console.WriteLine("- Pattern matching and switch expressions");
            Console.WriteLine("- Nullable reference types");
            Console.WriteLine("- Async/await and Task-based programming");
            Console.WriteLine("- Dependency injection and IoC containers");
            Console.WriteLine("- Repository and service patterns");
            Console.WriteLine("- Domain-driven design principles");
            Console.WriteLine("- SOLID principles implementation");
            Console.WriteLine("- Background services and hosted services");
            Console.WriteLine("- Extension methods and LINQ");
            Console.WriteLine("- Custom exceptions and error handling");
            Console.WriteLine("- Generic types and constraints");
            Console.WriteLine("- Reflection and attributes");
            Console.WriteLine("- JSON serialization");
            Console.WriteLine("- Concurrent collections and thread safety");
            Console.WriteLine("- Modern C# language features");
        }
        
        private static async Task RunDemoAsync(IServiceProvider serviceProvider)
        {
            var userService = serviceProvider.GetRequiredService<IUserService>();
            var taskService = serviceProvider.GetRequiredService<ITaskService>();
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
            
            // Create users
            Console.WriteLine("Creating users...");
            
            var admin = await userService.CreateUserAsync(
                "Admin", "User", "admin@example.com", new DateTime(1990, 1, 1),
                new Address("123 Main St", "Anytown", "ST", "12345"), UserRole.Admin);
            
            var user1 = await userService.CreateUserAsync(
                "Alice", "Johnson", "alice@example.com", new DateTime(1995, 5, 15),
                new Address("456 Oak Ave", "Somewhere", "ST", "67890"), UserRole.User);
            
            var user2 = await userService.CreateUserAsync(
                "Bob", "Smith", "bob@example.com", new DateTime(1988, 12, 3),
                new Address("789 Pine Rd", "Elsewhere", "ST", "54321"), UserRole.User);
            
            Console.WriteLine($"Created {(await userService.GetActiveUsersAsync()).Count()} users\n");
            
            // Create tasks
            Console.WriteLine("Creating tasks...");
            
            var task1 = await taskService.CreateTaskAsync(
                "Implement user authentication",
                "Add secure login system with JWT tokens",
                Priority.High,
                user1.Id,
                admin.Id,
                DateTime.Today.AddDays(7));
            
            var task2 = await taskService.CreateTaskAsync(
                "Design database schema",
                "Create comprehensive database design for the application",
                Priority.Medium,
                user2.Id,
                admin.Id);
            
            var task3 = await taskService.CreateTaskAsync(
                "Write API documentation",
                "Document all REST API endpoints with OpenAPI/Swagger",
                Priority.Low,
                user1.Id,
                admin.Id,
                DateTime.Today.AddDays(14));
            
            Console.WriteLine("Created 3 tasks\n");
            
            // Update task statuses
            Console.WriteLine("Updating task statuses...");
            await taskService.ChangeTaskStatusAsync(task1.Id, TaskStatus.InProgress, user1);
            await taskService.ChangeTaskStatusAsync(task2.Id, TaskStatus.Completed, user2);
            
            // Get statistics
            Console.WriteLine("Task Statistics:");
            var stats = await taskService.GetTaskStatisticsAsync();
            foreach (var stat in stats)
            {
                Console.WriteLine($"  {stat.Key}: {stat.Value}");
            }
            
            // Test search
            Console.WriteLine("\nSearching users with 'alice':");
            var searchResults = await userService.SearchUsersAsync("alice");
            foreach (var user in searchResults)
            {
                Console.WriteLine($"- {user.FullName} ({user.Email})");
            }
            
            // Test pattern matching and records
            Console.WriteLine("\nTesting pattern matching and records:");
            
            var userRole = UserRole.Admin;
            var permissions = userRole switch
            {
                UserRole.Guest => "Read only",
                UserRole.User => "Read and write own data",
                UserRole.Moderator => "Moderate content",
                UserRole.Admin => "Full access",
                UserRole.SuperAdmin => "System administration",
                _ => "No permissions"
            };
            Console.WriteLine($"Admin permissions: {permissions}");
            
            // Test record patterns
            var address = new Address("123 Main St", "Anytown", "ST", "12345");
            var addressInfo = address switch
            {
                { State: "CA" } => "California address",
                { State: "NY" } => "New York address",
                { Country: "USA" } => "US address",
                _ => "International address"
            };
            Console.WriteLine($"Address type: {addressInfo}");
            
            // Test email value object
            Email email = "test@example.com";
            Console.WriteLine($"Email domain: {email.Domain}");
            Console.WriteLine($"Email local part: {email.LocalPart}");
            
            // Test JSON serialization
            Console.WriteLine("\nUser JSON:");
            var userJson = JsonSerializer.Serialize(UserDto.FromDomain(user1), new JsonSerializerOptions 
            { 
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            Console.WriteLine(userJson);
            
            logger.LogInformation("Demo completed successfully");
        }
    }
}