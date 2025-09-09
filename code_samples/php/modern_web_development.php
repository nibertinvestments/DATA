<?php
/**
 * Comprehensive PHP Web Development Examples
 * Demonstrates modern PHP features, frameworks patterns, and best practices
 */

declare(strict_types=1);

require_once __DIR__ . '/vendor/autoload.php'; // Assuming Composer

use DateTime;
use DateTimeImmutable;
use PDO;
use PDOException;
use Exception;
use InvalidArgumentException;
use JsonException;

// ============ Modern PHP Features and Type System ============

/**
 * Enum for user roles (PHP 8.1+)
 */
enum UserRole: string
{
    case ADMIN = 'admin';
    case MODERATOR = 'moderator';
    case USER = 'user';
    case GUEST = 'guest';
    
    public function getDisplayName(): string
    {
        return match($this) {
            self::ADMIN => 'Administrator',
            self::MODERATOR => 'Moderator',
            self::USER => 'Regular User',
            self::GUEST => 'Guest User'
        };
    }
    
    public function getPermissions(): array
    {
        return match($this) {
            self::ADMIN => ['read', 'write', 'delete', 'manage_users', 'system_admin'],
            self::MODERATOR => ['read', 'write', 'delete', 'moderate_content'],
            self::USER => ['read', 'write'],
            self::GUEST => ['read']
        };
    }
    
    public function hasPermission(string $permission): bool
    {
        return in_array($permission, $this->getPermissions(), true);
    }
}

/**
 * Status enum for tasks
 */
enum TaskStatus: string
{
    case PENDING = 'pending';
    case IN_PROGRESS = 'in_progress';
    case REVIEW = 'review';
    case COMPLETED = 'completed';
    case CANCELLED = 'cancelled';
    
    public function getColor(): string
    {
        return match($this) {
            self::PENDING => '#fbbf24',
            self::IN_PROGRESS => '#3b82f6',
            self::REVIEW => '#f97316',
            self::COMPLETED => '#10b981',
            self::CANCELLED => '#ef4444'
        };
    }
}

/**
 * Priority enum
 */
enum Priority: int
{
    case LOW = 1;
    case MEDIUM = 2;
    case HIGH = 3;
    case URGENT = 4;
    
    public function getLabel(): string
    {
        return match($this) {
            self::LOW => 'Low',
            self::MEDIUM => 'Medium',
            self::HIGH => 'High',
            self::URGENT => 'Urgent'
        };
    }
}

// ============ Custom Exceptions ============

class ValidationException extends Exception
{
    private array $errors;
    
    public function __construct(array $errors, string $message = 'Validation failed')
    {
        $this->errors = $errors;
        parent::__construct($message);
    }
    
    public function getErrors(): array
    {
        return $this->errors;
    }
}

class EntityNotFoundException extends Exception
{
    public function __construct(string $entityType, int|string $id)
    {
        parent::__construct("$entityType with ID '$id' not found");
    }
}

class DatabaseException extends Exception
{
    public function __construct(string $message, int $code = 0, ?Exception $previous = null)
    {
        parent::__construct("Database error: $message", $code, $previous);
    }
}

class AuthenticationException extends Exception
{
    public function __construct(string $message = 'Authentication failed')
    {
        parent::__construct($message);
    }
}

class AuthorizationException extends Exception
{
    public function __construct(string $message = 'Access denied')
    {
        parent::__construct($message);
    }
}

// ============ Traits for Common Functionality ============

trait Timestampable
{
    private DateTimeImmutable $createdAt;
    private DateTimeImmutable $updatedAt;
    
    protected function initializeTimestamps(): void
    {
        $now = new DateTimeImmutable();
        $this->createdAt = $now;
        $this->updatedAt = $now;
    }
    
    public function getCreatedAt(): DateTimeImmutable
    {
        return $this->createdAt;
    }
    
    public function getUpdatedAt(): DateTimeImmutable
    {
        return $this->updatedAt;
    }
    
    protected function touch(): void
    {
        $this->updatedAt = new DateTimeImmutable();
    }
}

trait Validatable
{
    abstract public function validate(): array;
    
    public function isValid(): bool
    {
        return empty($this->validate());
    }
    
    public function validateOrThrow(): void
    {
        $errors = $this->validate();
        if (!empty($errors)) {
            throw new ValidationException($errors);
        }
    }
}

trait JsonSerializable
{
    public function toArray(): array
    {
        $reflection = new ReflectionClass($this);
        $properties = $reflection->getProperties(ReflectionProperty::IS_PRIVATE | ReflectionProperty::IS_PROTECTED | ReflectionProperty::IS_PUBLIC);
        
        $data = [];
        foreach ($properties as $property) {
            $property->setAccessible(true);
            $value = $property->getValue($this);
            
            // Handle special types
            if ($value instanceof DateTimeImmutable || $value instanceof DateTime) {
                $value = $value->format('Y-m-d H:i:s');
            } elseif ($value instanceof BackedEnum) {
                $value = $value->value;
            }
            
            $data[$property->getName()] = $value;
        }
        
        return $data;
    }
    
    public function toJson(): string
    {
        return json_encode($this->toArray(), JSON_THROW_ON_ERROR);
    }
}

// ============ Domain Models ============

readonly class UserId
{
    public function __construct(private int $value)
    {
        if ($value <= 0) {
            throw new InvalidArgumentException('User ID must be positive');
        }
    }
    
    public function getValue(): int
    {
        return $this->value;
    }
    
    public function equals(UserId $other): bool
    {
        return $this->value === $other->value;
    }
    
    public function __toString(): string
    {
        return (string) $this->value;
    }
}

readonly class Email
{
    public function __construct(private string $value)
    {
        if (!filter_var($value, FILTER_VALIDATE_EMAIL)) {
            throw new InvalidArgumentException("Invalid email address: $value");
        }
    }
    
    public function getValue(): string
    {
        return $this->value;
    }
    
    public function getDomain(): string
    {
        return substr($this->value, strpos($this->value, '@') + 1);
    }
    
    public function getLocalPart(): string
    {
        return substr($this->value, 0, strpos($this->value, '@'));
    }
    
    public function __toString(): string
    {
        return $this->value;
    }
}

class User
{
    use Timestampable, Validatable, JsonSerializable;
    
    private ?UserId $id = null;
    private UserRole $role;
    private bool $isActive = true;
    private ?DateTimeImmutable $lastLoginAt = null;
    private array $metadata = [];
    
    public function __construct(
        private string $name,
        private Email $email,
        private int $age,
        UserRole $role = UserRole::USER
    ) {
        $this->role = $role;
        $this->initializeTimestamps();
        $this->validateOrThrow();
    }
    
    public function validate(): array
    {
        $errors = [];
        
        if (empty(trim($this->name))) {
            $errors['name'] = 'Name cannot be empty';
        } elseif (strlen($this->name) < 2) {
            $errors['name'] = 'Name must be at least 2 characters';
        } elseif (strlen($this->name) > 100) {
            $errors['name'] = 'Name cannot exceed 100 characters';
        }
        
        if ($this->age < 0 || $this->age > 150) {
            $errors['age'] = 'Age must be between 0 and 150';
        }
        
        return $errors;
    }
    
    // Getters
    public function getId(): ?UserId { return $this->id; }
    public function getName(): string { return $this->name; }
    public function getEmail(): Email { return $this->email; }
    public function getAge(): int { return $this->age; }
    public function getRole(): UserRole { return $this->role; }
    public function isActive(): bool { return $this->isActive; }
    public function getLastLoginAt(): ?DateTimeImmutable { return $this->lastLoginAt; }
    public function getMetadata(): array { return $this->metadata; }
    
    // Setters with validation
    public function setId(UserId $id): void
    {
        if ($this->id !== null) {
            throw new InvalidArgumentException('User ID cannot be changed once set');
        }
        $this->id = $id;
    }
    
    public function updateProfile(string $name, Email $email, int $age): void
    {
        $this->name = $name;
        $this->email = $email;
        $this->age = $age;
        $this->touch();
        $this->validateOrThrow();
    }
    
    public function changeRole(UserRole $role): void
    {
        $this->role = $role;
        $this->touch();
    }
    
    public function deactivate(): void
    {
        $this->isActive = false;
        $this->touch();
    }
    
    public function activate(): void
    {
        $this->isActive = true;
        $this->touch();
    }
    
    public function recordLogin(): void
    {
        $this->lastLoginAt = new DateTimeImmutable();
        $this->touch();
    }
    
    public function setMetadata(string $key, mixed $value): void
    {
        $this->metadata[$key] = $value;
        $this->touch();
    }
    
    public function getMetadataValue(string $key): mixed
    {
        return $this->metadata[$key] ?? null;
    }
    
    public function hasPermission(string $permission): bool
    {
        return $this->role->hasPermission($permission);
    }
    
    public function isAdult(): bool
    {
        return $this->age >= 18;
    }
    
    public function canAccessAdminPanel(): bool
    {
        return $this->role === UserRole::ADMIN && $this->isActive;
    }
}

class Task
{
    use Timestampable, Validatable, JsonSerializable;
    
    private ?int $id = null;
    private TaskStatus $status;
    private ?DateTimeImmutable $dueDate = null;
    private ?DateTimeImmutable $completedAt = null;
    private array $tags = [];
    private array $attachments = [];
    
    public function __construct(
        private string $title,
        private string $description,
        private Priority $priority,
        private UserId $assignedTo,
        private UserId $createdBy
    ) {
        $this->status = TaskStatus::PENDING;
        $this->initializeTimestamps();
        $this->validateOrThrow();
    }
    
    public function validate(): array
    {
        $errors = [];
        
        if (empty(trim($this->title))) {
            $errors['title'] = 'Title cannot be empty';
        } elseif (strlen($this->title) > 200) {
            $errors['title'] = 'Title cannot exceed 200 characters';
        }
        
        if (strlen($this->description) > 2000) {
            $errors['description'] = 'Description cannot exceed 2000 characters';
        }
        
        if ($this->dueDate && $this->dueDate < new DateTimeImmutable('today')) {
            $errors['dueDate'] = 'Due date cannot be in the past';
        }
        
        return $errors;
    }
    
    // Getters
    public function getId(): ?int { return $this->id; }
    public function getTitle(): string { return $this->title; }
    public function getDescription(): string { return $this->description; }
    public function getPriority(): Priority { return $this->priority; }
    public function getStatus(): TaskStatus { return $this->status; }
    public function getAssignedTo(): UserId { return $this->assignedTo; }
    public function getCreatedBy(): UserId { return $this->createdBy; }
    public function getDueDate(): ?DateTimeImmutable { return $this->dueDate; }
    public function getCompletedAt(): ?DateTimeImmutable { return $this->completedAt; }
    public function getTags(): array { return $this->tags; }
    public function getAttachments(): array { return $this->attachments; }
    
    public function setId(int $id): void
    {
        if ($this->id !== null) {
            throw new InvalidArgumentException('Task ID cannot be changed once set');
        }
        $this->id = $id;
    }
    
    public function updateDetails(string $title, string $description, Priority $priority): void
    {
        $this->title = $title;
        $this->description = $description;
        $this->priority = $priority;
        $this->touch();
        $this->validateOrThrow();
    }
    
    public function setDueDate(DateTimeImmutable $dueDate): void
    {
        $this->dueDate = $dueDate;
        $this->touch();
        $this->validateOrThrow();
    }
    
    public function updateStatus(TaskStatus $status): void
    {
        $this->status = $status;
        
        if ($status === TaskStatus::COMPLETED) {
            $this->completedAt = new DateTimeImmutable();
        } else {
            $this->completedAt = null;
        }
        
        $this->touch();
    }
    
    public function addTag(string $tag): void
    {
        $tag = trim($tag);
        if (!empty($tag) && !in_array($tag, $this->tags, true)) {
            $this->tags[] = $tag;
            $this->touch();
        }
    }
    
    public function removeTag(string $tag): void
    {
        $index = array_search($tag, $this->tags, true);
        if ($index !== false) {
            array_splice($this->tags, $index, 1);
            $this->touch();
        }
    }
    
    public function addAttachment(string $filename, string $path): void
    {
        $this->attachments[] = [
            'filename' => $filename,
            'path' => $path,
            'uploadedAt' => new DateTimeImmutable()
        ];
        $this->touch();
    }
    
    public function isOverdue(): bool
    {
        return $this->dueDate && 
               $this->dueDate < new DateTimeImmutable() && 
               $this->status !== TaskStatus::COMPLETED;
    }
    
    public function isCompleted(): bool
    {
        return $this->status === TaskStatus::COMPLETED;
    }
    
    public function getDaysUntilDue(): ?int
    {
        if (!$this->dueDate) {
            return null;
        }
        
        $now = new DateTimeImmutable();
        $diff = $now->diff($this->dueDate);
        
        return $diff->invert ? -$diff->days : $diff->days;
    }
}

// ============ Repository Pattern ============

interface RepositoryInterface
{
    public function save(object $entity): object;
    public function findById(int $id): ?object;
    public function findAll(): array;
    public function delete(int $id): bool;
    public function count(): int;
}

abstract class AbstractRepository implements RepositoryInterface
{
    protected PDO $pdo;
    protected string $table;
    protected string $entityClass;
    
    public function __construct(PDO $pdo, string $table, string $entityClass)
    {
        $this->pdo = $pdo;
        $this->table = $table;
        $this->entityClass = $entityClass;
    }
    
    public function count(): int
    {
        $stmt = $this->pdo->prepare("SELECT COUNT(*) FROM {$this->table}");
        $stmt->execute();
        return (int) $stmt->fetchColumn();
    }
    
    public function delete(int $id): bool
    {
        $stmt = $this->pdo->prepare("DELETE FROM {$this->table} WHERE id = :id");
        return $stmt->execute(['id' => $id]);
    }
    
    protected function executeQuery(string $sql, array $params = []): array
    {
        try {
            $stmt = $this->pdo->prepare($sql);
            $stmt->execute($params);
            return $stmt->fetchAll(PDO::FETCH_ASSOC);
        } catch (PDOException $e) {
            throw new DatabaseException($e->getMessage(), $e->getCode(), $e);
        }
    }
    
    protected function executeStatement(string $sql, array $params = []): bool
    {
        try {
            $stmt = $this->pdo->prepare($sql);
            return $stmt->execute($params);
        } catch (PDOException $e) {
            throw new DatabaseException($e->getMessage(), $e->getCode(), $e);
        }
    }
    
    abstract protected function hydrate(array $data): object;
    abstract protected function extractData(object $entity): array;
}

class UserRepository extends AbstractRepository
{
    public function __construct(PDO $pdo)
    {
        parent::__construct($pdo, 'users', User::class);
    }
    
    public function save(object $user): User
    {
        if (!$user instanceof User) {
            throw new InvalidArgumentException('Expected User entity');
        }
        
        $data = $this->extractData($user);
        
        if ($user->getId() === null) {
            return $this->insert($user, $data);
        } else {
            return $this->update($user, $data);
        }
    }
    
    private function insert(User $user, array $data): User
    {
        $sql = "INSERT INTO {$this->table} (name, email, age, role, is_active, last_login_at, metadata, created_at, updated_at) 
                VALUES (:name, :email, :age, :role, :is_active, :last_login_at, :metadata, :created_at, :updated_at)";
        
        $this->executeStatement($sql, $data);
        
        $id = new UserId((int) $this->pdo->lastInsertId());
        $user->setId($id);
        
        return $user;
    }
    
    private function update(User $user, array $data): User
    {
        $sql = "UPDATE {$this->table} 
                SET name = :name, email = :email, age = :age, role = :role, 
                    is_active = :is_active, last_login_at = :last_login_at, 
                    metadata = :metadata, updated_at = :updated_at 
                WHERE id = :id";
        
        $data['id'] = $user->getId()->getValue();
        $this->executeStatement($sql, $data);
        
        return $user;
    }
    
    public function findById(int $id): ?User
    {
        $sql = "SELECT * FROM {$this->table} WHERE id = :id";
        $results = $this->executeQuery($sql, ['id' => $id]);
        
        return !empty($results) ? $this->hydrate($results[0]) : null;
    }
    
    public function findAll(): array
    {
        $sql = "SELECT * FROM {$this->table} ORDER BY created_at DESC";
        $results = $this->executeQuery($sql);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function findByEmail(Email $email): ?User
    {
        $sql = "SELECT * FROM {$this->table} WHERE email = :email";
        $results = $this->executeQuery($sql, ['email' => $email->getValue()]);
        
        return !empty($results) ? $this->hydrate($results[0]) : null;
    }
    
    public function findByRole(UserRole $role): array
    {
        $sql = "SELECT * FROM {$this->table} WHERE role = :role ORDER BY name";
        $results = $this->executeQuery($sql, ['role' => $role->value]);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function findActive(): array
    {
        $sql = "SELECT * FROM {$this->table} WHERE is_active = 1 ORDER BY name";
        $results = $this->executeQuery($sql);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function search(string $query): array
    {
        $sql = "SELECT * FROM {$this->table} 
                WHERE name LIKE :query OR email LIKE :query 
                ORDER BY name";
        $results = $this->executeQuery($sql, ['query' => "%$query%"]);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    protected function hydrate(array $data): User
    {
        $user = new User(
            $data['name'],
            new Email($data['email']),
            (int) $data['age'],
            UserRole::from($data['role'])
        );
        
        if ($data['id']) {
            $user->setId(new UserId((int) $data['id']));
        }
        
        if (!$data['is_active']) {
            $user->deactivate();
        }
        
        if ($data['metadata']) {
            $metadata = json_decode($data['metadata'], true, 512, JSON_THROW_ON_ERROR);
            foreach ($metadata as $key => $value) {
                $user->setMetadata($key, $value);
            }
        }
        
        return $user;
    }
    
    protected function extractData(object $user): array
    {
        if (!$user instanceof User) {
            throw new InvalidArgumentException('Expected User entity');
        }
        
        return [
            'name' => $user->getName(),
            'email' => $user->getEmail()->getValue(),
            'age' => $user->getAge(),
            'role' => $user->getRole()->value,
            'is_active' => $user->isActive() ? 1 : 0,
            'last_login_at' => $user->getLastLoginAt()?->format('Y-m-d H:i:s'),
            'metadata' => json_encode($user->getMetadata(), JSON_THROW_ON_ERROR),
            'created_at' => $user->getCreatedAt()->format('Y-m-d H:i:s'),
            'updated_at' => $user->getUpdatedAt()->format('Y-m-d H:i:s')
        ];
    }
}

class TaskRepository extends AbstractRepository
{
    public function __construct(PDO $pdo)
    {
        parent::__construct($pdo, 'tasks', Task::class);
    }
    
    public function save(object $task): Task
    {
        if (!$task instanceof Task) {
            throw new InvalidArgumentException('Expected Task entity');
        }
        
        $data = $this->extractData($task);
        
        if ($task->getId() === null) {
            return $this->insert($task, $data);
        } else {
            return $this->update($task, $data);
        }
    }
    
    private function insert(Task $task, array $data): Task
    {
        $sql = "INSERT INTO {$this->table} 
                (title, description, priority, status, assigned_to, created_by, due_date, 
                 completed_at, tags, attachments, created_at, updated_at) 
                VALUES (:title, :description, :priority, :status, :assigned_to, :created_by, 
                        :due_date, :completed_at, :tags, :attachments, :created_at, :updated_at)";
        
        $this->executeStatement($sql, $data);
        
        $task->setId((int) $this->pdo->lastInsertId());
        
        return $task;
    }
    
    private function update(Task $task, array $data): Task
    {
        $sql = "UPDATE {$this->table} 
                SET title = :title, description = :description, priority = :priority, 
                    status = :status, due_date = :due_date, completed_at = :completed_at, 
                    tags = :tags, attachments = :attachments, updated_at = :updated_at 
                WHERE id = :id";
        
        $data['id'] = $task->getId();
        $this->executeStatement($sql, $data);
        
        return $task;
    }
    
    public function findById(int $id): ?Task
    {
        $sql = "SELECT * FROM {$this->table} WHERE id = :id";
        $results = $this->executeQuery($sql, ['id' => $id]);
        
        return !empty($results) ? $this->hydrate($results[0]) : null;
    }
    
    public function findAll(): array
    {
        $sql = "SELECT * FROM {$this->table} ORDER BY priority DESC, created_at DESC";
        $results = $this->executeQuery($sql);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function findByStatus(TaskStatus $status): array
    {
        $sql = "SELECT * FROM {$this->table} WHERE status = :status ORDER BY priority DESC";
        $results = $this->executeQuery($sql, ['status' => $status->value]);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function findByAssignee(UserId $userId): array
    {
        $sql = "SELECT * FROM {$this->table} WHERE assigned_to = :user_id ORDER BY priority DESC";
        $results = $this->executeQuery($sql, ['user_id' => $userId->getValue()]);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function findOverdue(): array
    {
        $sql = "SELECT * FROM {$this->table} 
                WHERE due_date < NOW() AND status != 'completed' 
                ORDER BY due_date ASC";
        $results = $this->executeQuery($sql);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function findByPriority(Priority $priority): array
    {
        $sql = "SELECT * FROM {$this->table} WHERE priority = :priority ORDER BY created_at DESC";
        $results = $this->executeQuery($sql, ['priority' => $priority->value]);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function getStatistics(): array
    {
        $sql = "SELECT 
                    status,
                    priority,
                    COUNT(*) as count,
                    AVG(CASE WHEN completed_at IS NOT NULL THEN 
                        TIMESTAMPDIFF(HOUR, created_at, completed_at) 
                        ELSE NULL END) as avg_completion_hours
                FROM {$this->table} 
                GROUP BY status, priority";
        
        return $this->executeQuery($sql);
    }
    
    protected function hydrate(array $data): Task
    {
        $task = new Task(
            $data['title'],
            $data['description'],
            Priority::from((int) $data['priority']),
            new UserId((int) $data['assigned_to']),
            new UserId((int) $data['created_by'])
        );
        
        if ($data['id']) {
            $task->setId((int) $data['id']);
        }
        
        $task->updateStatus(TaskStatus::from($data['status']));
        
        if ($data['due_date']) {
            $task->setDueDate(new DateTimeImmutable($data['due_date']));
        }
        
        if ($data['tags']) {
            $tags = json_decode($data['tags'], true, 512, JSON_THROW_ON_ERROR);
            foreach ($tags as $tag) {
                $task->addTag($tag);
            }
        }
        
        return $task;
    }
    
    protected function extractData(object $task): array
    {
        if (!$task instanceof Task) {
            throw new InvalidArgumentException('Expected Task entity');
        }
        
        return [
            'title' => $task->getTitle(),
            'description' => $task->getDescription(),
            'priority' => $task->getPriority()->value,
            'status' => $task->getStatus()->value,
            'assigned_to' => $task->getAssignedTo()->getValue(),
            'created_by' => $task->getCreatedBy()->getValue(),
            'due_date' => $task->getDueDate()?->format('Y-m-d H:i:s'),
            'completed_at' => $task->getCompletedAt()?->format('Y-m-d H:i:s'),
            'tags' => json_encode($task->getTags(), JSON_THROW_ON_ERROR),
            'attachments' => json_encode($task->getAttachments(), JSON_THROW_ON_ERROR),
            'created_at' => $task->getCreatedAt()->format('Y-m-d H:i:s'),
            'updated_at' => $task->getUpdatedAt()->format('Y-m-d H:i:s')
        ];
    }
}

// ============ Service Layer ============

class UserService
{
    public function __construct(
        private UserRepository $userRepository,
        private PasswordHasher $passwordHasher,
        private EventDispatcher $eventDispatcher
    ) {}
    
    public function createUser(string $name, string $email, int $age, UserRole $role = UserRole::USER): User
    {
        $emailVO = new Email($email);
        
        // Check if user already exists
        if ($this->userRepository->findByEmail($emailVO)) {
            throw new ValidationException(['email' => 'Email already exists']);
        }
        
        $user = new User($name, $emailVO, $age, $role);
        $savedUser = $this->userRepository->save($user);
        
        $this->eventDispatcher->dispatch('user.created', ['user' => $savedUser]);
        
        return $savedUser;
    }
    
    public function updateUser(int $userId, string $name, string $email, int $age): User
    {
        $user = $this->getUserById($userId);
        $emailVO = new Email($email);
        
        // Check if email is taken by another user
        $existingUser = $this->userRepository->findByEmail($emailVO);
        if ($existingUser && !$existingUser->getId()->equals($user->getId())) {
            throw new ValidationException(['email' => 'Email already exists']);
        }
        
        $user->updateProfile($name, $emailVO, $age);
        $updatedUser = $this->userRepository->save($user);
        
        $this->eventDispatcher->dispatch('user.updated', ['user' => $updatedUser]);
        
        return $updatedUser;
    }
    
    public function deleteUser(int $userId): void
    {
        $user = $this->getUserById($userId);
        
        if (!$this->userRepository->delete($userId)) {
            throw new DatabaseException('Failed to delete user');
        }
        
        $this->eventDispatcher->dispatch('user.deleted', ['user' => $user]);
    }
    
    public function getUserById(int $userId): User
    {
        $user = $this->userRepository->findById($userId);
        if (!$user) {
            throw new EntityNotFoundException('User', $userId);
        }
        
        return $user;
    }
    
    public function authenticateUser(string $email, string $password): User
    {
        $emailVO = new Email($email);
        $user = $this->userRepository->findByEmail($emailVO);
        
        if (!$user || !$this->passwordHasher->verify($password, $user->getMetadataValue('password_hash'))) {
            throw new AuthenticationException('Invalid credentials');
        }
        
        if (!$user->isActive()) {
            throw new AuthenticationException('Account is deactivated');
        }
        
        $user->recordLogin();
        $this->userRepository->save($user);
        
        $this->eventDispatcher->dispatch('user.login', ['user' => $user]);
        
        return $user;
    }
    
    public function changeUserRole(int $userId, UserRole $newRole, User $currentUser): User
    {
        if (!$currentUser->hasPermission('manage_users')) {
            throw new AuthorizationException('Insufficient permissions to change user roles');
        }
        
        $user = $this->getUserById($userId);
        $user->changeRole($newRole);
        
        $updatedUser = $this->userRepository->save($user);
        
        $this->eventDispatcher->dispatch('user.role_changed', [
            'user' => $updatedUser,
            'changed_by' => $currentUser
        ]);
        
        return $updatedUser;
    }
    
    public function searchUsers(string $query): array
    {
        return $this->userRepository->search($query);
    }
    
    public function getActiveUsers(): array
    {
        return $this->userRepository->findActive();
    }
    
    public function getUsersByRole(UserRole $role): array
    {
        return $this->userRepository->findByRole($role);
    }
}

class TaskService
{
    public function __construct(
        private TaskRepository $taskRepository,
        private UserRepository $userRepository,
        private EventDispatcher $eventDispatcher,
        private NotificationService $notificationService
    ) {}
    
    public function createTask(
        string $title,
        string $description,
        Priority $priority,
        int $assignedToId,
        int $createdById,
        ?DateTimeImmutable $dueDate = null
    ): Task {
        // Validate users exist
        $assignedTo = new UserId($assignedToId);
        $createdBy = new UserId($createdById);
        
        $this->userRepository->findById($assignedToId) ?? throw new EntityNotFoundException('User', $assignedToId);
        $this->userRepository->findById($createdById) ?? throw new EntityNotFoundException('User', $createdById);
        
        $task = new Task($title, $description, $priority, $assignedTo, $createdBy);
        
        if ($dueDate) {
            $task->setDueDate($dueDate);
        }
        
        $savedTask = $this->taskRepository->save($task);
        
        $this->eventDispatcher->dispatch('task.created', ['task' => $savedTask]);
        $this->notificationService->notifyTaskAssigned($savedTask);
        
        return $savedTask;
    }
    
    public function updateTask(
        int $taskId,
        string $title,
        string $description,
        Priority $priority,
        User $currentUser
    ): Task {
        $task = $this->getTaskById($taskId);
        
        if (!$this->canModifyTask($task, $currentUser)) {
            throw new AuthorizationException('You do not have permission to modify this task');
        }
        
        $task->updateDetails($title, $description, $priority);
        $updatedTask = $this->taskRepository->save($task);
        
        $this->eventDispatcher->dispatch('task.updated', ['task' => $updatedTask]);
        
        return $updatedTask;
    }
    
    public function updateTaskStatus(int $taskId, TaskStatus $status, User $currentUser): Task
    {
        $task = $this->getTaskById($taskId);
        
        if (!$this->canModifyTask($task, $currentUser)) {
            throw new AuthorizationException('You do not have permission to modify this task');
        }
        
        $oldStatus = $task->getStatus();
        $task->updateStatus($status);
        $updatedTask = $this->taskRepository->save($task);
        
        $this->eventDispatcher->dispatch('task.status_changed', [
            'task' => $updatedTask,
            'old_status' => $oldStatus,
            'new_status' => $status
        ]);
        
        if ($status === TaskStatus::COMPLETED) {
            $this->notificationService->notifyTaskCompleted($updatedTask);
        }
        
        return $updatedTask;
    }
    
    public function deleteTask(int $taskId, User $currentUser): void
    {
        $task = $this->getTaskById($taskId);
        
        if (!$this->canDeleteTask($task, $currentUser)) {
            throw new AuthorizationException('You do not have permission to delete this task');
        }
        
        if (!$this->taskRepository->delete($taskId)) {
            throw new DatabaseException('Failed to delete task');
        }
        
        $this->eventDispatcher->dispatch('task.deleted', ['task' => $task]);
    }
    
    public function getTaskById(int $taskId): Task
    {
        $task = $this->taskRepository->findById($taskId);
        if (!$task) {
            throw new EntityNotFoundException('Task', $taskId);
        }
        
        return $task;
    }
    
    public function getTasksByUser(int $userId): array
    {
        $userId = new UserId($userId);
        return $this->taskRepository->findByAssignee($userId);
    }
    
    public function getTasksByStatus(TaskStatus $status): array
    {
        return $this->taskRepository->findByStatus($status);
    }
    
    public function getOverdueTasks(): array
    {
        return $this->taskRepository->findOverdue();
    }
    
    public function getHighPriorityTasks(): array
    {
        return array_merge(
            $this->taskRepository->findByPriority(Priority::URGENT),
            $this->taskRepository->findByPriority(Priority::HIGH)
        );
    }
    
    public function getTaskStatistics(): array
    {
        $rawStats = $this->taskRepository->getStatistics();
        $stats = [
            'total' => 0,
            'by_status' => [],
            'by_priority' => [],
            'avg_completion_time' => []
        ];
        
        foreach ($rawStats as $stat) {
            $status = $stat['status'];
            $priority = Priority::from((int) $stat['priority'])->getLabel();
            $count = (int) $stat['count'];
            
            $stats['total'] += $count;
            $stats['by_status'][$status] = ($stats['by_status'][$status] ?? 0) + $count;
            $stats['by_priority'][$priority] = ($stats['by_priority'][$priority] ?? 0) + $count;
            
            if ($stat['avg_completion_hours']) {
                $stats['avg_completion_time'][$priority] = round((float) $stat['avg_completion_hours'], 2);
            }
        }
        
        return $stats;
    }
    
    private function canModifyTask(Task $task, User $user): bool
    {
        return $user->getId()->equals($task->getAssignedTo()) ||
               $user->getId()->equals($task->getCreatedBy()) ||
               $user->hasPermission('manage_tasks');
    }
    
    private function canDeleteTask(Task $task, User $user): bool
    {
        return $user->getId()->equals($task->getCreatedBy()) ||
               $user->hasPermission('delete_tasks');
    }
}

// ============ Supporting Services ============

class PasswordHasher
{
    public function hash(string $password): string
    {
        return password_hash($password, PASSWORD_ARGON2ID);
    }
    
    public function verify(string $password, ?string $hash): bool
    {
        return $hash && password_verify($password, $hash);
    }
}

class EventDispatcher
{
    private array $listeners = [];
    
    public function addListener(string $eventName, callable $listener): void
    {
        $this->listeners[$eventName][] = $listener;
    }
    
    public function dispatch(string $eventName, array $data = []): void
    {
        if (isset($this->listeners[$eventName])) {
            foreach ($this->listeners[$eventName] as $listener) {
                $listener($data);
            }
        }
    }
}

class NotificationService
{
    public function __construct(
        private UserRepository $userRepository,
        private EmailService $emailService,
        private LoggerInterface $logger
    ) {}
    
    public function notifyTaskAssigned(Task $task): void
    {
        try {
            $assignee = $this->userRepository->findById($task->getAssignedTo()->getValue());
            if ($assignee) {
                $this->emailService->send(
                    $assignee->getEmail(),
                    'New Task Assigned',
                    "You have been assigned a new task: {$task->getTitle()}"
                );
            }
        } catch (Exception $e) {
            $this->logger->error('Failed to send task assignment notification', [
                'task_id' => $task->getId(),
                'error' => $e->getMessage()
            ]);
        }
    }
    
    public function notifyTaskCompleted(Task $task): void
    {
        try {
            $creator = $this->userRepository->findById($task->getCreatedBy()->getValue());
            if ($creator) {
                $this->emailService->send(
                    $creator->getEmail(),
                    'Task Completed',
                    "Task '{$task->getTitle()}' has been completed."
                );
            }
        } catch (Exception $e) {
            $this->logger->error('Failed to send task completion notification', [
                'task_id' => $task->getId(),
                'error' => $e->getMessage()
            ]);
        }
    }
}

interface EmailService
{
    public function send(Email $to, string $subject, string $body): bool;
}

interface LoggerInterface
{
    public function error(string $message, array $context = []): void;
    public function info(string $message, array $context = []): void;
    public function warning(string $message, array $context = []): void;
}

class SimpleLogger implements LoggerInterface
{
    public function error(string $message, array $context = []): void
    {
        $this->log('ERROR', $message, $context);
    }
    
    public function info(string $message, array $context = []): void
    {
        $this->log('INFO', $message, $context);
    }
    
    public function warning(string $message, array $context = []): void
    {
        $this->log('WARNING', $message, $context);
    }
    
    private function log(string $level, string $message, array $context): void
    {
        $timestamp = date('Y-m-d H:i:s');
        $contextJson = !empty($context) ? ' ' . json_encode($context) : '';
        echo "[$timestamp] $level: $message$contextJson\n";
    }
}

class MockEmailService implements EmailService
{
    public function send(Email $to, string $subject, string $body): bool
    {
        echo "Email sent to {$to->getValue()}: $subject - $body\n";
        return true;
    }
}

// ============ Web Controllers (Simplified) ============

abstract class BaseController
{
    protected function jsonResponse(array $data, int $status = 200): void
    {
        http_response_code($status);
        header('Content-Type: application/json');
        echo json_encode($data, JSON_THROW_ON_ERROR);
    }
    
    protected function errorResponse(string $message, int $status = 400): void
    {
        $this->jsonResponse(['error' => $message], $status);
    }
    
    protected function getJsonInput(): array
    {
        $input = file_get_contents('php://input');
        return json_decode($input, true, 512, JSON_THROW_ON_ERROR) ?? [];
    }
}

class UserController extends BaseController
{
    public function __construct(private UserService $userService) {}
    
    public function index(): void
    {
        try {
            $users = $this->userService->getActiveUsers();
            $this->jsonResponse(['users' => array_map(fn($u) => $u->toArray(), $users)]);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function show(int $id): void
    {
        try {
            $user = $this->userService->getUserById($id);
            $this->jsonResponse(['user' => $user->toArray()]);
        } catch (EntityNotFoundException $e) {
            $this->errorResponse($e->getMessage(), 404);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function store(): void
    {
        try {
            $data = $this->getJsonInput();
            $user = $this->userService->createUser(
                $data['name'] ?? '',
                $data['email'] ?? '',
                (int) ($data['age'] ?? 0),
                isset($data['role']) ? UserRole::from($data['role']) : UserRole::USER
            );
            
            $this->jsonResponse(['user' => $user->toArray()], 201);
        } catch (ValidationException $e) {
            $this->jsonResponse(['errors' => $e->getErrors()], 422);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function update(int $id): void
    {
        try {
            $data = $this->getJsonInput();
            $user = $this->userService->updateUser(
                $id,
                $data['name'] ?? '',
                $data['email'] ?? '',
                (int) ($data['age'] ?? 0)
            );
            
            $this->jsonResponse(['user' => $user->toArray()]);
        } catch (EntityNotFoundException $e) {
            $this->errorResponse($e->getMessage(), 404);
        } catch (ValidationException $e) {
            $this->jsonResponse(['errors' => $e->getErrors()], 422);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function delete(int $id): void
    {
        try {
            $this->userService->deleteUser($id);
            $this->jsonResponse(['message' => 'User deleted successfully']);
        } catch (EntityNotFoundException $e) {
            $this->errorResponse($e->getMessage(), 404);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
}

class TaskController extends BaseController
{
    public function __construct(
        private TaskService $taskService,
        private AuthenticationService $authService
    ) {}
    
    public function index(): void
    {
        try {
            $currentUser = $this->authService->getCurrentUser();
            $tasks = $this->taskService->getTasksByUser($currentUser->getId()->getValue());
            $this->jsonResponse(['tasks' => array_map(fn($t) => $t->toArray(), $tasks)]);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function store(): void
    {
        try {
            $currentUser = $this->authService->getCurrentUser();
            $data = $this->getJsonInput();
            
            $dueDate = isset($data['due_date']) ? new DateTimeImmutable($data['due_date']) : null;
            
            $task = $this->taskService->createTask(
                $data['title'] ?? '',
                $data['description'] ?? '',
                Priority::from((int) ($data['priority'] ?? Priority::MEDIUM->value)),
                (int) ($data['assigned_to'] ?? $currentUser->getId()->getValue()),
                $currentUser->getId()->getValue(),
                $dueDate
            );
            
            $this->jsonResponse(['task' => $task->toArray()], 201);
        } catch (ValidationException $e) {
            $this->jsonResponse(['errors' => $e->getErrors()], 422);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function updateStatus(int $id): void
    {
        try {
            $currentUser = $this->authService->getCurrentUser();
            $data = $this->getJsonInput();
            
            $task = $this->taskService->updateTaskStatus(
                $id,
                TaskStatus::from($data['status'] ?? ''),
                $currentUser
            );
            
            $this->jsonResponse(['task' => $task->toArray()]);
        } catch (EntityNotFoundException $e) {
            $this->errorResponse($e->getMessage(), 404);
        } catch (AuthorizationException $e) {
            $this->errorResponse($e->getMessage(), 403);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
    
    public function statistics(): void
    {
        try {
            $stats = $this->taskService->getTaskStatistics();
            $this->jsonResponse(['statistics' => $stats]);
        } catch (Exception $e) {
            $this->errorResponse($e->getMessage(), 500);
        }
    }
}

// Simplified authentication service
class AuthenticationService
{
    private ?User $currentUser = null;
    
    public function setCurrentUser(User $user): void
    {
        $this->currentUser = $user;
    }
    
    public function getCurrentUser(): User
    {
        if (!$this->currentUser) {
            throw new AuthenticationException('User not authenticated');
        }
        
        return $this->currentUser;
    }
}

// ============ Database Schema and Setup ============

class DatabaseSetup
{
    public static function createTables(PDO $pdo): void
    {
        $queries = [
            "CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                age INT NOT NULL,
                role ENUM('admin', 'moderator', 'user', 'guest') NOT NULL DEFAULT 'user',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                last_login_at DATETIME NULL,
                metadata JSON NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_email (email),
                INDEX idx_role (role),
                INDEX idx_is_active (is_active)
            )",
            
            "CREATE TABLE IF NOT EXISTS tasks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                description TEXT NOT NULL,
                priority TINYINT NOT NULL,
                status ENUM('pending', 'in_progress', 'review', 'completed', 'cancelled') NOT NULL DEFAULT 'pending',
                assigned_to INT NOT NULL,
                created_by INT NOT NULL,
                due_date DATETIME NULL,
                completed_at DATETIME NULL,
                tags JSON NULL,
                attachments JSON NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                FOREIGN KEY (assigned_to) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_status (status),
                INDEX idx_priority (priority),
                INDEX idx_assigned_to (assigned_to),
                INDEX idx_due_date (due_date)
            )"
        ];
        
        foreach ($queries as $query) {
            $pdo->exec($query);
        }
    }
}

// ============ Demo and Testing ============

function runDemo(): void
{
    echo "=== PHP Comprehensive Examples Demo ===\n\n";
    
    try {
        // Setup database
        $pdo = new PDO('sqlite::memory:');
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        DatabaseSetup::createTables($pdo);
        
        // Initialize services
        $userRepo = new UserRepository($pdo);
        $taskRepo = new TaskRepository($pdo);
        $passwordHasher = new PasswordHasher();
        $eventDispatcher = new EventDispatcher();
        $logger = new SimpleLogger();
        $emailService = new MockEmailService();
        $notificationService = new NotificationService($userRepo, $emailService, $logger);
        
        $userService = new UserService($userRepo, $passwordHasher, $eventDispatcher);
        $taskService = new TaskService($taskRepo, $userRepo, $eventDispatcher, $notificationService);
        
        // Add event listeners
        $eventDispatcher->addListener('user.created', function($data) use ($logger) {
            $logger->info('User created', ['user_id' => $data['user']->getId()->getValue()]);
        });
        
        $eventDispatcher->addListener('task.completed', function($data) use ($logger) {
            $logger->info('Task completed', ['task_id' => $data['task']->getId()]);
        });
        
        // Create users
        echo "Creating users...\n";
        $admin = $userService->createUser('Admin User', 'admin@example.com', 30, UserRole::ADMIN);
        $user1 = $userService->createUser('Alice Johnson', 'alice@example.com', 28, UserRole::USER);
        $user2 = $userService->createUser('Bob Smith', 'bob@example.com', 35, UserRole::USER);
        
        echo "Created " . count($userService->getActiveUsers()) . " users\n\n";
        
        // Create tasks
        echo "Creating tasks...\n";
        $task1 = $taskService->createTask(
            'Implement user authentication',
            'Add secure login system with JWT tokens',
            Priority::HIGH,
            $user1->getId()->getValue(),
            $admin->getId()->getValue(),
            new DateTimeImmutable('+1 week')
        );
        
        $task2 = $taskService->createTask(
            'Design database schema',
            'Create comprehensive database design for the application',
            Priority::MEDIUM,
            $user2->getId()->getValue(),
            $admin->getId()->getValue()
        );
        
        $task3 = $taskService->createTask(
            'Write API documentation',
            'Document all REST API endpoints',
            Priority::LOW,
            $user1->getId()->getValue(),
            $admin->getId()->getValue()
        );
        
        echo "Created 3 tasks\n\n";
        
        // Update task status
        echo "Updating task statuses...\n";
        $taskService->updateTaskStatus($task1->getId(), TaskStatus::IN_PROGRESS, $user1);
        $taskService->updateTaskStatus($task2->getId(), TaskStatus::COMPLETED, $user2);
        
        // Get statistics
        echo "Task Statistics:\n";
        $stats = $taskService->getTaskStatistics();
        print_r($stats);
        
        // Test search
        echo "\nSearching users with 'alice':\n";
        $searchResults = $userService->searchUsers('alice');
        foreach ($searchResults as $user) {
            echo "- {$user->getName()} ({$user->getEmail()})\n";
        }
        
        // Test enums and value objects
        echo "\nTesting enums and value objects:\n";
        echo "Admin role permissions: " . implode(', ', UserRole::ADMIN->getPermissions()) . "\n";
        echo "Task priority HIGH label: " . Priority::HIGH->getLabel() . "\n";
        echo "Task status COMPLETED color: " . TaskStatus::COMPLETED->getColor() . "\n";
        
        // Test email value object
        $email = new Email('test@example.com');
        echo "Email domain: " . $email->getDomain() . "\n";
        echo "Email local part: " . $email->getLocalPart() . "\n";
        
        // Test JSON serialization
        echo "\nUser JSON:\n";
        echo $user1->toJson() . "\n";
        
        echo "\nTask JSON:\n";
        echo $task1->toJson() . "\n";
        
    } catch (Exception $e) {
        echo "Demo error: " . $e->getMessage() . "\n";
        echo "Stack trace:\n" . $e->getTraceAsString() . "\n";
    }
    
    echo "\n=== PHP Features Demonstrated ===\n";
    echo "- PHP 8.1+ enums with methods and backed enums\n";
    echo "- Readonly classes and properties\n";
    echo "- Union types and nullable types\n";
    echo "- Attributes and reflection\n";
    echo "- Traits for code reuse\n";
    echo "- Abstract classes and interfaces\n";
    echo "- Exception handling and custom exceptions\n";
    echo "- Value objects and domain modeling\n";
    echo "- Repository pattern with PDO\n";
    echo "- Service layer architecture\n";
    echo "- Event-driven programming\n";
    echo "- Dependency injection principles\n";
    echo "- JSON serialization and validation\n";
    echo "- Modern PHP best practices\n";
    echo "- Type declarations and strict typing\n";
    echo "- Match expressions (PHP 8.0+)\n";
    echo "- Constructor property promotion\n";
    echo "- Named arguments support\n";
}

// Run the demo if this file is executed directly
if (basename(__FILE__) === basename($_SERVER['SCRIPT_NAME'])) {
    runDemo();
}

?>