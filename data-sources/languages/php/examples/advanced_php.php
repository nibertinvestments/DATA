<?php
/**
 * Advanced PHP Programming Examples for AI Coding Agents
 * ======================================================
 * 
 * This module demonstrates advanced PHP features including:
 * - Modern PHP 8.3+ features and patterns
 * - Object-oriented programming with traits and interfaces
 * - Functional programming and closures
 * - Type system with union types and generics
 * - Exception handling and error management
 * - Performance optimization and memory management
 * - Database patterns with PDO and prepared statements
 * - HTTP client/server patterns and APIs
 * - Testing patterns and frameworks
 * - Security best practices
 * 
 * Author: AI Dataset Creation Team
 * License: MIT
 * Created: 2024
 */

declare(strict_types=1);

namespace AdvancedPhpExamples;

use DateTime;
use DateTimeInterface;
use Exception;
use InvalidArgumentException;
use RuntimeException;
use PDO;
use PDOException;
use PDOStatement;
use Closure;
use Generator;
use ReflectionClass;
use ReflectionException;
use ReflectionFunction;
use WeakMap;
use SplObjectStorage;

// =============================================================================
// Modern PHP 8.3+ Features and Type System
// =============================================================================

/**
 * Enum for order status using PHP 8.1+ backed enums
 */
enum OrderStatus: string
{
    case PENDING = 'pending';
    case PROCESSING = 'processing';
    case SHIPPED = 'shipped';
    case DELIVERED = 'delivered';
    case CANCELLED = 'cancelled';
    
    /**
     * Get the display name for the status
     */
    public function getDisplayName(): string
    {
        return match($this) {
            self::PENDING => 'Pending',
            self::PROCESSING => 'Processing',
            self::SHIPPED => 'Shipped',
            self::DELIVERED => 'Delivered',
            self::CANCELLED => 'Cancelled',
        };
    }
    
    /**
     * Check if the status can be transitioned to another status
     */
    public function canTransitionTo(self $newStatus): bool
    {
        return match([$this, $newStatus]) {
            [self::PENDING, self::PROCESSING] => true,
            [self::PENDING, self::CANCELLED] => true,
            [self::PROCESSING, self::SHIPPED] => true,
            [self::PROCESSING, self::CANCELLED] => true,
            [self::SHIPPED, self::DELIVERED] => true,
            default => false,
        };
    }
    
    /**
     * Get all valid next statuses
     */
    public function getValidNextStatuses(): array
    {
        return array_filter(
            self::cases(),
            fn(self $status) => $this->canTransitionTo($status)
        );
    }
}

/**
 * Advanced class using PHP 8+ features
 */
readonly class ImmutableValue
{
    public function __construct(
        public int|float $number,
        public string $text,
        public array $data = [],
        public ?DateTimeInterface $timestamp = null
    ) {
        // Constructor property promotion with default values
    }
    
    /**
     * Create a new instance with modified values
     */
    public function with(
        int|float $number = null,
        string $text = null,
        array $data = null,
        ?DateTimeInterface $timestamp = null
    ): self {
        return new self(
            $number ?? $this->number,
            $text ?? $this->text,
            $data ?? $this->data,
            $timestamp ?? $this->timestamp
        );
    }
    
    /**
     * Convert to array representation
     */
    public function toArray(): array
    {
        return [
            'number' => $this->number,
            'text' => $this->text,
            'data' => $this->data,
            'timestamp' => $this->timestamp?->format(DateTime::ATOM),
        ];
    }
}

/**
 * Generic-like container using template annotations
 * 
 * @template T
 */
class Container
{
    /** @var T[] */
    private array $items = [];
    
    /**
     * @param T $item
     */
    public function add(mixed $item): void
    {
        $this->items[] = $item;
    }
    
    /**
     * @return T|null
     */
    public function get(int $index): mixed
    {
        return $this->items[$index] ?? null;
    }
    
    /**
     * @param callable(T): bool $predicate
     * @return T[]
     */
    public function filter(callable $predicate): array
    {
        return array_filter($this->items, $predicate);
    }
    
    /**
     * @template U
     * @param callable(T): U $mapper
     * @return U[]
     */
    public function map(callable $mapper): array
    {
        return array_map($mapper, $this->items);
    }
    
    /**
     * @template U
     * @param U $initial
     * @param callable(U, T): U $reducer
     * @return U
     */
    public function reduce(mixed $initial, callable $reducer): mixed
    {
        return array_reduce($this->items, $reducer, $initial);
    }
    
    /**
     * @return T[]
     */
    public function toArray(): array
    {
        return $this->items;
    }
    
    public function count(): int
    {
        return count($this->items);
    }
    
    public function isEmpty(): bool
    {
        return empty($this->items);
    }
}

/**
 * Result type for error handling without exceptions
 * 
 * @template T
 * @template E
 */
abstract class Result
{
    /**
     * @template U
     * @param U $value
     * @return Ok<U>
     */
    public static function ok(mixed $value): Ok
    {
        return new Ok($value);
    }
    
    /**
     * @template F
     * @param F $error
     * @return Err<F>
     */
    public static function err(mixed $error): Err
    {
        return new Err($error);
    }
    
    abstract public function isOk(): bool;
    abstract public function isErr(): bool;
    
    /**
     * @return T
     * @throws RuntimeException
     */
    abstract public function unwrap(): mixed;
    
    /**
     * @param T $default
     * @return T
     */
    abstract public function unwrapOr(mixed $default): mixed;
    
    /**
     * @template U
     * @param callable(T): U $mapper
     * @return Result<U, E>
     */
    abstract public function map(callable $mapper): Result;
    
    /**
     * @template F
     * @param callable(E): F $mapper
     * @return Result<T, F>
     */
    abstract public function mapErr(callable $mapper): Result;
    
    /**
     * @template U
     * @param callable(T): Result<U, E> $mapper
     * @return Result<U, E>
     */
    abstract public function flatMap(callable $mapper): Result;
}

/**
 * @template T
 * @extends Result<T, mixed>
 */
final class Ok extends Result
{
    /**
     * @param T $value
     */
    public function __construct(private readonly mixed $value)
    {
    }
    
    public function isOk(): bool
    {
        return true;
    }
    
    public function isErr(): bool
    {
        return false;
    }
    
    /**
     * @return T
     */
    public function unwrap(): mixed
    {
        return $this->value;
    }
    
    /**
     * @param T $default
     * @return T
     */
    public function unwrapOr(mixed $default): mixed
    {
        return $this->value;
    }
    
    /**
     * @template U
     * @param callable(T): U $mapper
     * @return Ok<U>
     */
    public function map(callable $mapper): Result
    {
        return Result::ok($mapper($this->value));
    }
    
    /**
     * @template F
     * @param callable(mixed): F $mapper
     * @return Ok<T>
     */
    public function mapErr(callable $mapper): Result
    {
        return $this;
    }
    
    /**
     * @template U
     * @param callable(T): Result<U, mixed> $mapper
     * @return Result<U, mixed>
     */
    public function flatMap(callable $mapper): Result
    {
        return $mapper($this->value);
    }
}

/**
 * @template E
 * @extends Result<mixed, E>
 */
final class Err extends Result
{
    /**
     * @param E $error
     */
    public function __construct(private readonly mixed $error)
    {
    }
    
    public function isOk(): bool
    {
        return false;
    }
    
    public function isErr(): bool
    {
        return true;
    }
    
    /**
     * @throws RuntimeException
     */
    public function unwrap(): mixed
    {
        throw new RuntimeException('Called unwrap on Err: ' . $this->error);
    }
    
    public function unwrapOr(mixed $default): mixed
    {
        return $default;
    }
    
    /**
     * @template U
     * @param callable(mixed): U $mapper
     * @return Err<E>
     */
    public function map(callable $mapper): Result
    {
        return $this;
    }
    
    /**
     * @template F
     * @param callable(E): F $mapper
     * @return Err<F>
     */
    public function mapErr(callable $mapper): Result
    {
        return Result::err($mapper($this->error));
    }
    
    /**
     * @template U
     * @param callable(mixed): Result<U, E> $mapper
     * @return Err<E>
     */
    public function flatMap(callable $mapper): Result
    {
        return $this;
    }
}

// =============================================================================
// Advanced Object-Oriented Programming
// =============================================================================

/**
 * Trait for providing caching functionality
 */
trait CacheableTrait
{
    private array $cache = [];
    
    /**
     * Get value from cache or compute it
     */
    protected function cached(string $key, callable $computer): mixed
    {
        if (!array_key_exists($key, $this->cache)) {
            $this->cache[$key] = $computer();
        }
        
        return $this->cache[$key];
    }
    
    /**
     * Clear cache
     */
    protected function clearCache(): void
    {
        $this->cache = [];
    }
    
    /**
     * Remove specific key from cache
     */
    protected function forgetCached(string $key): void
    {
        unset($this->cache[$key]);
    }
}

/**
 * Trait for providing event handling functionality
 */
trait EventEmitterTrait
{
    private array $listeners = [];
    
    /**
     * Add event listener
     */
    public function on(string $event, callable $listener): void
    {
        $this->listeners[$event][] = $listener;
    }
    
    /**
     * Remove event listener
     */
    public function off(string $event, callable $listener): void
    {
        if (!isset($this->listeners[$event])) {
            return;
        }
        
        $this->listeners[$event] = array_filter(
            $this->listeners[$event],
            fn($l) => $l !== $listener
        );
    }
    
    /**
     * Emit event to all listeners
     */
    protected function emit(string $event, mixed ...$args): void
    {
        if (!isset($this->listeners[$event])) {
            return;
        }
        
        foreach ($this->listeners[$event] as $listener) {
            $listener(...$args);
        }
    }
}

/**
 * Interface for objects that can be serialized to array
 */
interface ArraySerializable
{
    public function toArray(): array;
    public static function fromArray(array $data): static;
}

/**
 * Interface for validatable objects
 */
interface Validatable
{
    /**
     * @return string[] Array of validation errors
     */
    public function validate(): array;
    
    public function isValid(): bool;
}

/**
 * Abstract base entity class
 */
abstract class BaseEntity implements ArraySerializable, Validatable
{
    use CacheableTrait, EventEmitterTrait;
    
    protected ?int $id = null;
    protected ?DateTimeInterface $createdAt = null;
    protected ?DateTimeInterface $updatedAt = null;
    
    public function __construct(?int $id = null)
    {
        $this->id = $id;
        $this->createdAt = new DateTime();
    }
    
    public function getId(): ?int
    {
        return $this->id;
    }
    
    public function setId(int $id): void
    {
        $this->id = $id;
    }
    
    public function getCreatedAt(): ?DateTimeInterface
    {
        return $this->createdAt;
    }
    
    public function getUpdatedAt(): ?DateTimeInterface
    {
        return $this->updatedAt;
    }
    
    public function markAsUpdated(): void
    {
        $this->updatedAt = new DateTime();
        $this->emit('updated', $this);
    }
    
    public function isValid(): bool
    {
        return empty($this->validate());
    }
    
    /**
     * Get validation errors with caching
     */
    public function getValidationErrors(): array
    {
        return $this->cached('validation_errors', fn() => $this->validate());
    }
    
    public function toArray(): array
    {
        return [
            'id' => $this->id,
            'created_at' => $this->createdAt?->format(DateTime::ATOM),
            'updated_at' => $this->updatedAt?->format(DateTime::ATOM),
        ];
    }
}

/**
 * Example entity demonstrating advanced OOP features
 */
class User extends BaseEntity
{
    private string $email;
    private string $firstName;
    private string $lastName;
    private array $roles = [];
    private bool $isActive = true;
    
    public function __construct(
        string $email,
        string $firstName,
        string $lastName,
        ?int $id = null
    ) {
        parent::__construct($id);
        $this->setEmail($email);
        $this->firstName = $firstName;
        $this->lastName = $lastName;
    }
    
    public function getEmail(): string
    {
        return $this->email;
    }
    
    public function setEmail(string $email): void
    {
        if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            throw new InvalidArgumentException('Invalid email format');
        }
        
        $oldEmail = $this->email ?? null;
        $this->email = $email;
        
        if ($oldEmail !== null && $oldEmail !== $email) {
            $this->markAsUpdated();
            $this->emit('email_changed', $oldEmail, $email);
        }
    }
    
    public function getFirstName(): string
    {
        return $this->firstName;
    }
    
    public function setFirstName(string $firstName): void
    {
        $this->firstName = $firstName;
        $this->markAsUpdated();
        $this->forgetCached('full_name');
    }
    
    public function getLastName(): string
    {
        return $this->lastName;
    }
    
    public function setLastName(string $lastName): void
    {
        $this->lastName = $lastName;
        $this->markAsUpdated();
        $this->forgetCached('full_name');
    }
    
    public function getFullName(): string
    {
        return $this->cached('full_name', fn() => "{$this->firstName} {$this->lastName}");
    }
    
    public function getRoles(): array
    {
        return $this->roles;
    }
    
    public function addRole(string $role): void
    {
        if (!in_array($role, $this->roles, true)) {
            $this->roles[] = $role;
            $this->markAsUpdated();
            $this->emit('role_added', $role);
        }
    }
    
    public function removeRole(string $role): void
    {
        $index = array_search($role, $this->roles, true);
        if ($index !== false) {
            unset($this->roles[$index]);
            $this->roles = array_values($this->roles);
            $this->markAsUpdated();
            $this->emit('role_removed', $role);
        }
    }
    
    public function hasRole(string $role): bool
    {
        return in_array($role, $this->roles, true);
    }
    
    public function isActive(): bool
    {
        return $this->isActive;
    }
    
    public function setActive(bool $isActive): void
    {
        if ($this->isActive !== $isActive) {
            $this->isActive = $isActive;
            $this->markAsUpdated();
            $this->emit($isActive ? 'activated' : 'deactivated');
        }
    }
    
    public function validate(): array
    {
        $errors = [];
        
        if (empty($this->email)) {
            $errors[] = 'Email is required';
        } elseif (!filter_var($this->email, FILTER_VALIDATE_EMAIL)) {
            $errors[] = 'Email format is invalid';
        }
        
        if (empty($this->firstName)) {
            $errors[] = 'First name is required';
        }
        
        if (empty($this->lastName)) {
            $errors[] = 'Last name is required';
        }
        
        return $errors;
    }
    
    public function toArray(): array
    {
        return array_merge(parent::toArray(), [
            'email' => $this->email,
            'first_name' => $this->firstName,
            'last_name' => $this->lastName,
            'full_name' => $this->getFullName(),
            'roles' => $this->roles,
            'is_active' => $this->isActive,
        ]);
    }
    
    public static function fromArray(array $data): static
    {
        $user = new self(
            $data['email'],
            $data['first_name'],
            $data['last_name'],
            $data['id'] ?? null
        );
        
        if (isset($data['roles'])) {
            foreach ($data['roles'] as $role) {
                $user->addRole($role);
            }
        }
        
        if (isset($data['is_active'])) {
            $user->setActive($data['is_active']);
        }
        
        return $user;
    }
}

// =============================================================================
// Design Patterns Implementation
// =============================================================================

/**
 * Repository pattern interface
 * 
 * @template T
 */
interface RepositoryInterface
{
    /**
     * @return T|null
     */
    public function find(int $id): ?object;
    
    /**
     * @return T[]
     */
    public function findAll(): array;
    
    /**
     * @param T $entity
     * @return T
     */
    public function save(object $entity): object;
    
    public function delete(int $id): bool;
}

/**
 * Generic repository implementation
 * 
 * @template T of BaseEntity
 * @implements RepositoryInterface<T>
 */
abstract class AbstractRepository implements RepositoryInterface
{
    protected PDO $pdo;
    protected string $tableName;
    protected string $entityClass;
    
    public function __construct(PDO $pdo, string $tableName, string $entityClass)
    {
        $this->pdo = $pdo;
        $this->tableName = $tableName;
        $this->entityClass = $entityClass;
    }
    
    public function find(int $id): ?object
    {
        $stmt = $this->pdo->prepare("SELECT * FROM {$this->tableName} WHERE id = :id");
        $stmt->bindValue(':id', $id, PDO::PARAM_INT);
        $stmt->execute();
        
        $data = $stmt->fetch(PDO::FETCH_ASSOC);
        
        if (!$data) {
            return null;
        }
        
        return $this->hydrate($data);
    }
    
    public function findAll(): array
    {
        $stmt = $this->pdo->query("SELECT * FROM {$this->tableName}");
        $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    public function save(object $entity): object
    {
        if (!$entity instanceof BaseEntity) {
            throw new InvalidArgumentException('Entity must extend BaseEntity');
        }
        
        if ($entity->getId() === null) {
            return $this->insert($entity);
        } else {
            return $this->update($entity);
        }
    }
    
    public function delete(int $id): bool
    {
        $stmt = $this->pdo->prepare("DELETE FROM {$this->tableName} WHERE id = :id");
        $stmt->bindValue(':id', $id, PDO::PARAM_INT);
        
        return $stmt->execute() && $stmt->rowCount() > 0;
    }
    
    /**
     * @param array<string, mixed> $criteria
     * @return T[]
     */
    public function findBy(array $criteria): array
    {
        $conditions = [];
        $params = [];
        
        foreach ($criteria as $field => $value) {
            $conditions[] = "{$field} = :{$field}";
            $params[":{$field}"] = $value;
        }
        
        $sql = "SELECT * FROM {$this->tableName}";
        if (!empty($conditions)) {
            $sql .= ' WHERE ' . implode(' AND ', $conditions);
        }
        
        $stmt = $this->pdo->prepare($sql);
        foreach ($params as $param => $value) {
            $stmt->bindValue($param, $value);
        }
        $stmt->execute();
        
        $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        return array_map([$this, 'hydrate'], $results);
    }
    
    /**
     * Convert array data to entity instance
     */
    abstract protected function hydrate(array $data): object;
    
    /**
     * Convert entity to array for database storage
     */
    abstract protected function extract(BaseEntity $entity): array;
    
    private function insert(BaseEntity $entity): BaseEntity
    {
        $data = $this->extract($entity);
        unset($data['id']); // Remove ID for insert
        
        $columns = array_keys($data);
        $placeholders = array_map(fn($col) => ":{$col}", $columns);
        
        $sql = "INSERT INTO {$this->tableName} (" . implode(', ', $columns) . 
               ") VALUES (" . implode(', ', $placeholders) . ")";
        
        $stmt = $this->pdo->prepare($sql);
        foreach ($data as $column => $value) {
            $stmt->bindValue(":{$column}", $value);
        }
        
        $stmt->execute();
        
        $entity->setId((int)$this->pdo->lastInsertId());
        
        return $entity;
    }
    
    private function update(BaseEntity $entity): BaseEntity
    {
        $data = $this->extract($entity);
        $id = $data['id'];
        unset($data['id']);
        
        $setClause = array_map(fn($col) => "{$col} = :{$col}", array_keys($data));
        
        $sql = "UPDATE {$this->tableName} SET " . implode(', ', $setClause) . " WHERE id = :id";
        
        $stmt = $this->pdo->prepare($sql);
        foreach ($data as $column => $value) {
            $stmt->bindValue(":{$column}", $value);
        }
        $stmt->bindValue(':id', $id, PDO::PARAM_INT);
        
        $stmt->execute();
        
        $entity->markAsUpdated();
        
        return $entity;
    }
}

/**
 * User repository implementation
 * 
 * @extends AbstractRepository<User>
 */
class UserRepository extends AbstractRepository
{
    public function __construct(PDO $pdo)
    {
        parent::__construct($pdo, 'users', User::class);
    }
    
    protected function hydrate(array $data): User
    {
        $user = new User(
            $data['email'],
            $data['first_name'],
            $data['last_name'],
            (int)$data['id']
        );
        
        if (!empty($data['roles'])) {
            $roles = json_decode($data['roles'], true) ?: [];
            foreach ($roles as $role) {
                $user->addRole($role);
            }
        }
        
        if (isset($data['is_active'])) {
            $user->setActive((bool)$data['is_active']);
        }
        
        return $user;
    }
    
    protected function extract(BaseEntity $entity): array
    {
        if (!$entity instanceof User) {
            throw new InvalidArgumentException('Entity must be a User instance');
        }
        
        return [
            'id' => $entity->getId(),
            'email' => $entity->getEmail(),
            'first_name' => $entity->getFirstName(),
            'last_name' => $entity->getLastName(),
            'roles' => json_encode($entity->getRoles()),
            'is_active' => $entity->isActive() ? 1 : 0,
            'created_at' => $entity->getCreatedAt()?->format('Y-m-d H:i:s'),
            'updated_at' => $entity->getUpdatedAt()?->format('Y-m-d H:i:s'),
        ];
    }
    
    public function findByEmail(string $email): ?User
    {
        $results = $this->findBy(['email' => $email]);
        return $results[0] ?? null;
    }
    
    public function findActiveUsers(): array
    {
        return $this->findBy(['is_active' => 1]);
    }
}

/**
 * Command pattern for undo/redo functionality
 */
interface CommandInterface
{
    public function execute(): mixed;
    public function undo(): mixed;
    public function canUndo(): bool;
}

/**
 * Command to update user email
 */
class UpdateUserEmailCommand implements CommandInterface
{
    private string $oldEmail;
    private bool $executed = false;
    
    public function __construct(
        private User $user,
        private string $newEmail
    ) {
        $this->oldEmail = $user->getEmail();
    }
    
    public function execute(): mixed
    {
        if ($this->executed) {
            throw new RuntimeException('Command already executed');
        }
        
        $this->user->setEmail($this->newEmail);
        $this->executed = true;
        
        return $this->user;
    }
    
    public function undo(): mixed
    {
        if (!$this->executed) {
            throw new RuntimeException('Cannot undo non-executed command');
        }
        
        $this->user->setEmail($this->oldEmail);
        $this->executed = false;
        
        return $this->user;
    }
    
    public function canUndo(): bool
    {
        return $this->executed;
    }
}

/**
 * Command manager for handling command history
 */
class CommandManager
{
    /** @var CommandInterface[] */
    private array $history = [];
    private int $currentIndex = -1;
    private int $maxHistory;
    
    public function __construct(int $maxHistory = 50)
    {
        $this->maxHistory = $maxHistory;
    }
    
    public function execute(CommandInterface $command): mixed
    {
        $result = $command->execute();
        
        // Remove any commands after current index
        $this->history = array_slice($this->history, 0, $this->currentIndex + 1);
        
        // Add new command
        $this->history[] = $command;
        $this->currentIndex++;
        
        // Limit history size
        if (count($this->history) > $this->maxHistory) {
            array_shift($this->history);
            $this->currentIndex--;
        }
        
        return $result;
    }
    
    public function undo(): mixed
    {
        if (!$this->canUndo()) {
            throw new RuntimeException('Nothing to undo');
        }
        
        $command = $this->history[$this->currentIndex];
        $result = $command->undo();
        $this->currentIndex--;
        
        return $result;
    }
    
    public function redo(): mixed
    {
        if (!$this->canRedo()) {
            throw new RuntimeException('Nothing to redo');
        }
        
        $this->currentIndex++;
        $command = $this->history[$this->currentIndex];
        
        return $command->execute();
    }
    
    public function canUndo(): bool
    {
        return $this->currentIndex >= 0 && 
               $this->history[$this->currentIndex]->canUndo();
    }
    
    public function canRedo(): bool
    {
        return $this->currentIndex < count($this->history) - 1;
    }
    
    public function clear(): void
    {
        $this->history = [];
        $this->currentIndex = -1;
    }
}

// =============================================================================
// Functional Programming and Closures
// =============================================================================

/**
 * Functional programming utilities
 */
class Functional
{
    /**
     * Curry a function to enable partial application
     */
    public static function curry(callable $function, int $arity = null): Closure
    {
        if ($arity === null) {
            $reflection = new ReflectionFunction($function);
            $arity = $reflection->getNumberOfRequiredParameters();
        }
        
        return function (...$args) use ($function, $arity): mixed {
            if (count($args) >= $arity) {
                return $function(...$args);
            }
            
            return static::curry(
                function (...$remainingArgs) use ($function, $args): mixed {
                    return $function(...$args, ...$remainingArgs);
                },
                $arity - count($args)
            );
        };
    }
    
    /**
     * Compose functions right to left
     */
    public static function compose(callable ...$functions): Closure
    {
        return array_reduce(
            $functions,
            function ($carry, $function) {
                return function ($value) use ($carry, $function) {
                    return $function($carry($value));
                };
            },
            function ($value) { return $value; }
        );
    }
    
    /**
     * Pipe functions left to right
     */
    public static function pipe(callable ...$functions): Closure
    {
        return static::compose(...array_reverse($functions));
    }
    
    /**
     * Memoize function results
     */
    public static function memoize(callable $function): Closure
    {
        $cache = [];
        
        return function (...$args) use ($function, &$cache): mixed {
            $key = serialize($args);
            
            if (!array_key_exists($key, $cache)) {
                $cache[$key] = $function(...$args);
            }
            
            return $cache[$key];
        };
    }
    
    /**
     * Debounce function calls
     */
    public static function debounce(callable $function, float $delay): Closure
    {
        $lastCall = 0;
        
        return function (...$args) use ($function, $delay, &$lastCall): mixed {
            $now = microtime(true);
            
            if ($now - $lastCall >= $delay) {
                $lastCall = $now;
                return $function(...$args);
            }
            
            return null;
        };
    }
    
    /**
     * Throttle function calls
     */
    public static function throttle(callable $function, float $interval): Closure
    {
        $lastCall = 0;
        $pending = false;
        
        return function (...$args) use ($function, $interval, &$lastCall, &$pending): mixed {
            $now = microtime(true);
            
            if ($now - $lastCall >= $interval) {
                $lastCall = $now;
                return $function(...$args);
            } elseif (!$pending) {
                $pending = true;
                // In a real implementation, you'd use a timer here
                // This is a simplified version
                return null;
            }
            
            return null;
        };
    }
    
    /**
     * Partial application of function arguments
     */
    public static function partial(callable $function, ...$partialArgs): Closure
    {
        return function (...$remainingArgs) use ($function, $partialArgs): mixed {
            return $function(...$partialArgs, ...$remainingArgs);
        };
    }
}

/**
 * Collection class with functional methods
 * 
 * @template T
 */
class Collection
{
    /** @var T[] */
    private array $items;
    
    /**
     * @param T[] $items
     */
    public function __construct(array $items = [])
    {
        $this->items = array_values($items);
    }
    
    /**
     * @param callable(T): bool $predicate
     * @return Collection<T>
     */
    public function filter(callable $predicate): self
    {
        return new self(array_filter($this->items, $predicate));
    }
    
    /**
     * @template U
     * @param callable(T): U $mapper
     * @return Collection<U>
     */
    public function map(callable $mapper): self
    {
        return new self(array_map($mapper, $this->items));
    }
    
    /**
     * @template U
     * @param U $initial
     * @param callable(U, T): U $reducer
     * @return U
     */
    public function reduce(mixed $initial, callable $reducer): mixed
    {
        return array_reduce($this->items, $reducer, $initial);
    }
    
    /**
     * @param callable(T): bool $predicate
     * @return T|null
     */
    public function find(callable $predicate): mixed
    {
        foreach ($this->items as $item) {
            if ($predicate($item)) {
                return $item;
            }
        }
        
        return null;
    }
    
    /**
     * @param callable(T): bool $predicate
     */
    public function every(callable $predicate): bool
    {
        foreach ($this->items as $item) {
            if (!$predicate($item)) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * @param callable(T): bool $predicate
     */
    public function some(callable $predicate): bool
    {
        foreach ($this->items as $item) {
            if ($predicate($item)) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @param int $size
     * @return Collection<Collection<T>>
     */
    public function chunk(int $size): self
    {
        if ($size <= 0) {
            throw new InvalidArgumentException('Chunk size must be positive');
        }
        
        $chunks = array_chunk($this->items, $size);
        
        return new self(array_map(fn($chunk) => new self($chunk), $chunks));
    }
    
    /**
     * @param callable(T): mixed $keyExtractor
     * @return array<mixed, Collection<T>>
     */
    public function groupBy(callable $keyExtractor): array
    {
        $groups = [];
        
        foreach ($this->items as $item) {
            $key = $keyExtractor($item);
            
            if (!isset($groups[$key])) {
                $groups[$key] = new self();
            }
            
            $groups[$key]->items[] = $item;
        }
        
        return $groups;
    }
    
    public function take(int $count): self
    {
        return new self(array_slice($this->items, 0, $count));
    }
    
    public function skip(int $count): self
    {
        return new self(array_slice($this->items, $count));
    }
    
    /**
     * @param callable(T, T): int $comparator
     */
    public function sort(callable $comparator = null): self
    {
        $sorted = $this->items;
        
        if ($comparator === null) {
            sort($sorted);
        } else {
            usort($sorted, $comparator);
        }
        
        return new self($sorted);
    }
    
    public function reverse(): self
    {
        return new self(array_reverse($this->items));
    }
    
    public function unique(): self
    {
        return new self(array_unique($this->items, SORT_REGULAR));
    }
    
    /**
     * @return T[]
     */
    public function toArray(): array
    {
        return $this->items;
    }
    
    public function count(): int
    {
        return count($this->items);
    }
    
    public function isEmpty(): bool
    {
        return empty($this->items);
    }
    
    /**
     * @return T|null
     */
    public function first(): mixed
    {
        return $this->items[0] ?? null;
    }
    
    /**
     * @return T|null
     */
    public function last(): mixed
    {
        return $this->items[count($this->items) - 1] ?? null;
    }
}

// =============================================================================
// Performance Optimization and Generators
// =============================================================================

/**
 * Performance monitoring and optimization utilities
 */
class Performance
{
    /**
     * Measure execution time and memory usage
     */
    public static function measure(callable $operation): array
    {
        $startTime = hrtime(true);
        $startMemory = memory_get_usage(true);
        
        $result = $operation();
        
        $endTime = hrtime(true);
        $endMemory = memory_get_usage(true);
        
        return [
            'result' => $result,
            'execution_time_ms' => ($endTime - $startTime) / 1_000_000,
            'memory_used_bytes' => $endMemory - $startMemory,
            'peak_memory_bytes' => memory_get_peak_usage(true),
        ];
    }
    
    /**
     * Benchmark function with multiple iterations
     */
    public static function benchmark(callable $operation, int $iterations = 1000): array
    {
        $times = [];
        $memoryUsages = [];
        
        // Warmup
        for ($i = 0; $i < min($iterations / 10, 100); $i++) {
            $operation();
        }
        
        // Clear memory
        gc_collect_cycles();
        
        for ($i = 0; $i < $iterations; $i++) {
            $measurement = self::measure($operation);
            $times[] = $measurement['execution_time_ms'];
            $memoryUsages[] = $measurement['memory_used_bytes'];
        }
        
        return [
            'iterations' => $iterations,
            'total_time_ms' => array_sum($times),
            'average_time_ms' => array_sum($times) / count($times),
            'min_time_ms' => min($times),
            'max_time_ms' => max($times),
            'average_memory_bytes' => array_sum($memoryUsages) / count($memoryUsages),
            'operations_per_second' => $iterations / (array_sum($times) / 1000),
        ];
    }
}

/**
 * Generator-based utilities for memory-efficient processing
 */
class Generators
{
    /**
     * Generate range of numbers
     */
    public static function range(int $start, int $end, int $step = 1): Generator
    {
        for ($i = $start; $i <= $end; $i += $step) {
            yield $i;
        }
    }
    
    /**
     * Generate fibonacci sequence
     */
    public static function fibonacci(int $count): Generator
    {
        $a = 0;
        $b = 1;
        
        for ($i = 0; $i < $count; $i++) {
            yield $a;
            [$a, $b] = [$b, $a + $b];
        }
    }
    
    /**
     * Batch process large arrays without loading everything into memory
     */
    public static function batchProcess(array $items, int $batchSize): Generator
    {
        $batch = [];
        
        foreach ($items as $item) {
            $batch[] = $item;
            
            if (count($batch) >= $batchSize) {
                yield $batch;
                $batch = [];
            }
        }
        
        if (!empty($batch)) {
            yield $batch;
        }
    }
    
    /**
     * Read file line by line without loading entire file
     */
    public static function readFileLines(string $filename): Generator
    {
        $handle = fopen($filename, 'r');
        
        if (!$handle) {
            throw new RuntimeException("Cannot open file: {$filename}");
        }
        
        try {
            while (($line = fgets($handle)) !== false) {
                yield rtrim($line, "\r\n");
            }
        } finally {
            fclose($handle);
        }
    }
    
    /**
     * Filter generator results
     */
    public static function filter(Generator $generator, callable $predicate): Generator
    {
        foreach ($generator as $key => $value) {
            if ($predicate($value)) {
                yield $key => $value;
            }
        }
    }
    
    /**
     * Map generator results
     */
    public static function map(Generator $generator, callable $mapper): Generator
    {
        foreach ($generator as $key => $value) {
            yield $key => $mapper($value);
        }
    }
    
    /**
     * Take first N items from generator
     */
    public static function take(Generator $generator, int $count): Generator
    {
        $taken = 0;
        
        foreach ($generator as $key => $value) {
            if ($taken >= $count) {
                break;
            }
            
            yield $key => $value;
            $taken++;
        }
    }
}

// =============================================================================
// Example Usage and Demonstrations
// =============================================================================

/**
 * Main demonstration class
 */
class AdvancedPhpDemo
{
    /**
     * Run comprehensive demonstration of all features
     */
    public static function run(): void
    {
        echo "🚀 Advanced PHP Programming Demonstration\n";
        echo str_repeat('=', 50) . "\n";
        
        self::demonstrateModernPhpFeatures();
        self::demonstrateObjectOrientedProgramming();
        self::demonstrateDesignPatterns();
        self::demonstrateFunctionalProgramming();
        self::demonstratePerformanceOptimization();
        
        echo "\n✅ All PHP demonstrations completed!\n";
    }
    
    private static function demonstrateModernPhpFeatures(): void
    {
        echo "\n📝 Modern PHP Features:\n";
        
        // Enum demonstration
        $status = OrderStatus::PENDING;
        echo "Order status: {$status->getDisplayName()}\n";
        echo "Can transition to PROCESSING: " . 
             ($status->canTransitionTo(OrderStatus::PROCESSING) ? 'Yes' : 'No') . "\n";
        
        $validNextStatuses = $status->getValidNextStatuses();
        echo "Valid next statuses: " . 
             implode(', ', array_map(fn($s) => $s->getDisplayName(), $validNextStatuses)) . "\n";
        
        // Readonly class demonstration
        $value = new ImmutableValue(42, "Hello", ['key' => 'value']);
        $newValue = $value->with(number: 100, text: "World");
        
        echo "Original value: {$value->number}, {$value->text}\n";
        echo "New value: {$newValue->number}, {$newValue->text}\n";
        
        // Container (generic-like) demonstration
        $container = new Container();
        $container->add("Hello");
        $container->add("World");
        $container->add("PHP");
        
        $filtered = $container->filter(fn($item) => strlen($item) > 3);
        $mapped = $container->map(fn($item) => strtoupper($item));
        $reduced = $container->reduce("", fn($acc, $item) => $acc . $item . " ");
        
        echo "Container items: " . implode(', ', $container->toArray()) . "\n";
        echo "Filtered items: " . implode(', ', $filtered) . "\n";
        echo "Mapped items: " . implode(', ', $mapped) . "\n";
        echo "Reduced result: {$reduced}\n";
        
        // Result type demonstration
        $successResult = Result::ok(42);
        $errorResult = Result::err("Something went wrong");
        
        echo "Success result: " . $successResult->unwrapOr(0) . "\n";
        echo "Error result: " . $errorResult->unwrapOr(0) . "\n";
        
        $mappedResult = $successResult->map(fn($x) => $x * 2);
        echo "Mapped success result: " . $mappedResult->unwrapOr(0) . "\n";
    }
    
    private static function demonstrateObjectOrientedProgramming(): void
    {
        echo "\n🏗️ Object-Oriented Programming:\n";
        
        // Create and manipulate user
        $user = new User("john.doe@example.com", "John", "Doe");
        
        // Add event listeners
        $user->on('email_changed', function ($oldEmail, $newEmail) {
            echo "Email changed from {$oldEmail} to {$newEmail}\n";
        });
        
        $user->on('role_added', function ($role) {
            echo "Role added: {$role}\n";
        });
        
        // Demonstrate functionality
        echo "User: {$user->getFullName()}\n";
        echo "Email: {$user->getEmail()}\n";
        
        $user->addRole('admin');
        $user->addRole('user');
        
        echo "Roles: " . implode(', ', $user->getRoles()) . "\n";
        echo "Is admin: " . ($user->hasRole('admin') ? 'Yes' : 'No') . "\n";
        echo "Is valid: " . ($user->isValid() ? 'Yes' : 'No') . "\n";
        
        // Test caching
        echo "Full name (cached): {$user->getFullName()}\n";
        
        // Change email to trigger event
        $user->setEmail("john.smith@example.com");
        
        // Convert to array
        $userArray = $user->toArray();
        echo "User as array: " . json_encode($userArray, JSON_PRETTY_PRINT) . "\n";
    }
    
    private static function demonstrateDesignPatterns(): void
    {
        echo "\n🔧 Design Patterns:\n";
        
        // Command pattern demonstration
        $user = new User("test@example.com", "Test", "User");
        $commandManager = new CommandManager();
        
        // Execute command
        $command = new UpdateUserEmailCommand($user, "newemail@example.com");
        $commandManager->execute($command);
        
        echo "After command execution - Email: {$user->getEmail()}\n";
        
        // Undo command
        if ($commandManager->canUndo()) {
            $commandManager->undo();
            echo "After undo - Email: {$user->getEmail()}\n";
        }
        
        // Redo command
        if ($commandManager->canRedo()) {
            $commandManager->redo();
            echo "After redo - Email: {$user->getEmail()}\n";
        }
    }
    
    private static function demonstrateFunctionalProgramming(): void
    {
        echo "\n🔧 Functional Programming:\n";
        
        // Function composition
        $add = fn($a, $b) => $a + $b;
        $multiply = fn($a, $b) => $a * $b;
        $double = Functional::partial($multiply, 2);
        
        echo "Double 5: " . $double(5) . "\n";
        
        // Currying
        $curriedAdd = Functional::curry($add);
        $addFive = $curriedAdd(5);
        
        echo "Add 5 to 10: " . $addFive(10) . "\n";
        
        // Function composition
        $addThenDouble = Functional::compose($double, $addFive);
        echo "Add 5 then double 3: " . $addThenDouble(3) . "\n";
        
        // Memoization
        $expensiveFunction = Functional::memoize(function ($n) {
            echo "Computing factorial of {$n}...\n";
            return $n <= 1 ? 1 : $n * call_user_func(__FUNCTION__, $n - 1);
        });
        
        echo "Factorial 5 (first call): " . $expensiveFunction(5) . "\n";
        echo "Factorial 5 (cached): " . $expensiveFunction(5) . "\n";
        
        // Collection operations
        $numbers = new Collection([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        
        $evenDoubled = $numbers
            ->filter(fn($n) => $n % 2 === 0)
            ->map(fn($n) => $n * 2)
            ->toArray();
        
        echo "Even numbers doubled: " . implode(', ', $evenDoubled) . "\n";
        
        $sum = $numbers->reduce(0, fn($acc, $n) => $acc + $n);
        echo "Sum of all numbers: {$sum}\n";
        
        // Grouping
        $words = new Collection(['apple', 'banana', 'apricot', 'blueberry', 'cherry']);
        $grouped = $words->groupBy(fn($word) => $word[0]);
        
        echo "Words grouped by first letter:\n";
        foreach ($grouped as $letter => $wordGroup) {
            echo "  {$letter}: " . implode(', ', $wordGroup->toArray()) . "\n";
        }
    }
    
    private static function demonstratePerformanceOptimization(): void
    {
        echo "\n⚡ Performance Optimization:\n";
        
        // Performance measurement
        $measurement = Performance::measure(function () {
            $array = [];
            for ($i = 0; $i < 10000; $i++) {
                $array[] = $i * $i;
            }
            return $array;
        });
        
        echo "Array creation performance:\n";
        echo "  Execution time: " . number_format($measurement['execution_time_ms'], 2) . " ms\n";
        echo "  Memory used: {$measurement['memory_used_bytes']} bytes\n";
        echo "  Array length: " . count($measurement['result']) . "\n";
        
        // Benchmark comparison
        echo "\nBenchmark - Array creation methods:\n";
        
        $arrayFillBenchmark = Performance::benchmark(function () {
            return array_fill(0, 1000, 0);
        }, 1000);
        
        $loopBenchmark = Performance::benchmark(function () {
            $array = [];
            for ($i = 0; $i < 1000; $i++) {
                $array[] = 0;
            }
            return $array;
        }, 1000);
        
        echo "  array_fill: " . number_format($arrayFillBenchmark['average_time_ms'], 4) . " ms avg\n";
        echo "  loop: " . number_format($loopBenchmark['average_time_ms'], 4) . " ms avg\n";
        echo "  Speedup: " . round($loopBenchmark['average_time_ms'] / $arrayFillBenchmark['average_time_ms'], 2) . "x\n";
        
        // Generator demonstration
        echo "\nGenerator examples:\n";
        
        echo "Fibonacci sequence (first 10): ";
        $fibonacci = Generators::fibonacci(10);
        $fibArray = [];
        foreach ($fibonacci as $fib) {
            $fibArray[] = $fib;
        }
        echo implode(', ', $fibArray) . "\n";
        
        echo "Range 1-10 step 2: ";
        $range = Generators::range(1, 10, 2);
        $rangeArray = [];
        foreach ($range as $number) {
            $rangeArray[] = $number;
        }
        echo implode(', ', $rangeArray) . "\n";
        
        // Batch processing
        $largeArray = range(1, 100);
        $batchCount = 0;
        $itemCount = 0;
        
        foreach (Generators::batchProcess($largeArray, 10) as $batch) {
            $batchCount++;
            $itemCount += count($batch);
        }
        
        echo "Processed {$itemCount} items in {$batchCount} batches\n";
    }
}

// Run the demonstration if this file is executed directly
if (basename(__FILE__) === basename($_SERVER['SCRIPT_NAME'])) {
    AdvancedPhpDemo::run();
}

?>