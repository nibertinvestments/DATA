<?php
/**
 * Comprehensive PHP Examples
 * Demonstrates modern PHP features, OOP, traits, namespaces, and best practices
 */

declare(strict_types=1);

namespace Examples;

use DateTime;
use Exception;
use InvalidArgumentException;
use RuntimeException;
use JsonSerializable;
use Iterator;
use ArrayAccess;
use Countable;

// ========== Interfaces and Contracts ==========

interface UserRepositoryInterface
{
    public function create(User $user): int;
    public function findById(int $id): ?User;
    public function findAll(): array;
    public function update(User $user): bool;
    public function delete(int $id): bool;
    public function findByEmail(string $email): ?User;
}

interface ValidatorInterface
{
    public function validate(object $entity): array;
}

interface CacheInterface
{
    public function get(string $key): mixed;
    public function set(string $key, mixed $value, int $ttl = 3600): bool;
    public function delete(string $key): bool;
    public function clear(): bool;
}

// ========== Traits ==========

trait Timestampable
{
    private DateTime $createdAt;
    private DateTime $updatedAt;

    protected function initializeTimestamps(): void
    {
        $this->createdAt = new DateTime();
        $this->updatedAt = new DateTime();
    }

    public function getCreatedAt(): DateTime
    {
        return $this->createdAt;
    }

    public function getUpdatedAt(): DateTime
    {
        return $this->updatedAt;
    }

    public function touch(): void
    {
        $this->updatedAt = new DateTime();
    }
}

trait Validateable
{
    private array $errors = [];

    protected function addError(string $field, string $message): void
    {
        $this->errors[$field][] = $message;
    }

    public function getErrors(): array
    {
        return $this->errors;
    }

    public function hasErrors(): bool
    {
        return !empty($this->errors);
    }

    public function clearErrors(): void
    {
        $this->errors = [];
    }
}

// ========== Value Objects ==========

readonly class Email implements JsonSerializable
{
    public function __construct(private string $value)
    {
        if (!filter_var($value, FILTER_VALIDATE_EMAIL)) {
            throw new InvalidArgumentException("Invalid email format: {$value}");
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

    public function jsonSerialize(): string
    {
        return $this->value;
    }

    public function equals(Email $other): bool
    {
        return $this->value === $other->value;
    }
}

readonly class Money implements JsonSerializable
{
    public function __construct(
        private float $amount,
        private string $currency = 'USD'
    ) {
        if ($amount < 0) {
            throw new InvalidArgumentException('Amount cannot be negative');
        }
    }

    public function getAmount(): float
    {
        return $this->amount;
    }

    public function getCurrency(): string
    {
        return $this->currency;
    }

    public function add(Money $other): Money
    {
        if ($this->currency !== $other->currency) {
            throw new InvalidArgumentException('Cannot add different currencies');
        }
        return new Money($this->amount + $other->amount, $this->currency);
    }

    public function subtract(Money $other): Money
    {
        if ($this->currency !== $other->currency) {
            throw new InvalidArgumentException('Cannot subtract different currencies');
        }
        return new Money($this->amount - $other->amount, $this->currency);
    }

    public function multiply(float $factor): Money
    {
        return new Money($this->amount * $factor, $this->currency);
    }

    public function format(): string
    {
        return sprintf('%.2f %s', $this->amount, $this->currency);
    }

    public function __toString(): string
    {
        return $this->format();
    }

    public function jsonSerialize(): array
    {
        return [
            'amount' => $this->amount,
            'currency' => $this->currency
        ];
    }
}

// ========== Enums (PHP 8.1+) ==========

enum UserRole: string
{
    case ADMIN = 'admin';
    case USER = 'user';
    case MODERATOR = 'moderator';
    case GUEST = 'guest';

    public function getPermissions(): array
    {
        return match($this) {
            self::ADMIN => ['read', 'write', 'delete', 'admin'],
            self::MODERATOR => ['read', 'write', 'moderate'],
            self::USER => ['read', 'write'],
            self::GUEST => ['read']
        };
    }

    public function getDisplayName(): string
    {
        return match($this) {
            self::ADMIN => 'Administrator',
            self::MODERATOR => 'Moderator',
            self::USER => 'User',
            self::GUEST => 'Guest'
        };
    }

    public function hasPermission(string $permission): bool
    {
        return in_array($permission, $this->getPermissions(), true);
    }
}

enum ProductCategory: string
{
    case ELECTRONICS = 'electronics';
    case BOOKS = 'books';
    case CLOTHING = 'clothing';
    case HOME = 'home';
    case SPORTS = 'sports';
}

// ========== Domain Models ==========

class User implements JsonSerializable
{
    use Timestampable, Validateable;

    private int $id;
    private string $username;
    private Email $email;
    private ?int $age;
    private UserRole $role;
    private bool $active;
    private array $metadata;

    public function __construct(
        string $username,
        Email $email,
        ?int $age = null,
        UserRole $role = UserRole::USER,
        bool $active = true,
        array $metadata = []
    ) {
        $this->username = $username;
        $this->email = $email;
        $this->age = $age;
        $this->role = $role;
        $this->active = $active;
        $this->metadata = $metadata;
        $this->initializeTimestamps();
        $this->validate();
    }

    // Getters
    public function getId(): int
    {
        return $this->id ?? 0;
    }

    public function getUsername(): string
    {
        return $this->username;
    }

    public function getEmail(): Email
    {
        return $this->email;
    }

    public function getAge(): ?int
    {
        return $this->age;
    }

    public function getRole(): UserRole
    {
        return $this->role;
    }

    public function isActive(): bool
    {
        return $this->active;
    }

    public function getMetadata(): array
    {
        return $this->metadata;
    }

    // Setters with validation
    public function setId(int $id): void
    {
        $this->id = $id;
    }

    public function setUsername(string $username): void
    {
        $this->username = $username;
        $this->touch();
        $this->validate();
    }

    public function setEmail(Email $email): void
    {
        $this->email = $email;
        $this->touch();
    }

    public function setAge(?int $age): void
    {
        $this->age = $age;
        $this->touch();
        $this->validate();
    }

    public function setRole(UserRole $role): void
    {
        $this->role = $role;
        $this->touch();
    }

    public function activate(): void
    {
        $this->active = true;
        $this->touch();
    }

    public function deactivate(): void
    {
        $this->active = false;
        $this->touch();
    }

    public function setMetadata(array $metadata): void
    {
        $this->metadata = $metadata;
        $this->touch();
    }

    public function addMetadata(string $key, mixed $value): void
    {
        $this->metadata[$key] = $value;
        $this->touch();
    }

    // Business logic
    public function isAdult(): bool
    {
        return $this->age !== null && $this->age >= 18;
    }

    public function hasPermission(string $permission): bool
    {
        return $this->role->hasPermission($permission);
    }

    public function getDisplayName(): string
    {
        return sprintf('%s (%s)', $this->username, $this->role->getDisplayName());
    }

    // Validation
    private function validate(): void
    {
        $this->clearErrors();

        if (strlen($this->username) < 3) {
            $this->addError('username', 'Username must be at least 3 characters long');
        }

        if (strlen($this->username) > 50) {
            $this->addError('username', 'Username cannot exceed 50 characters');
        }

        if (!preg_match('/^[a-zA-Z0-9_]+$/', $this->username)) {
            $this->addError('username', 'Username can only contain letters, numbers, and underscores');
        }

        if ($this->age !== null && ($this->age < 0 || $this->age > 150)) {
            $this->addError('age', 'Age must be between 0 and 150');
        }

        if ($this->hasErrors()) {
            throw new InvalidArgumentException('Validation failed: ' . json_encode($this->getErrors()));
        }
    }

    public function jsonSerialize(): array
    {
        return [
            'id' => $this->getId(),
            'username' => $this->username,
            'email' => $this->email,
            'age' => $this->age,
            'role' => $this->role->value,
            'active' => $this->active,
            'metadata' => $this->metadata,
            'created_at' => $this->createdAt->format('Y-m-d H:i:s'),
            'updated_at' => $this->updatedAt->format('Y-m-d H:i:s')
        ];
    }

    public function __toString(): string
    {
        return sprintf(
            'User{id=%d, username=%s, email=%s, role=%s}',
            $this->getId(),
            $this->username,
            $this->email->getValue(),
            $this->role->value
        );
    }
}

abstract class Product implements JsonSerializable
{
    use Timestampable;

    protected int $id;
    protected string $name;
    protected string $description;
    protected Money $price;
    protected ProductCategory $category;
    protected bool $available;

    public function __construct(
        string $name,
        string $description,
        Money $price,
        ProductCategory $category,
        bool $available = true
    ) {
        $this->name = $name;
        $this->description = $description;
        $this->price = $price;
        $this->category = $category;
        $this->available = $available;
        $this->initializeTimestamps();
    }

    // Abstract methods
    abstract public function calculateShippingCost(): Money;
    abstract public function getProductType(): string;

    // Getters
    public function getId(): int
    {
        return $this->id ?? 0;
    }

    public function getName(): string
    {
        return $this->name;
    }

    public function getDescription(): string
    {
        return $this->description;
    }

    public function getPrice(): Money
    {
        return $this->price;
    }

    public function getCategory(): ProductCategory
    {
        return $this->category;
    }

    public function isAvailable(): bool
    {
        return $this->available;
    }

    // Setters
    public function setId(int $id): void
    {
        $this->id = $id;
    }

    public function setPrice(Money $price): void
    {
        $this->price = $price;
        $this->touch();
    }

    public function setAvailable(bool $available): void
    {
        $this->available = $available;
        $this->touch();
    }

    // Business logic
    public function getTotalCost(): Money
    {
        return $this->price->add($this->calculateShippingCost());
    }

    public function applyDiscount(float $percentage): Money
    {
        if ($percentage < 0 || $percentage > 100) {
            throw new InvalidArgumentException('Discount percentage must be between 0 and 100');
        }
        
        $discountAmount = $this->price->multiply($percentage / 100);
        return $this->price->subtract($discountAmount);
    }

    public function jsonSerialize(): array
    {
        return [
            'id' => $this->getId(),
            'name' => $this->name,
            'description' => $this->description,
            'price' => $this->price,
            'category' => $this->category->value,
            'available' => $this->available,
            'product_type' => $this->getProductType(),
            'shipping_cost' => $this->calculateShippingCost(),
            'total_cost' => $this->getTotalCost(),
            'created_at' => $this->createdAt->format('Y-m-d H:i:s'),
            'updated_at' => $this->updatedAt->format('Y-m-d H:i:s')
        ];
    }
}

class PhysicalProduct extends Product
{
    private float $weight;
    private array $dimensions; // [length, width, height]

    public function __construct(
        string $name,
        string $description,
        Money $price,
        ProductCategory $category,
        float $weight,
        array $dimensions,
        bool $available = true
    ) {
        parent::__construct($name, $description, $price, $category, $available);
        $this->weight = $weight;
        $this->dimensions = $dimensions;
    }

    public function getWeight(): float
    {
        return $this->weight;
    }

    public function getDimensions(): array
    {
        return $this->dimensions;
    }

    public function getVolume(): float
    {
        return $this->dimensions[0] * $this->dimensions[1] * $this->dimensions[2];
    }

    public function calculateShippingCost(): Money
    {
        $weightCost = $this->weight * 2.0;
        $volumeCost = $this->getVolume() * 0.1;
        return new Money($weightCost + $volumeCost, $this->price->getCurrency());
    }

    public function getProductType(): string
    {
        return 'physical';
    }
}

class DigitalProduct extends Product
{
    private int $fileSizeBytes;
    private ?string $downloadUrl;
    private int $downloadLimit;

    public function __construct(
        string $name,
        string $description,
        Money $price,
        ProductCategory $category,
        int $fileSizeBytes,
        int $downloadLimit = 5,
        ?string $downloadUrl = null
    ) {
        parent::__construct($name, $description, $price, $category);
        $this->fileSizeBytes = $fileSizeBytes;
        $this->downloadLimit = $downloadLimit;
        $this->downloadUrl = $downloadUrl;
    }

    public function getFileSizeBytes(): int
    {
        return $this->fileSizeBytes;
    }

    public function getDownloadUrl(): ?string
    {
        return $this->downloadUrl;
    }

    public function getDownloadLimit(): int
    {
        return $this->downloadLimit;
    }

    public function setDownloadUrl(string $url): void
    {
        $this->downloadUrl = $url;
        $this->touch();
    }

    public function getFileSizeFormatted(): string
    {
        $sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        $bytes = $this->fileSizeBytes;
        $factor = floor((strlen((string)$bytes) - 1) / 3);
        return sprintf("%.1f", $bytes / pow(1024, $factor)) . ' ' . $sizes[$factor];
    }

    public function calculateShippingCost(): Money
    {
        return new Money(0, $this->price->getCurrency()); // No shipping for digital products
    }

    public function getProductType(): string
    {
        return 'digital';
    }
}

// ========== Repository Implementation ==========

class InMemoryUserRepository implements UserRepositoryInterface
{
    private array $users = [];
    private int $nextId = 1;

    public function create(User $user): int
    {
        $user->setId($this->nextId);
        $this->users[$this->nextId] = $user;
        return $this->nextId++;
    }

    public function findById(int $id): ?User
    {
        return $this->users[$id] ?? null;
    }

    public function findAll(): array
    {
        return array_values($this->users);
    }

    public function update(User $user): bool
    {
        $id = $user->getId();
        if (!isset($this->users[$id])) {
            return false;
        }
        
        $this->users[$id] = $user;
        return true;
    }

    public function delete(int $id): bool
    {
        if (!isset($this->users[$id])) {
            return false;
        }
        
        unset($this->users[$id]);
        return true;
    }

    public function findByEmail(string $email): ?User
    {
        foreach ($this->users as $user) {
            if ($user->getEmail()->getValue() === $email) {
                return $user;
            }
        }
        return null;
    }

    public function findByRole(UserRole $role): array
    {
        return array_filter($this->users, fn(User $user) => $user->getRole() === $role);
    }

    public function findActive(): array
    {
        return array_filter($this->users, fn(User $user) => $user->isActive());
    }

    public function count(): int
    {
        return count($this->users);
    }
}

// ========== Services ==========

class UserService
{
    public function __construct(
        private UserRepositoryInterface $repository,
        private ValidatorInterface $validator
    ) {}

    public function createUser(string $username, string $email, ?int $age = null): User
    {
        // Check if user already exists
        $existingUser = $this->repository->findByEmail($email);
        if ($existingUser !== null) {
            throw new InvalidArgumentException("User with email {$email} already exists");
        }

        $user = new User(
            username: $username,
            email: new Email($email),
            age: $age
        );

        $errors = $this->validator->validate($user);
        if (!empty($errors)) {
            throw new InvalidArgumentException('Validation failed: ' . json_encode($errors));
        }

        $id = $this->repository->create($user);
        return $this->repository->findById($id);
    }

    public function getUserById(int $id): User
    {
        $user = $this->repository->findById($id);
        if ($user === null) {
            throw new RuntimeException("User with ID {$id} not found");
        }
        return $user;
    }

    public function updateUser(int $id, ?string $username = null, ?string $email = null, ?int $age = null): User
    {
        $user = $this->getUserById($id);

        if ($username !== null) {
            $user->setUsername($username);
        }

        if ($email !== null) {
            $user->setEmail(new Email($email));
        }

        if ($age !== null) {
            $user->setAge($age);
        }

        $errors = $this->validator->validate($user);
        if (!empty($errors)) {
            throw new InvalidArgumentException('Validation failed: ' . json_encode($errors));
        }

        $this->repository->update($user);
        return $user;
    }

    public function deleteUser(int $id): bool
    {
        $user = $this->getUserById($id); // Ensure user exists
        return $this->repository->delete($id);
    }

    public function getAllUsers(): array
    {
        return $this->repository->findAll();
    }

    public function getActiveUsers(): array
    {
        return array_filter(
            $this->repository->findAll(),
            fn(User $user) => $user->isActive()
        );
    }

    public function getUsersByRole(UserRole $role): array
    {
        return array_filter(
            $this->repository->findAll(),
            fn(User $user) => $user->getRole() === $role
        );
    }

    public function searchUsers(string $query): array
    {
        $query = strtolower($query);
        return array_filter(
            $this->repository->findAll(),
            fn(User $user) => 
                str_contains(strtolower($user->getUsername()), $query) ||
                str_contains(strtolower($user->getEmail()->getValue()), $query)
        );
    }
}

// ========== Validation ==========

class UserValidator implements ValidatorInterface
{
    public function validate(object $entity): array
    {
        if (!$entity instanceof User) {
            throw new InvalidArgumentException('Entity must be an instance of User');
        }

        $errors = [];

        // Username validation
        if (strlen($entity->getUsername()) < 3) {
            $errors['username'][] = 'Username must be at least 3 characters long';
        }

        if (strlen($entity->getUsername()) > 50) {
            $errors['username'][] = 'Username cannot exceed 50 characters';
        }

        // Age validation
        if ($entity->getAge() !== null && ($entity->getAge() < 13)) {
            $errors['age'][] = 'Users must be at least 13 years old';
        }

        return $errors;
    }
}

// ========== Cache Implementation ==========

class SimpleCache implements CacheInterface
{
    private array $cache = [];
    private array $expiry = [];

    public function get(string $key): mixed
    {
        if (!isset($this->cache[$key])) {
            return null;
        }

        if (isset($this->expiry[$key]) && time() > $this->expiry[$key]) {
            unset($this->cache[$key], $this->expiry[$key]);
            return null;
        }

        return $this->cache[$key];
    }

    public function set(string $key, mixed $value, int $ttl = 3600): bool
    {
        $this->cache[$key] = $value;
        $this->expiry[$key] = time() + $ttl;
        return true;
    }

    public function delete(string $key): bool
    {
        unset($this->cache[$key], $this->expiry[$key]);
        return true;
    }

    public function clear(): bool
    {
        $this->cache = [];
        $this->expiry = [];
        return true;
    }

    public function size(): int
    {
        return count($this->cache);
    }
}

// ========== Collection Classes ==========

class UserCollection implements Iterator, ArrayAccess, Countable, JsonSerializable
{
    private array $users = [];
    private int $position = 0;

    public function add(User $user): void
    {
        $this->users[] = $user;
    }

    public function remove(User $user): bool
    {
        $key = array_search($user, $this->users, true);
        if ($key !== false) {
            unset($this->users[$key]);
            $this->users = array_values($this->users); // Reindex
            return true;
        }
        return false;
    }

    public function filter(callable $callback): UserCollection
    {
        $filtered = new UserCollection();
        foreach ($this->users as $user) {
            if ($callback($user)) {
                $filtered->add($user);
            }
        }
        return $filtered;
    }

    public function map(callable $callback): array
    {
        return array_map($callback, $this->users);
    }

    public function findBy(string $property, mixed $value): ?User
    {
        foreach ($this->users as $user) {
            $getter = 'get' . ucfirst($property);
            if (method_exists($user, $getter) && $user->$getter() === $value) {
                return $user;
            }
        }
        return null;
    }

    public function sortBy(string $property, string $direction = 'ASC'): UserCollection
    {
        $users = $this->users;
        $getter = 'get' . ucfirst($property);
        
        usort($users, function (User $a, User $b) use ($getter, $direction) {
            $aValue = $a->$getter();
            $bValue = $b->$getter();
            
            $result = $aValue <=> $bValue;
            return $direction === 'DESC' ? -$result : $result;
        });

        $sorted = new UserCollection();
        foreach ($users as $user) {
            $sorted->add($user);
        }
        
        return $sorted;
    }

    // Iterator implementation
    public function current(): User
    {
        return $this->users[$this->position];
    }

    public function key(): int
    {
        return $this->position;
    }

    public function next(): void
    {
        ++$this->position;
    }

    public function rewind(): void
    {
        $this->position = 0;
    }

    public function valid(): bool
    {
        return isset($this->users[$this->position]);
    }

    // ArrayAccess implementation
    public function offsetExists(mixed $offset): bool
    {
        return isset($this->users[$offset]);
    }

    public function offsetGet(mixed $offset): User
    {
        return $this->users[$offset];
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        if ($offset === null) {
            $this->users[] = $value;
        } else {
            $this->users[$offset] = $value;
        }
    }

    public function offsetUnset(mixed $offset): void
    {
        unset($this->users[$offset]);
    }

    // Countable implementation
    public function count(): int
    {
        return count($this->users);
    }

    public function jsonSerialize(): array
    {
        return $this->users;
    }

    public function toArray(): array
    {
        return $this->users;
    }
}

// ========== File Operations ==========

class FileService
{
    public function saveUsersToJson(array $users, string $filename): bool
    {
        $data = array_map(fn(User $user) => $user->jsonSerialize(), $users);
        $json = json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
        
        if ($json === false) {
            throw new RuntimeException('Failed to encode users to JSON');
        }

        return file_put_contents($filename, $json) !== false;
    }

    public function loadUsersFromJson(string $filename): array
    {
        if (!file_exists($filename)) {
            throw new RuntimeException("File not found: {$filename}");
        }

        $json = file_get_contents($filename);
        if ($json === false) {
            throw new RuntimeException("Failed to read file: {$filename}");
        }

        $data = json_decode($json, true);
        if ($data === null) {
            throw new RuntimeException("Invalid JSON in file: {$filename}");
        }

        $users = [];
        foreach ($data as $userData) {
            $user = new User(
                username: $userData['username'],
                email: new Email($userData['email']),
                age: $userData['age'],
                role: UserRole::from($userData['role']),
                active: $userData['active']
            );
            $user->setId($userData['id']);
            $users[] = $user;
        }

        return $users;
    }

    public function saveUsersToCsv(array $users, string $filename): bool
    {
        $file = fopen($filename, 'w');
        if ($file === false) {
            throw new RuntimeException("Failed to open file for writing: {$filename}");
        }

        // Write header
        fputcsv($file, ['ID', 'Username', 'Email', 'Age', 'Role', 'Active', 'Created At']);

        // Write data
        foreach ($users as $user) {
            fputcsv($file, [
                $user->getId(),
                $user->getUsername(),
                $user->getEmail()->getValue(),
                $user->getAge(),
                $user->getRole()->value,
                $user->isActive() ? 'Yes' : 'No',
                $user->getCreatedAt()->format('Y-m-d H:i:s')
            ]);
        }

        fclose($file);
        return true;
    }
}

// ========== Functional Programming Utilities ==========

class FunctionalUtils
{
    public static function map(array $array, callable $callback): array
    {
        return array_map($callback, $array);
    }

    public static function filter(array $array, callable $callback): array
    {
        return array_filter($array, $callback);
    }

    public static function reduce(array $array, callable $callback, mixed $initial = null): mixed
    {
        return array_reduce($array, $callback, $initial);
    }

    public static function groupBy(array $array, callable $keySelector): array
    {
        $groups = [];
        foreach ($array as $item) {
            $key = $keySelector($item);
            $groups[$key][] = $item;
        }
        return $groups;
    }

    public static function sortBy(array $array, callable $keySelector, string $direction = 'ASC'): array
    {
        usort($array, function ($a, $b) use ($keySelector, $direction) {
            $aValue = $keySelector($a);
            $bValue = $keySelector($b);
            $result = $aValue <=> $bValue;
            return $direction === 'DESC' ? -$result : $result;
        });
        return $array;
    }

    public static function unique(array $array, ?callable $keySelector = null): array
    {
        if ($keySelector === null) {
            return array_unique($array);
        }

        $seen = [];
        $result = [];
        
        foreach ($array as $item) {
            $key = $keySelector($item);
            if (!in_array($key, $seen, true)) {
                $seen[] = $key;
                $result[] = $item;
            }
        }
        
        return $result;
    }
}

// ========== Demonstration Functions ==========

function demonstrateBasicFeatures(): void
{
    echo "=== Basic PHP Features Demo ===\n";

    // Create users
    $user1 = new User('alice', new Email('alice@example.com'), 25, UserRole::USER);
    $user2 = new User('bob', new Email('bob@example.com'), 17, UserRole::MODERATOR);

    echo "Created users:\n";
    echo "- {$user1}\n";
    echo "- {$user2}\n";

    // Business logic
    echo "Alice is adult: " . ($user1->isAdult() ? 'yes' : 'no') . "\n";
    echo "Bob is adult: " . ($user2->isAdult() ? 'yes' : 'no') . "\n";

    // Role permissions
    echo "Alice can write: " . ($user1->hasPermission('write') ? 'yes' : 'no') . "\n";
    echo "Bob can moderate: " . ($user2->hasPermission('moderate') ? 'yes' : 'no') . "\n";

    // Value objects
    $price = new Money(99.99, 'USD');
    $discountedPrice = $price->multiply(0.9);
    echo "Original price: {$price}\n";
    echo "Discounted price: {$discountedPrice}\n";

    echo "\n";
}

function demonstrateProducts(): void
{
    echo "=== Product Polymorphism Demo ===\n";

    $products = [
        new PhysicalProduct(
            'Laptop',
            'Gaming laptop',
            new Money(999.99),
            ProductCategory::ELECTRONICS,
            2.5,
            [35, 25, 3]
        ),
        new DigitalProduct(
            'Software',
            'Development tools',
            new Money(199.99),
            ProductCategory::SOFTWARE,
            1024 * 1024 * 500
        )
    ];

    foreach ($products as $product) {
        echo "Product: {$product->getName()}\n";
        echo "  Type: {$product->getProductType()}\n";
        echo "  Price: {$product->getPrice()}\n";
        echo "  Shipping: {$product->calculateShippingCost()}\n";
        echo "  Total: {$product->getTotalCost()}\n";
        
        if ($product instanceof DigitalProduct) {
            echo "  File size: {$product->getFileSizeFormatted()}\n";
        }
        echo "\n";
    }
}

function demonstrateRepository(): void
{
    echo "=== Repository Pattern Demo ===\n";

    $repository = new InMemoryUserRepository();
    $validator = new UserValidator();
    $service = new UserService($repository, $validator);

    // Create users
    $user1 = $service->createUser('charlie', 'charlie@example.com', 30);
    $user2 = $service->createUser('diana', 'diana@example.com', 22);

    echo "Created users:\n";
    echo "- {$user1}\n";
    echo "- {$user2}\n";

    // Update user
    $updatedUser = $service->updateUser($user1->getId(), age: 31);
    echo "Updated user: {$updatedUser}\n";

    // Search users
    $searchResults = $service->searchUsers('charlie');
    echo "Search results for 'charlie': " . count($searchResults) . " users\n";

    // Get by role
    $userRoleUsers = $service->getUsersByRole(UserRole::USER);
    echo "Users with USER role: " . count($userRoleUsers) . " users\n";

    echo "\n";
}

function demonstrateCollections(): void
{
    echo "=== Collections Demo ===\n";

    $collection = new UserCollection();
    
    // Add users
    $collection->add(new User('eve', new Email('eve@example.com'), 28));
    $collection->add(new User('frank', new Email('frank@example.com'), 35));
    $collection->add(new User('grace', new Email('grace@example.com'), 19));

    echo "Collection count: {$collection->count()}\n";

    // Filter adults
    $adults = $collection->filter(fn(User $user) => $user->isAdult());
    echo "Adult users: {$adults->count()}\n";

    // Sort by age
    $sortedByAge = $collection->sortBy('age', 'DESC');
    echo "Users sorted by age (DESC):\n";
    foreach ($sortedByAge as $user) {
        echo "- {$user->getUsername()} ({$user->getAge()})\n";
    }

    // Map to usernames
    $usernames = $collection->map(fn(User $user) => $user->getUsername());
    echo "Usernames: " . implode(', ', $usernames) . "\n";

    echo "\n";
}

function demonstrateFunctionalProgramming(): void
{
    echo "=== Functional Programming Demo ===\n";

    $users = [
        new User('user1', new Email('user1@example.com'), 25),
        new User('user2', new Email('user2@gmail.com'), 17),
        new User('user3', new Email('user3@yahoo.com'), 30),
        new User('user4', new Email('user4@example.com'), 22)
    ];

    // Filter adults
    $adults = FunctionalUtils::filter($users, fn(User $user) => $user->isAdult());
    echo "Adult users: " . count($adults) . "\n";

    // Group by email domain
    $groupedByDomain = FunctionalUtils::groupBy(
        $users,
        fn(User $user) => $user->getEmail()->getDomain()
    );
    echo "Users by domain:\n";
    foreach ($groupedByDomain as $domain => $domainUsers) {
        echo "- {$domain}: " . count($domainUsers) . " users\n";
    }

    // Calculate average age
    $totalAge = FunctionalUtils::reduce(
        $users,
        fn($carry, User $user) => $carry + ($user->getAge() ?? 0),
        0
    );
    $averageAge = $totalAge / count($users);
    echo "Average age: {$averageAge}\n";

    // Sort by username
    $sortedUsers = FunctionalUtils::sortBy(
        $users,
        fn(User $user) => $user->getUsername()
    );
    echo "Sorted usernames: " . implode(', ', array_map(
        fn(User $user) => $user->getUsername(),
        $sortedUsers
    )) . "\n";

    echo "\n";
}

function demonstrateFileOperations(): void
{
    echo "=== File Operations Demo ===\n";

    $fileService = new FileService();
    $users = [
        new User('file_user1', new Email('file1@example.com'), 25),
        new User('file_user2', new Email('file2@example.com'), 30)
    ];

    // Save to JSON
    $jsonFile = '/tmp/users.json';
    $fileService->saveUsersToJson($users, $jsonFile);
    echo "Saved " . count($users) . " users to JSON file\n";

    // Load from JSON
    $loadedUsers = $fileService->loadUsersFromJson($jsonFile);
    echo "Loaded " . count($loadedUsers) . " users from JSON file\n";

    // Save to CSV
    $csvFile = '/tmp/users.csv';
    $fileService->saveUsersToCsv($users, $csvFile);
    echo "Saved users to CSV file\n";

    echo "\n";
}

function demonstrateErrorHandling(): void
{
    echo "=== Error Handling Demo ===\n";

    try {
        // Invalid email
        new Email('invalid-email');
    } catch (InvalidArgumentException $e) {
        echo "Caught email validation error: {$e->getMessage()}\n";
    }

    try {
        // Invalid user data
        new User('', new Email('test@example.com'), -5);
    } catch (InvalidArgumentException $e) {
        echo "Caught user validation error: {$e->getMessage()}\n";
    }

    try {
        // Invalid money operation
        $money1 = new Money(100, 'USD');
        $money2 = new Money(50, 'EUR');
        $money1->add($money2);
    } catch (InvalidArgumentException $e) {
        echo "Caught money operation error: {$e->getMessage()}\n";
    }

    echo "\n";
}

// ========== Main Execution ==========

function main(): void
{
    echo "=== Comprehensive PHP Examples ===\n\n";

    demonstrateBasicFeatures();
    demonstrateProducts();
    demonstrateRepository();
    demonstrateCollections();
    demonstrateFunctionalProgramming();
    demonstrateFileOperations();
    demonstrateErrorHandling();

    echo "=== PHP Features Demonstrated ===\n";
    echo "- Modern PHP 8.1+ features (enums, readonly classes, union types)\n";
    echo "- Object-oriented programming (classes, inheritance, polymorphism)\n";
    echo "- Traits for code reuse\n";
    echo "- Interfaces and dependency injection\n";
    echo "- Value objects and domain modeling\n";
    echo "- Repository pattern\n";
    echo "- Collections with SPL interfaces\n";
    echo "- Functional programming utilities\n";
    echo "- Exception handling and validation\n";
    echo "- File I/O operations\n";
    echo "- JSON serialization\n";
    echo "- Type safety with strict types\n";
    echo "- Design patterns (Repository, Strategy)\n";
}

// Run the demonstration
if (php_sapi_name() === 'cli') {
    main();
}

?>