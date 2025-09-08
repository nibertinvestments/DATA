// Comprehensive Swift Examples
// Demonstrates modern Swift features, protocols, generics, and iOS patterns

import Foundation
import Combine

// ========== Protocols and Protocol-Oriented Programming ==========

protocol Identifiable {
    var id: UUID { get }
}

protocol Validatable {
    func validate() throws
}

protocol Cacheable {
    associatedtype CacheKey: Hashable
    var cacheKey: CacheKey { get }
}

protocol Repository {
    associatedtype Entity: Identifiable
    
    func create(_ entity: Entity) async throws -> Entity
    func read(id: UUID) async throws -> Entity?
    func update(_ entity: Entity) async throws -> Entity
    func delete(id: UUID) async throws -> Bool
    func findAll() async throws -> [Entity]
}

// ========== Error Handling ==========

enum UserError: Error, LocalizedError {
    case invalidEmail(String)
    case invalidAge(Int)
    case userNotFound(UUID)
    case duplicateEmail(String)
    
    var errorDescription: String? {
        switch self {
        case .invalidEmail(let email):
            return "Invalid email format: \(email)"
        case .invalidAge(let age):
            return "Invalid age: \(age). Must be between 0 and 150."
        case .userNotFound(let id):
            return "User not found with ID: \(id)"
        case .duplicateEmail(let email):
            return "User with email \(email) already exists"
        }
    }
}

enum NetworkError: Error {
    case invalidURL
    case noData
    case decodingError(Error)
    case serverError(Int)
    case timeout
}

// ========== Value Types and Data Models ==========

struct User: Identifiable, Codable, Equatable, Hashable {
    let id: UUID
    var username: String
    var email: String
    var age: Int?
    var isActive: Bool
    let createdAt: Date
    var updatedAt: Date
    var roles: Set<UserRole>
    
    init(username: String, email: String, age: Int? = nil, roles: Set<UserRole> = [.user]) {
        self.id = UUID()
        self.username = username
        self.email = email
        self.age = age
        self.isActive = true
        self.createdAt = Date()
        self.updatedAt = Date()
        self.roles = roles
    }
    
    // Computed properties
    var isAdult: Bool {
        guard let age = age else { return false }
        return age >= 18
    }
    
    var displayName: String {
        "\(username) (\(email))"
    }
    
    // Mutating methods
    mutating func updateEmail(_ newEmail: String) throws {
        guard isValidEmail(newEmail) else {
            throw UserError.invalidEmail(newEmail)
        }
        email = newEmail
        updatedAt = Date()
    }
    
    mutating func updateAge(_ newAge: Int) throws {
        guard newAge >= 0 && newAge <= 150 else {
            throw UserError.invalidAge(newAge)
        }
        age = newAge
        updatedAt = Date()
    }
    
    mutating func addRole(_ role: UserRole) {
        roles.insert(role)
        updatedAt = Date()
    }
    
    mutating func deactivate() {
        isActive = false
        updatedAt = Date()
    }
    
    // Helper methods
    func hasRole(_ role: UserRole) -> Bool {
        roles.contains(role)
    }
    
    private func isValidEmail(_ email: String) -> Bool {
        let emailRegex = #"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"#
        return email.range(of: emailRegex, options: .regularExpression) != nil
    }
}

extension User: Validatable {
    func validate() throws {
        guard !username.isEmpty else {
            throw UserError.invalidEmail("Username cannot be empty")
        }
        
        guard isValidEmail(email) else {
            throw UserError.invalidEmail(email)
        }
        
        if let age = age {
            guard age >= 0 && age <= 150 else {
                throw UserError.invalidAge(age)
            }
        }
    }
}

extension User: Cacheable {
    typealias CacheKey = UUID
    var cacheKey: UUID { id }
}

enum UserRole: String, CaseIterable, Codable {
    case admin = "admin"
    case moderator = "moderator"
    case user = "user"
    case guest = "guest"
    
    var permissions: Set<Permission> {
        switch self {
        case .admin:
            return [.read, .write, .delete, .admin]
        case .moderator:
            return [.read, .write, .moderate]
        case .user:
            return [.read, .write]
        case .guest:
            return [.read]
        }
    }
    
    var displayName: String {
        switch self {
        case .admin: return "Administrator"
        case .moderator: return "Moderator"
        case .user: return "User"
        case .guest: return "Guest"
        }
    }
}

enum Permission: String, CaseIterable {
    case read, write, delete, admin, moderate
}

// ========== Generic Types and Collections ==========

struct InMemoryRepository<T: Identifiable & Codable>: Repository {
    typealias Entity = T
    
    private var storage: [UUID: T] = [:]
    private let queue = DispatchQueue(label: "repository.queue", attributes: .concurrent)
    
    func create(_ entity: T) async throws -> T {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async(flags: .barrier) {
                if storage[entity.id] != nil {
                    continuation.resume(throwing: UserError.duplicateEmail("Entity already exists"))
                } else {
                    storage[entity.id] = entity
                    continuation.resume(returning: entity)
                }
            }
        }
    }
    
    func read(id: UUID) async throws -> T? {
        return await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: storage[id])
            }
        }
    }
    
    func update(_ entity: T) async throws -> T {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async(flags: .barrier) {
                if storage[entity.id] != nil {
                    storage[entity.id] = entity
                    continuation.resume(returning: entity)
                } else {
                    continuation.resume(throwing: UserError.userNotFound(entity.id))
                }
            }
        }
    }
    
    func delete(id: UUID) async throws -> Bool {
        return await withCheckedContinuation { continuation in
            queue.async(flags: .barrier) {
                let existed = storage.removeValue(forKey: id) != nil
                continuation.resume(returning: existed)
            }
        }
    }
    
    func findAll() async throws -> [T] {
        return await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: Array(storage.values))
            }
        }
    }
}

// ========== Generic Cache Implementation ==========

actor Cache<Key: Hashable, Value> {
    private var storage: [Key: CacheEntry<Value>] = [:]
    private let maxSize: Int
    private let ttl: TimeInterval
    
    init(maxSize: Int = 100, ttl: TimeInterval = 300) {
        self.maxSize = maxSize
        self.ttl = ttl
    }
    
    func set(_ value: Value, forKey key: Key) {
        cleanupExpiredEntries()
        
        if storage.count >= maxSize && storage[key] == nil {
            // Remove oldest entry
            let oldestKey = storage.min { $0.value.createdAt < $1.value.createdAt }?.key
            if let keyToRemove = oldestKey {
                storage.removeValue(forKey: keyToRemove)
            }
        }
        
        storage[key] = CacheEntry(value: value)
    }
    
    func get(forKey key: Key) -> Value? {
        guard let entry = storage[key] else { return nil }
        
        if Date().timeIntervalSince(entry.createdAt) > ttl {
            storage.removeValue(forKey: key)
            return nil
        }
        
        return entry.value
    }
    
    func remove(forKey key: Key) {
        storage.removeValue(forKey: key)
    }
    
    func clear() {
        storage.removeAll()
    }
    
    private func cleanupExpiredEntries() {
        let now = Date()
        storage = storage.filter { now.timeIntervalSince($0.value.createdAt) <= ttl }
    }
}

private struct CacheEntry<Value> {
    let value: Value
    let createdAt: Date
    
    init(value: Value) {
        self.value = value
        self.createdAt = Date()
    }
}

// ========== Service Layer with Dependency Injection ==========

protocol UserServiceProtocol {
    func createUser(username: String, email: String, age: Int?) async throws -> User
    func getUser(id: UUID) async throws -> User?
    func updateUser(_ user: User) async throws -> User
    func deleteUser(id: UUID) async throws -> Bool
    func getAllUsers() async throws -> [User]
    func searchUsers(query: String) async throws -> [User]
}

final class UserService: UserServiceProtocol {
    private let repository: InMemoryRepository<User>
    private let cache: Cache<UUID, User>
    private let validator: UserValidator
    
    init(repository: InMemoryRepository<User> = InMemoryRepository<User>(),
         cache: Cache<UUID, User> = Cache<UUID, User>(),
         validator: UserValidator = UserValidator()) {
        self.repository = repository
        self.cache = cache
        self.validator = validator
    }
    
    func createUser(username: String, email: String, age: Int?) async throws -> User {
        var user = User(username: username, email: email, age: age)
        try user.validate()
        try validator.validateUniqueEmail(email, in: await repository.findAll())
        
        let createdUser = try await repository.create(user)
        await cache.set(createdUser, forKey: createdUser.id)
        
        return createdUser
    }
    
    func getUser(id: UUID) async throws -> User? {
        // Check cache first
        if let cachedUser = await cache.get(forKey: id) {
            return cachedUser
        }
        
        // Fetch from repository
        let user = try await repository.read(id: id)
        if let user = user {
            await cache.set(user, forKey: id)
        }
        
        return user
    }
    
    func updateUser(_ user: User) async throws -> User {
        try user.validate()
        let updatedUser = try await repository.update(user)
        await cache.set(updatedUser, forKey: updatedUser.id)
        return updatedUser
    }
    
    func deleteUser(id: UUID) async throws -> Bool {
        await cache.remove(forKey: id)
        return try await repository.delete(id: id)
    }
    
    func getAllUsers() async throws -> [User] {
        return try await repository.findAll()
    }
    
    func searchUsers(query: String) async throws -> [User] {
        let allUsers = try await repository.findAll()
        let lowercaseQuery = query.lowercased()
        
        return allUsers.filter { user in
            user.username.lowercased().contains(lowercaseQuery) ||
            user.email.lowercased().contains(lowercaseQuery)
        }
    }
}

struct UserValidator {
    func validateUniqueEmail(_ email: String, in users: [User]) throws {
        if users.contains(where: { $0.email == email }) {
            throw UserError.duplicateEmail(email)
        }
    }
}

// ========== Functional Programming Patterns ==========

extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        
        for element in self {
            let transformed = try await transform(element)
            result.append(transformed)
        }
        
        return result
    }
    
    func asyncCompactMap<T>(_ transform: (Element) async throws -> T?) async rethrows -> [T] {
        var result: [T] = []
        
        for element in self {
            if let transformed = try await transform(element) {
                result.append(transformed)
            }
        }
        
        return result
    }
    
    func asyncFilter(_ isIncluded: (Element) async throws -> Bool) async rethrows -> [Element] {
        var result: [Element] = []
        
        for element in self {
            if try await isIncluded(element) {
                result.append(element)
            }
        }
        
        return result
    }
}

struct FunctionalUtils {
    static func pipe<A, B, C>(_ value: A, _ f1: (A) throws -> B, _ f2: (B) throws -> C) rethrows -> C {
        try f2(f1(value))
    }
    
    static func compose<A, B, C>(_ f1: @escaping (A) throws -> B, _ f2: @escaping (B) throws -> C) -> (A) throws -> C {
        return { a in try f2(f1(a)) }
    }
    
    static func curry<A, B, C>(_ function: @escaping (A, B) throws -> C) -> (A) -> (B) throws -> C {
        return { a in { b in try function(a, b) } }
    }
    
    static func memoize<Input: Hashable, Output>(_ function: @escaping (Input) -> Output) -> (Input) -> Output {
        var cache: [Input: Output] = [:]
        return { input in
            if let cached = cache[input] {
                return cached
            }
            let result = function(input)
            cache[input] = result
            return result
        }
    }
}

// ========== Combine Framework Integration ==========

@available(iOS 13.0, macOS 10.15, *)
final class UserPublisher: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading: Bool = false
    @Published var error: Error?
    
    private let userService: UserServiceProtocol
    private var cancellables = Set<AnyCancellable>()
    
    init(userService: UserServiceProtocol) {
        self.userService = userService
    }
    
    func loadUsers() {
        isLoading = true
        error = nil
        
        Task { @MainActor in
            do {
                let fetchedUsers = try await userService.getAllUsers()
                self.users = fetchedUsers
                self.isLoading = false
            } catch {
                self.error = error
                self.isLoading = false
            }
        }
    }
    
    func createUser(username: String, email: String, age: Int?) {
        Task { @MainActor in
            do {
                let newUser = try await userService.createUser(username: username, email: email, age: age)
                self.users.append(newUser)
            } catch {
                self.error = error
            }
        }
    }
    
    func deleteUser(id: UUID) {
        Task { @MainActor in
            do {
                let success = try await userService.deleteUser(id: id)
                if success {
                    self.users.removeAll { $0.id == id }
                }
            } catch {
                self.error = error
            }
        }
    }
}

// ========== Result Builders ==========

@resultBuilder
struct UserListBuilder {
    static func buildBlock(_ users: User...) -> [User] {
        return users
    }
    
    static func buildOptional(_ user: User?) -> [User] {
        return user.map { [$0] } ?? []
    }
    
    static func buildEither(first user: User) -> [User] {
        return [user]
    }
    
    static func buildEither(second users: [User]) -> [User] {
        return users
    }
    
    static func buildArray(_ users: [[User]]) -> [User] {
        return users.flatMap { $0 }
    }
}

struct UserListFactory {
    @UserListBuilder
    static func createSampleUsers() -> [User] {
        User(username: "alice", email: "alice@example.com", age: 25)
        User(username: "bob", email: "bob@example.com", age: 30)
        
        if Bool.random() {
            User(username: "charlie", email: "charlie@example.com", age: 22)
        }
        
        for i in 1...3 {
            User(username: "user\(i)", email: "user\(i)@example.com", age: 20 + i)
        }
    }
}

// ========== Property Wrappers ==========

@propertyWrapper
struct Clamped<T: Comparable> {
    private var value: T
    private let range: ClosedRange<T>
    
    init(wrappedValue: T, _ range: ClosedRange<T>) {
        self.range = range
        self.value = min(max(wrappedValue, range.lowerBound), range.upperBound)
    }
    
    var wrappedValue: T {
        get { value }
        set { value = min(max(newValue, range.lowerBound), range.upperBound) }
    }
}

@propertyWrapper
struct UserDefault<T> {
    let key: String
    let defaultValue: T
    
    var wrappedValue: T {
        get {
            UserDefaults.standard.object(forKey: key) as? T ?? defaultValue
        }
        set {
            UserDefaults.standard.set(newValue, forKey: key)
        }
    }
}

struct AppSettings {
    @UserDefault(key: "max_users", defaultValue: 100)
    static var maxUsers: Int
    
    @UserDefault(key: "cache_enabled", defaultValue: true)
    static var cacheEnabled: Bool
    
    @Clamped(1...50)
    var pageSize: Int = 20
}

// ========== Extensions and Protocol Extensions ==========

extension Collection {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[index(startIndex, offsetBy: $0)..<index(startIndex, offsetBy: min($0 + size, count))])
        }
    }
    
    func unique<T: Hashable>(by keyPath: KeyPath<Element, T>) -> [Element] {
        var seen = Set<T>()
        return filter { element in
            let key = element[keyPath: keyPath]
            return seen.insert(key).inserted
        }
    }
}

extension Sequence {
    func grouped<Key: Hashable>(by keyPath: KeyPath<Element, Key>) -> [Key: [Element]] {
        return Dictionary(grouping: self) { $0[keyPath: keyPath] }
    }
    
    func sorted<T: Comparable>(by keyPath: KeyPath<Element, T>, ascending: Bool = true) -> [Element] {
        return sorted { lhs, rhs in
            ascending ? lhs[keyPath: keyPath] < rhs[keyPath: keyPath] : lhs[keyPath: keyPath] > rhs[keyPath: keyPath]
        }
    }
}

// ========== Demonstration Class ==========

@MainActor
class SwiftExamplesDemo {
    static func runAllDemos() async {
        print("=== Comprehensive Swift Examples Demo ===\n")
        
        await demonstrateBasicTypes()
        await demonstrateGenericRepository()
        await demonstrateAsyncProgramming()
        await demonstrateFunctionalProgramming()
        await demonstrateErrorHandling()
        await demonstrateCollectionExtensions()
        await demonstratePropertyWrappers()
        await demonstrateResultBuilders()
        
        printFeaturesSummary()
    }
    
    private static func demonstrateBasicTypes() async {
        print("=== Basic Types and Protocols Demo ===")
        
        var user1 = User(username: "alice", email: "alice@example.com", age: 25, roles: [.user, .moderator])
        var user2 = User(username: "bob", email: "bob@example.com", age: 17)
        
        print("Created users:")
        print("- \(user1.displayName) (Adult: \(user1.isAdult))")
        print("- \(user2.displayName) (Adult: \(user2.isAdult))")
        
        // Demonstrate mutations
        try? user1.updateAge(26)
        user1.addRole(.admin)
        print("Updated user: \(user1.displayName) with roles: \(user1.roles)")
        
        // Demonstrate role permissions
        let adminPermissions = UserRole.admin.permissions
        print("Admin permissions: \(adminPermissions)")
        print()
    }
    
    private static func demonstrateGenericRepository() async {
        print("=== Generic Repository Demo ===")
        
        let repository = InMemoryRepository<User>()
        let userService = UserService(repository: repository)
        
        do {
            // Create users
            let user1 = try await userService.createUser(username: "charlie", email: "charlie@example.com", age: 30)
            let user2 = try await userService.createUser(username: "diana", email: "diana@example.com", age: 25)
            
            print("Created users:")
            print("- \(user1.displayName)")
            print("- \(user2.displayName)")
            
            // Fetch users
            let allUsers = try await userService.getAllUsers()
            print("Total users: \(allUsers.count)")
            
            // Search users
            let searchResults = try await userService.searchUsers(query: "charlie")
            print("Search results for 'charlie': \(searchResults.count) users")
            
        } catch {
            print("Error: \(error)")
        }
        print()
    }
    
    private static func demonstrateAsyncProgramming() async {
        print("=== Async Programming Demo ===")
        
        let numbers = [1, 2, 3, 4, 5]
        
        // Async map
        let doubled = await numbers.asyncMap { number in
            // Simulate async work
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            return number * 2
        }
        print("Doubled numbers: \(doubled)")
        
        // Async filter
        let evens = await numbers.asyncFilter { number in
            try? await Task.sleep(nanoseconds: 5_000_000) // 5ms
            return number % 2 == 0
        }
        print("Even numbers: \(evens)")
        print()
    }
    
    private static func demonstrateFunctionalProgramming() async {
        print("=== Functional Programming Demo ===")
        
        // Function composition
        let addOne: (Int) -> Int = { $0 + 1 }
        let multiplyByTwo: (Int) -> Int = { $0 * 2 }
        
        let composed = FunctionalUtils.compose(addOne, multiplyByTwo)
        let result = try! composed(5) // (5 + 1) * 2 = 12
        print("Composed function result: \(result)")
        
        // Currying
        let multiply: (Int, Int) -> Int = { $0 * $1 }
        let double = FunctionalUtils.curry(multiply)(2)
        print("Curried function result: \(try! double(6))") // 2 * 6 = 12
        
        // Memoization
        let expensiveFunction = FunctionalUtils.memoize { (n: Int) -> Int in
            print("Computing for \(n)...")
            return n * n
        }
        
        print("First call: \(expensiveFunction(5))")
        print("Second call: \(expensiveFunction(5))") // Should not print "Computing..."
        print()
    }
    
    private static func demonstrateErrorHandling() async {
        print("=== Error Handling Demo ===")
        
        let userService = UserService()
        
        do {
            // Try to create user with invalid email
            let _ = try await userService.createUser(username: "test", email: "invalid-email", age: 25)
        } catch let error as UserError {
            print("User error caught: \(error.errorDescription ?? "Unknown error")")
        } catch {
            print("Other error: \(error)")
        }
        
        do {
            // Try to create user with invalid age
            let _ = try await userService.createUser(username: "test", email: "test@example.com", age: 200)
        } catch {
            print("Validation error: \(error)")
        }
        print()
    }
    
    private static func demonstrateCollectionExtensions() async {
        print("=== Collection Extensions Demo ===")
        
        let users = UserListFactory.createSampleUsers()
        
        // Chunking
        let chunks = users.chunked(into: 2)
        print("Users in chunks of 2: \(chunks.count) chunks")
        
        // Grouping
        let usersByRole = users.flatMap { $0.roles }.grouped(by: \.self)
        print("Users by role: \(usersByRole.mapValues { $0.count })")
        
        // Unique users by email domain
        let uniqueByDomain = users.unique(by: \.email)
        print("Unique users by email: \(uniqueByDomain.count)")
        
        // Sorting
        let sortedByUsername = users.sorted(by: \.username)
        print("Users sorted by username: \(sortedByUsername.map(\.username))")
        print()
    }
    
    private static func demonstratePropertyWrappers() async {
        print("=== Property Wrappers Demo ===")
        
        var settings = AppSettings()
        settings.pageSize = 100  // Will be clamped to 50
        print("Page size (clamped): \(settings.pageSize)")
        
        AppSettings.maxUsers = 200
        print("Max users setting: \(AppSettings.maxUsers)")
        
        print("Cache enabled: \(AppSettings.cacheEnabled)")
        print()
    }
    
    private static func demonstrateResultBuilders() async {
        print("=== Result Builders Demo ===")
        
        let sampleUsers = UserListFactory.createSampleUsers()
        print("Created \(sampleUsers.count) sample users using result builder:")
        sampleUsers.forEach { user in
            print("- \(user.displayName)")
        }
        print()
    }
    
    private static func printFeaturesSummary() {
        print("=== Swift Features Demonstrated ===")
        print("- Protocol-oriented programming")
        print("- Value types (structs) and reference types (classes)")
        print("- Generics and associated types")
        print("- Async/await concurrency")
        print("- Error handling with typed errors")
        print("- Property wrappers")
        print("- Result builders")
        print("- Extensions and protocol extensions")
        print("- Functional programming patterns")
        print("- Actor-based concurrency (Cache)")
        print("- Memory management with ARC")
        print("- Type safety and optionals")
        print("- Collection protocols and algorithms")
        print("- Combine framework integration")
        print("- Modern Swift syntax and features")
    }
}

// Example usage
Task {
    await SwiftExamplesDemo.runAllDemos()
}