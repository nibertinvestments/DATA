// Comprehensive Swift iOS and macOS Development Examples
// Demonstrates modern Swift features, iOS frameworks, and architectural patterns

import Foundation
import SwiftUI
import Combine
import CoreData
import Network
import CryptoKit

// ============ Protocol-Oriented Programming ============

/**
 * Protocol-oriented architecture for data persistence
 */
protocol DataPersistable {
    associatedtype T
    func save(_ item: T) async throws
    func load(id: String) async throws -> T?
    func loadAll() async throws -> [T]
    func delete(id: String) async throws
}

protocol Identifiable {
    var id: String { get }
}

protocol Validatable {
    func validate() throws
}

// ============ Custom Errors ============

enum DataError: Error, LocalizedError {
    case notFound
    case invalidData
    case networkFailure
    case persistenceFailure(String)
    case validationError(String)
    
    var errorDescription: String? {
        switch self {
        case .notFound:
            return "Item not found"
        case .invalidData:
            return "Invalid data format"
        case .networkFailure:
            return "Network request failed"
        case .persistenceFailure(let message):
            return "Persistence error: \(message)"
        case .validationError(let message):
            return "Validation error: \(message)"
        }
    }
}

enum APIError: Error {
    case invalidURL
    case noData
    case decodingError
    case serverError(Int)
}

// ============ Domain Models ============

struct User: Codable, Identifiable, Validatable, Equatable {
    let id: String
    var name: String
    var email: String
    var age: Int
    var profileImageURL: URL?
    var preferences: UserPreferences
    var createdAt: Date
    var isActive: Bool
    
    init(name: String, email: String, age: Int) {
        self.id = UUID().uuidString
        self.name = name
        self.email = email
        self.age = age
        self.profileImageURL = nil
        self.preferences = UserPreferences()
        self.createdAt = Date()
        self.isActive = true
    }
    
    func validate() throws {
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DataError.validationError("Name cannot be empty")
        }
        
        guard email.contains("@") && email.contains(".") else {
            throw DataError.validationError("Invalid email format")
        }
        
        guard age >= 0 && age <= 150 else {
            throw DataError.validationError("Age must be between 0 and 150")
        }
    }
    
    mutating func updateProfile(name: String? = nil, email: String? = nil, age: Int? = nil) {
        if let name = name { self.name = name }
        if let email = email { self.email = email }
        if let age = age { self.age = age }
    }
    
    func isAdult() -> Bool {
        return age >= 18
    }
}

struct UserPreferences: Codable {
    var theme: Theme = .system
    var notifications: NotificationSettings = NotificationSettings()
    var privacy: PrivacySettings = PrivacySettings()
    
    enum Theme: String, Codable, CaseIterable {
        case light, dark, system
        
        var displayName: String {
            switch self {
            case .light: return "Light"
            case .dark: return "Dark"
            case .system: return "System"
            }
        }
    }
}

struct NotificationSettings: Codable {
    var email: Bool = true
    var push: Bool = true
    var marketing: Bool = false
    var security: Bool = true
}

struct PrivacySettings: Codable {
    var profileVisible: Bool = true
    var analyticsEnabled: Bool = false
    var locationSharing: Bool = false
}

struct Task: Codable, Identifiable, Validatable {
    let id: String
    var title: String
    var description: String
    var priority: Priority
    var status: Status
    var dueDate: Date?
    var tags: [String]
    var assignedTo: String?
    var createdAt: Date
    var updatedAt: Date
    
    enum Priority: String, Codable, CaseIterable {
        case low = "low"
        case medium = "medium"
        case high = "high"
        case urgent = "urgent"
        
        var color: String {
            switch self {
            case .low: return "blue"
            case .medium: return "yellow"
            case .high: return "orange"
            case .urgent: return "red"
            }
        }
        
        var sortOrder: Int {
            switch self {
            case .low: return 0
            case .medium: return 1
            case .high: return 2
            case .urgent: return 3
            }
        }
    }
    
    enum Status: String, Codable, CaseIterable {
        case todo = "todo"
        case inProgress = "in_progress"
        case review = "review"
        case done = "done"
        
        var displayName: String {
            switch self {
            case .todo: return "To Do"
            case .inProgress: return "In Progress"
            case .review: return "Review"
            case .done: return "Done"
            }
        }
    }
    
    init(title: String, description: String, priority: Priority = .medium) {
        self.id = UUID().uuidString
        self.title = title
        self.description = description
        self.priority = priority
        self.status = .todo
        self.dueDate = nil
        self.tags = []
        self.assignedTo = nil
        self.createdAt = Date()
        self.updatedAt = Date()
    }
    
    func validate() throws {
        guard !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DataError.validationError("Title cannot be empty")
        }
    }
    
    mutating func updateStatus(_ newStatus: Status) {
        status = newStatus
        updatedAt = Date()
    }
    
    mutating func addTag(_ tag: String) {
        if !tags.contains(tag) {
            tags.append(tag)
            updatedAt = Date()
        }
    }
    
    func isOverdue() -> Bool {
        guard let dueDate = dueDate else { return false }
        return Date() > dueDate && status != .done
    }
}

// ============ Repository Pattern Implementation ============

actor UserRepository: DataPersistable {
    typealias T = User
    
    private var users: [String: User] = [:]
    private let persistenceManager: PersistenceManager
    
    init(persistenceManager: PersistenceManager = LocalPersistenceManager()) {
        self.persistenceManager = persistenceManager
    }
    
    func save(_ user: User) async throws {
        try user.validate()
        users[user.id] = user
        try await persistenceManager.save(user, key: user.id)
    }
    
    func load(id: String) async throws -> User? {
        if let user = users[id] {
            return user
        }
        
        let user: User? = try await persistenceManager.load(key: id)
        if let user = user {
            users[user.id] = user
        }
        return user
    }
    
    func loadAll() async throws -> [User] {
        let persistedUsers: [User] = try await persistenceManager.loadAll()
        for user in persistedUsers {
            users[user.id] = user
        }
        return Array(users.values)
    }
    
    func delete(id: String) async throws {
        users.removeValue(forKey: id)
        try await persistenceManager.delete(key: id)
    }
    
    func search(query: String) async -> [User] {
        let lowercaseQuery = query.lowercased()
        return users.values.filter { user in
            user.name.lowercased().contains(lowercaseQuery) ||
            user.email.lowercased().contains(lowercaseQuery)
        }
    }
}

actor TaskRepository: DataPersistable {
    typealias T = Task
    
    private var tasks: [String: Task] = [:]
    private let persistenceManager: PersistenceManager
    
    init(persistenceManager: PersistenceManager = LocalPersistenceManager()) {
        self.persistenceManager = persistenceManager
    }
    
    func save(_ task: Task) async throws {
        try task.validate()
        tasks[task.id] = task
        try await persistenceManager.save(task, key: task.id)
    }
    
    func load(id: String) async throws -> Task? {
        if let task = tasks[id] {
            return task
        }
        
        let task: Task? = try await persistenceManager.load(key: id)
        if let task = task {
            tasks[task.id] = task
        }
        return task
    }
    
    func loadAll() async throws -> [Task] {
        let persistedTasks: [Task] = try await persistenceManager.loadAll()
        for task in persistedTasks {
            tasks[task.id] = task
        }
        return Array(tasks.values)
    }
    
    func delete(id: String) async throws {
        tasks.removeValue(forKey: id)
        try await persistenceManager.delete(key: id)
    }
    
    func getTasksByStatus(_ status: Task.Status) async -> [Task] {
        return tasks.values.filter { $0.status == status }
    }
    
    func getTasksByPriority(_ priority: Task.Priority) async -> [Task] {
        return tasks.values.filter { $0.priority == priority }
    }
    
    func getOverdueTasks() async -> [Task] {
        return tasks.values.filter { $0.isOverdue() }
    }
    
    func getTasksAssignedTo(_ userId: String) async -> [Task] {
        return tasks.values.filter { $0.assignedTo == userId }
    }
}

// ============ Persistence Layer ============

protocol PersistenceManager {
    func save<T: Codable>(_ item: T, key: String) async throws
    func load<T: Codable>(key: String) async throws -> T?
    func loadAll<T: Codable>() async throws -> [T]
    func delete(key: String) async throws
}

class LocalPersistenceManager: PersistenceManager {
    private let documentsDirectory: URL
    
    init() {
        documentsDirectory = FileManager.default.urls(for: .documentDirectory, 
                                                     in: .userDomainMask).first!
    }
    
    func save<T: Codable>(_ item: T, key: String) async throws {
        let url = documentsDirectory.appendingPathComponent("\(key).json")
        let data = try JSONEncoder().encode(item)
        try data.write(to: url)
    }
    
    func load<T: Codable>(key: String) async throws -> T? {
        let url = documentsDirectory.appendingPathComponent("\(key).json")
        
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T.self, from: data)
    }
    
    func loadAll<T: Codable>() async throws -> [T] {
        let urls = try FileManager.default.contentsOfDirectory(at: documentsDirectory,
                                                              includingPropertiesForKeys: nil)
        var items: [T] = []
        
        for url in urls where url.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: url)
                let item = try JSONDecoder().decode(T.self, from: data)
                items.append(item)
            } catch {
                // Skip invalid files
                print("Failed to load item from \(url): \(error)")
            }
        }
        
        return items
    }
    
    func delete(key: String) async throws {
        let url = documentsDirectory.appendingPathComponent("\(key).json")
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}

// ============ Network Layer ============

protocol NetworkService {
    func request<T: Codable>(_ endpoint: APIEndpoint) async throws -> T
    func upload(data: Data, to endpoint: APIEndpoint) async throws -> UploadResponse
    func download(from url: URL) async throws -> Data
}

struct APIEndpoint {
    let path: String
    let method: HTTPMethod
    let headers: [String: String]
    let body: Data?
    
    enum HTTPMethod: String {
        case GET, POST, PUT, DELETE, PATCH
    }
}

struct UploadResponse: Codable {
    let id: String
    let url: URL
    let size: Int
}

class URLSessionNetworkService: NetworkService {
    private let baseURL: URL
    private let session: URLSession
    
    init(baseURL: URL) {
        self.baseURL = baseURL
        
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }
    
    func request<T: Codable>(_ endpoint: APIEndpoint) async throws -> T {
        let url = baseURL.appendingPathComponent(endpoint.path)
        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method.rawValue
        request.httpBody = endpoint.body
        
        for (key, value) in endpoint.headers {
            request.setValue(value, forHTTPHeaderField: key)
        }
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidURL
        }
        
        guard 200...299 ~= httpResponse.statusCode else {
            throw APIError.serverError(httpResponse.statusCode)
        }
        
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError.decodingError
        }
    }
    
    func upload(data: Data, to endpoint: APIEndpoint) async throws -> UploadResponse {
        let url = baseURL.appendingPathComponent(endpoint.path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        
        let (responseData, response) = try await session.upload(for: request, from: data)
        
        guard let httpResponse = response as? HTTPURLResponse,
              200...299 ~= httpResponse.statusCode else {
            throw APIError.serverError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        
        return try JSONDecoder().decode(UploadResponse.self, from: responseData)
    }
    
    func download(from url: URL) async throws -> Data {
        let (data, response) = try await session.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse,
              200...299 ~= httpResponse.statusCode else {
            throw APIError.serverError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        
        return data
    }
}

// ============ Business Logic / Use Cases ============

@MainActor
class UserService: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let repository: UserRepository
    private let networkService: NetworkService?
    
    init(repository: UserRepository, networkService: NetworkService? = nil) {
        self.repository = repository
        self.networkService = networkService
    }
    
    func loadUsers() async {
        isLoading = true
        errorMessage = nil
        
        do {
            users = try await repository.loadAll()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func createUser(name: String, email: String, age: Int) async {
        let user = User(name: name, email: email, age: age)
        
        do {
            try await repository.save(user)
            users.append(user)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func updateUser(_ user: User) async {
        do {
            try await repository.save(user)
            if let index = users.firstIndex(where: { $0.id == user.id }) {
                users[index] = user
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func deleteUser(_ user: User) async {
        do {
            try await repository.delete(id: user.id)
            users.removeAll { $0.id == user.id }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func searchUsers(query: String) async {
        guard !query.isEmpty else {
            await loadUsers()
            return
        }
        
        isLoading = true
        users = await repository.search(query: query)
        isLoading = false
    }
}

@MainActor
class TaskService: ObservableObject {
    @Published var tasks: [Task] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let repository: TaskRepository
    private var cancellables = Set<AnyCancellable>()
    
    init(repository: TaskRepository) {
        self.repository = repository
    }
    
    func loadTasks() async {
        isLoading = true
        errorMessage = nil
        
        do {
            tasks = try await repository.loadAll()
            sortTasks()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func createTask(title: String, description: String, priority: Task.Priority) async {
        let task = Task(title: title, description: description, priority: priority)
        
        do {
            try await repository.save(task)
            tasks.append(task)
            sortTasks()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func updateTask(_ task: Task) async {
        do {
            try await repository.save(task)
            if let index = tasks.firstIndex(where: { $0.id == task.id }) {
                tasks[index] = task
            }
            sortTasks()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func deleteTask(_ task: Task) async {
        do {
            try await repository.delete(id: task.id)
            tasks.removeAll { $0.id == task.id }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func updateTaskStatus(_ task: Task, newStatus: Task.Status) async {
        var updatedTask = task
        updatedTask.updateStatus(newStatus)
        await updateTask(updatedTask)
    }
    
    func getTasksByStatus(_ status: Task.Status) -> [Task] {
        return tasks.filter { $0.status == status }
    }
    
    func getOverdueTasks() -> [Task] {
        return tasks.filter { $0.isOverdue() }
    }
    
    private func sortTasks() {
        tasks.sort { task1, task2 in
            if task1.priority.sortOrder != task2.priority.sortOrder {
                return task1.priority.sortOrder > task2.priority.sortOrder
            }
            return task1.createdAt < task2.createdAt
        }
    }
}

// ============ SwiftUI Views ============

struct ContentView: View {
    @StateObject private var userService = UserService(repository: UserRepository())
    @StateObject private var taskService = TaskService(repository: TaskRepository())
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            UserListView()
                .environmentObject(userService)
                .tabItem {
                    Image(systemName: "person.2")
                    Text("Users")
                }
                .tag(0)
            
            TaskListView()
                .environmentObject(taskService)
                .tabItem {
                    Image(systemName: "list.bullet")
                    Text("Tasks")
                }
                .tag(1)
            
            AnalyticsView()
                .environmentObject(taskService)
                .tabItem {
                    Image(systemName: "chart.bar")
                    Text("Analytics")
                }
                .tag(2)
        }
        .task {
            await userService.loadUsers()
            await taskService.loadTasks()
        }
    }
}

struct UserListView: View {
    @EnvironmentObject var userService: UserService
    @State private var showingAddUser = false
    @State private var searchText = ""
    
    var body: some View {
        NavigationView {
            List {
                ForEach(userService.users) { user in
                    UserRowView(user: user)
                }
                .onDelete(perform: deleteUsers)
            }
            .navigationTitle("Users")
            .searchable(text: $searchText)
            .onChange(of: searchText) { newValue in
                Task {
                    await userService.searchUsers(query: newValue)
                }
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Add") {
                        showingAddUser = true
                    }
                }
            }
            .sheet(isPresented: $showingAddUser) {
                AddUserView()
                    .environmentObject(userService)
            }
            .refreshable {
                await userService.loadUsers()
            }
            .overlay {
                if userService.isLoading {
                    ProgressView("Loading...")
                }
            }
        }
    }
    
    private func deleteUsers(offsets: IndexSet) {
        Task {
            for index in offsets {
                await userService.deleteUser(userService.users[index])
            }
        }
    }
}

struct UserRowView: View {
    let user: User
    
    var body: some View {
        HStack {
            AsyncImage(url: user.profileImageURL) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                Image(systemName: "person.circle.fill")
                    .foregroundColor(.gray)
            }
            .frame(width: 50, height: 50)
            .clipShape(Circle())
            
            VStack(alignment: .leading, spacing: 4) {
                Text(user.name)
                    .font(.headline)
                
                Text(user.email)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text("Age: \(user.age)")
                        .font(.caption)
                    
                    Spacer()
                    
                    if user.isAdult() {
                        Text("Adult")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.green)
                            .foregroundColor(.white)
                            .clipShape(Capsule())
                    }
                    
                    if !user.isActive {
                        Text("Inactive")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .clipShape(Capsule())
                    }
                }
            }
            
            Spacer()
        }
        .padding(.vertical, 4)
    }
}

struct AddUserView: View {
    @EnvironmentObject var userService: UserService
    @Environment(\.dismiss) private var dismiss
    
    @State private var name = ""
    @State private var email = ""
    @State private var age = ""
    @State private var showingError = false
    @State private var errorMessage = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section("User Information") {
                    TextField("Name", text: $name)
                    TextField("Email", text: $email)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                    TextField("Age", text: $age)
                        .keyboardType(.numberPad)
                }
            }
            .navigationTitle("Add User")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveUser()
                    }
                    .disabled(!isValidInput)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private var isValidInput: Bool {
        !name.isEmpty && !email.isEmpty && Int(age) != nil
    }
    
    private func saveUser() {
        guard let ageInt = Int(age) else {
            errorMessage = "Please enter a valid age"
            showingError = true
            return
        }
        
        Task {
            await userService.createUser(name: name, email: email, age: ageInt)
            dismiss()
        }
    }
}

struct TaskListView: View {
    @EnvironmentObject var taskService: TaskService
    @State private var showingAddTask = false
    @State private var selectedStatus: Task.Status?
    
    var body: some View {
        NavigationView {
            List {
                ForEach(filteredTasks) { task in
                    TaskRowView(task: task)
                        .environmentObject(taskService)
                }
                .onDelete(perform: deleteTasks)
            }
            .navigationTitle("Tasks")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Menu("Filter") {
                        Button("All") {
                            selectedStatus = nil
                        }
                        
                        ForEach(Task.Status.allCases, id: \.self) { status in
                            Button(status.displayName) {
                                selectedStatus = status
                            }
                        }
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Add") {
                        showingAddTask = true
                    }
                }
            }
            .sheet(isPresented: $showingAddTask) {
                AddTaskView()
                    .environmentObject(taskService)
            }
            .refreshable {
                await taskService.loadTasks()
            }
        }
    }
    
    private var filteredTasks: [Task] {
        if let status = selectedStatus {
            return taskService.getTasksByStatus(status)
        }
        return taskService.tasks
    }
    
    private func deleteTasks(offsets: IndexSet) {
        Task {
            for index in offsets {
                await taskService.deleteTask(filteredTasks[index])
            }
        }
    }
}

struct TaskRowView: View {
    let task: Task
    @EnvironmentObject var taskService: TaskService
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(task.title)
                    .font(.headline)
                
                Spacer()
                
                priorityIndicator
                
                if task.isOverdue() {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                }
            }
            
            if !task.description.isEmpty {
                Text(task.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
            
            HStack {
                statusPicker
                
                Spacer()
                
                if let dueDate = task.dueDate {
                    Text("Due: \(dueDate, style: .date)")
                        .font(.caption)
                        .foregroundColor(task.isOverdue() ? .red : .secondary)
                }
            }
            
            if !task.tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(task.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(Color.blue.opacity(0.2))
                                .clipShape(Capsule())
                        }
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
    
    private var priorityIndicator: some View {
        Circle()
            .fill(priorityColor)
            .frame(width: 12, height: 12)
    }
    
    private var priorityColor: Color {
        switch task.priority {
        case .low: return .blue
        case .medium: return .yellow
        case .high: return .orange
        case .urgent: return .red
        }
    }
    
    private var statusPicker: some View {
        Menu(task.status.displayName) {
            ForEach(Task.Status.allCases, id: \.self) { status in
                Button(status.displayName) {
                    Task {
                        await taskService.updateTaskStatus(task, newStatus: status)
                    }
                }
            }
        }
        .font(.caption)
        .foregroundColor(.blue)
    }
}

struct AddTaskView: View {
    @EnvironmentObject var taskService: TaskService
    @Environment(\.dismiss) private var dismiss
    
    @State private var title = ""
    @State private var description = ""
    @State private var priority: Task.Priority = .medium
    @State private var dueDate = Date()
    @State private var hasDueDate = false
    
    var body: some View {
        NavigationView {
            Form {
                Section("Task Details") {
                    TextField("Title", text: $title)
                    TextField("Description", text: $description, axis: .vertical)
                        .lineLimit(3...6)
                }
                
                Section("Priority") {
                    Picker("Priority", selection: $priority) {
                        ForEach(Task.Priority.allCases, id: \.self) { priority in
                            Text(priority.rawValue.capitalized)
                                .tag(priority)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                
                Section("Due Date") {
                    Toggle("Set due date", isOn: $hasDueDate)
                    
                    if hasDueDate {
                        DatePicker("Due date", selection: $dueDate, displayedComponents: [.date])
                    }
                }
            }
            .navigationTitle("Add Task")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveTask()
                    }
                    .disabled(title.isEmpty)
                }
            }
        }
    }
    
    private func saveTask() {
        Task {
            await taskService.createTask(
                title: title,
                description: description,
                priority: priority
            )
            dismiss()
        }
    }
}

struct AnalyticsView: View {
    @EnvironmentObject var taskService: TaskService
    
    private var taskStats: (total: Int, completed: Int, overdue: Int, byPriority: [Task.Priority: Int]) {
        let tasks = taskService.tasks
        let completed = tasks.filter { $0.status == .done }.count
        let overdue = taskService.getOverdueTasks().count
        
        var byPriority: [Task.Priority: Int] = [:]
        for priority in Task.Priority.allCases {
            byPriority[priority] = tasks.filter { $0.priority == priority }.count
        }
        
        return (tasks.count, completed, overdue, byPriority)
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Summary Cards
                    HStack(spacing: 15) {
                        StatCard(title: "Total Tasks", value: "\(taskStats.total)", color: .blue)
                        StatCard(title: "Completed", value: "\(taskStats.completed)", color: .green)
                        StatCard(title: "Overdue", value: "\(taskStats.overdue)", color: .red)
                    }
                    
                    // Priority Distribution
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Priority Distribution")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        ForEach(Task.Priority.allCases, id: \.self) { priority in
                            PriorityBar(
                                priority: priority,
                                count: taskStats.byPriority[priority] ?? 0,
                                total: taskStats.total
                            )
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
                    
                    // Status Overview
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Status Overview")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        ForEach(Task.Status.allCases, id: \.self) { status in
                            let count = taskService.getTasksByStatus(status).count
                            StatusBar(status: status, count: count, total: taskStats.total)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }
            .navigationTitle("Analytics")
            .refreshable {
                await taskService.loadTasks()
            }
        }
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Text(value)
                .font(.title.bold())
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(radius: 2)
    }
}

struct PriorityBar: View {
    let priority: Task.Priority
    let count: Int
    let total: Int
    
    private var percentage: Double {
        total > 0 ? Double(count) / Double(total) : 0
    }
    
    var body: some View {
        HStack {
            Text(priority.rawValue.capitalized)
                .font(.subheadline)
                .frame(width: 80, alignment: .leading)
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))
                        .frame(height: 8)
                    
                    Rectangle()
                        .fill(priorityColor)
                        .frame(width: geometry.size.width * percentage, height: 8)
                }
                .clipShape(Capsule())
            }
            .frame(height: 8)
            
            Text("\(count)")
                .font(.subheadline)
                .monospacedDigit()
                .frame(width: 30, alignment: .trailing)
        }
        .padding(.horizontal)
    }
    
    private var priorityColor: Color {
        switch priority {
        case .low: return .blue
        case .medium: return .yellow
        case .high: return .orange
        case .urgent: return .red
        }
    }
}

struct StatusBar: View {
    let status: Task.Status
    let count: Int
    let total: Int
    
    private var percentage: Double {
        total > 0 ? Double(count) / Double(total) : 0
    }
    
    var body: some View {
        HStack {
            Text(status.displayName)
                .font(.subheadline)
                .frame(width: 100, alignment: .leading)
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))
                        .frame(height: 8)
                    
                    Rectangle()
                        .fill(statusColor)
                        .frame(width: geometry.size.width * percentage, height: 8)
                }
                .clipShape(Capsule())
            }
            .frame(height: 8)
            
            Text("\(count)")
                .font(.subheadline)
                .monospacedDigit()
                .frame(width: 30, alignment: .trailing)
        }
        .padding(.horizontal)
    }
    
    private var statusColor: Color {
        switch status {
        case .todo: return .gray
        case .inProgress: return .blue
        case .review: return .orange
        case .done: return .green
        }
    }
}

// ============ Cryptography and Security ============

class SecurityManager {
    
    static func generateSecurePassword(length: Int = 16) -> String {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        var password = ""
        
        for _ in 0..<length {
            let randomIndex = Int.random(in: 0..<chars.count)
            let char = chars[chars.index(chars.startIndex, offsetBy: randomIndex)]
            password.append(char)
        }
        
        return password
    }
    
    static func hashPassword(_ password: String) -> String {
        let data = password.data(using: .utf8)!
        let hashed = SHA256.hash(data: data)
        return hashed.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    static func encryptData(_ data: Data, key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined!
    }
    
    static func decryptData(_ encryptedData: Data, key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
        return try AES.GCM.open(sealedBox, using: key)
    }
    
    static func generateSymmetricKey() -> SymmetricKey {
        return SymmetricKey(size: .bits256)
    }
}

// ============ Functional Programming Utilities ============

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
    
    func asyncMap<T>(_ transform: @escaping (Element) async throws -> T) async rethrows -> [T] {
        var results: [T] = []
        results.reserveCapacity(count)
        
        for element in self {
            try await results.append(transform(element))
        }
        
        return results
    }
    
    func asyncCompactMap<T>(_ transform: @escaping (Element) async throws -> T?) async rethrows -> [T] {
        var results: [T] = []
        
        for element in self {
            if let result = try await transform(element) {
                results.append(result)
            }
        }
        
        return results
    }
}

extension Sequence {
    func asyncForEach(_ operation: @escaping (Element) async throws -> Void) async rethrows {
        for element in self {
            try await operation(element)
        }
    }
}

// ============ Demo and Testing ============

@main
struct TaskManagerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// For testing purposes (would normally be in test files)
class DemoRunner {
    static func runDemo() async {
        print("=== Swift Comprehensive Examples Demo ===\n")
        
        // Test repositories
        let userRepo = UserRepository()
        let taskRepo = TaskRepository()
        
        // Create sample users
        let user1 = User(name: "Alice Johnson", email: "alice@example.com", age: 28)
        let user2 = User(name: "Bob Smith", email: "bob@example.com", age: 35)
        
        do {
            try await userRepo.save(user1)
            try await userRepo.save(user2)
            
            let users = try await userRepo.loadAll()
            print("Created \(users.count) users")
            
            // Create sample tasks
            let task1 = Task(title: "Implement user authentication", 
                           description: "Add secure login system", 
                           priority: .high)
            let task2 = Task(title: "Design UI mockups", 
                           description: "Create wireframes for new features", 
                           priority: .medium)
            
            try await taskRepo.save(task1)
            try await taskRepo.save(task2)
            
            let tasks = try await taskRepo.loadAll()
            print("Created \(tasks.count) tasks")
            
            // Test security features
            let password = SecurityManager.generateSecurePassword()
            let hashedPassword = SecurityManager.hashPassword(password)
            print("Generated secure password hash: \(hashedPassword.prefix(16))...")
            
            let key = SecurityManager.generateSymmetricKey()
            let testData = "Sensitive information".data(using: .utf8)!
            let encrypted = try SecurityManager.encryptData(testData, key: key)
            let decrypted = try SecurityManager.decryptData(encrypted, key: key)
            let decryptedString = String(data: decrypted, encoding: .utf8)!
            print("Encryption/Decryption test: \(decryptedString)")
            
        } catch {
            print("Demo error: \(error)")
        }
        
        print("\n=== Swift Features Demonstrated ===")
        print("- Protocol-oriented programming")
        print("- Async/await concurrency")
        print("- Actor-based thread safety")
        print("- SwiftUI declarative UI")
        print("- Combine reactive programming")
        print("- Generic repositories and services")
        print("- Result builders and DSLs")
        print("- Property wrappers (@Published, @State)")
        print("- Error handling with throwing functions")
        print("- Cryptography with CryptoKit")
        print("- Network programming with URLSession")
        print("- File system operations")
        print("- Functional programming utilities")
        print("- Modern Swift features (actors, async sequences)")
    }
}