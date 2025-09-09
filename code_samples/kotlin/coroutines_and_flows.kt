// Comprehensive Kotlin Android and Coroutines Development Examples
// Demonstrates modern Kotlin features, coroutines, and Android-specific patterns

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.channels.*
import java.time.*
import java.util.concurrent.atomic.AtomicLong
import kotlin.reflect.full.*
import kotlin.random.Random

// ============ Sealed Classes and ADTs ============

/**
 * Sealed class for representing different types of UI states
 */
sealed class UiState<out T> {
    object Loading : UiState<Nothing>()
    object Empty : UiState<Nothing>()
    data class Success<T>(val data: T) : UiState<T>()
    data class Error(val exception: Throwable, val message: String = exception.message ?: "Unknown error") : UiState<Nothing>()
    
    inline fun <R> fold(
        onLoading: () -> R,
        onEmpty: () -> R,
        onSuccess: (T) -> R,
        onError: (Throwable, String) -> R
    ): R = when (this) {
        is Loading -> onLoading()
        is Empty -> onEmpty()
        is Success -> onSuccess(data)
        is Error -> onError(exception, message)
    }
    
    fun isLoading(): Boolean = this is Loading
    fun isSuccess(): Boolean = this is Success
    fun isError(): Boolean = this is Error
    
    fun getOrNull(): T? = when (this) {
        is Success -> data
        else -> null
    }
}

/**
 * Sealed class for representing different types of network responses
 */
sealed class NetworkResult<out T> {
    data class Success<T>(val data: T) : NetworkResult<T>()
    data class Error(val code: Int, val message: String) : NetworkResult<Nothing>()
    data class Exception(val exception: Throwable) : NetworkResult<Nothing>()
    
    companion object {
        fun <T> success(data: T): NetworkResult<T> = Success(data)
        fun error(code: Int, message: String): NetworkResult<Nothing> = Error(code, message)
        fun exception(exception: Throwable): NetworkResult<Nothing> = Exception(exception)
    }
    
    inline fun <R> map(transform: (T) -> R): NetworkResult<R> = when (this) {
        is Success -> Success(transform(data))
        is Error -> this
        is Exception -> this
    }
    
    inline fun <R> flatMap(transform: (T) -> NetworkResult<R>): NetworkResult<R> = when (this) {
        is Success -> transform(data)
        is Error -> this
        is Exception -> this
    }
    
    fun getOrThrow(): T = when (this) {
        is Success -> data
        is Error -> throw RuntimeException("Network error: $code - $message")
        is Exception -> throw exception
    }
    
    fun getOrNull(): T? = when (this) {
        is Success -> data
        else -> null
    }
}

/**
 * Event wrapper for UI events that should only be consumed once
 */
class Event<out T>(private val content: T) {
    private var hasBeenHandled = false
    
    fun getContentIfNotHandled(): T? {
        return if (hasBeenHandled) {
            null
        } else {
            hasBeenHandled = true
            content
        }
    }
    
    fun peekContent(): T = content
}

// ============ Data Classes with Business Logic ============

/**
 * Value class for type-safe IDs
 */
@JvmInline
value class UserId(val value: Long) {
    init {
        require(value > 0) { "User ID must be positive" }
    }
    
    companion object {
        fun generate(): UserId = UserId(Random.nextLong(1, Long.MAX_VALUE))
    }
}

@JvmInline
value class Email(val value: String) {
    init {
        require(value.isNotBlank()) { "Email cannot be blank" }
        require(value.contains("@")) { "Email must contain @" }
        require(value.contains(".")) { "Email must contain domain" }
    }
    
    val domain: String get() = value.substringAfter("@")
    val localPart: String get() = value.substringBefore("@")
}

/**
 * Comprehensive user data class with business logic
 */
data class User(
    val id: UserId,
    val firstName: String,
    val lastName: String,
    val email: Email,
    val role: UserRole,
    val isActive: Boolean = true,
    val createdAt: Instant = Instant.now(),
    val updatedAt: Instant = Instant.now(),
    val lastLoginAt: Instant? = null,
    val preferences: UserPreferences = UserPreferences(),
    val profile: UserProfile = UserProfile()
) {
    
    val fullName: String get() = "$firstName $lastName"
    val displayName: String get() = profile.displayName ?: fullName
    val age: Int? get() = profile.dateOfBirth?.let { 
        Period.between(it, LocalDate.now()).years 
    }
    val isAdult: Boolean get() = age?.let { it >= 18 } ?: false
    
    fun updateProfile(updates: UserProfile.() -> UserProfile): User {
        return copy(
            profile = profile.updates(),
            updatedAt = Instant.now()
        )
    }
    
    fun updatePreferences(updates: UserPreferences.() -> UserPreferences): User {
        return copy(
            preferences = preferences.updates(),
            updatedAt = Instant.now()
        )
    }
    
    fun recordLogin(): User = copy(
        lastLoginAt = Instant.now(),
        updatedAt = Instant.now()
    )
    
    fun deactivate(): User = copy(
        isActive = false,
        updatedAt = Instant.now()
    )
    
    fun activate(): User = copy(
        isActive = true,
        updatedAt = Instant.now()
    )
    
    fun hasPermission(permission: String): Boolean {
        return isActive && role.hasPermission(permission)
    }
    
    companion object {
        fun create(
            firstName: String,
            lastName: String,
            email: String,
            role: UserRole = UserRole.USER
        ): User {
            return User(
                id = UserId.generate(),
                firstName = firstName.trim(),
                lastName = lastName.trim(),
                email = Email(email.lowercase().trim()),
                role = role
            )
        }
    }
}

data class UserPreferences(
    val theme: Theme = Theme.SYSTEM,
    val language: String = "en",
    val timezone: String = "UTC",
    val notifications: NotificationSettings = NotificationSettings(),
    val privacy: PrivacySettings = PrivacySettings()
) {
    
    enum class Theme(val displayName: String) {
        LIGHT("Light"),
        DARK("Dark"),
        SYSTEM("System");
        
        companion object {
            fun fromString(value: String): Theme = values().find { 
                it.name.equals(value, ignoreCase = true) 
            } ?: SYSTEM
        }
    }
}

data class NotificationSettings(
    val email: Boolean = true,
    val push: Boolean = true,
    val sms: Boolean = false,
    val marketing: Boolean = false,
    val security: Boolean = true
) {
    fun enableAll(): NotificationSettings = copy(
        email = true,
        push = true,
        sms = true,
        marketing = true,
        security = true
    )
    
    fun disableAll(): NotificationSettings = copy(
        email = false,
        push = false,
        sms = false,
        marketing = false,
        security = false
    )
}

data class PrivacySettings(
    val profileVisible: Boolean = true,
    val onlineStatusVisible: Boolean = true,
    val analyticsEnabled: Boolean = false,
    val locationSharingEnabled: Boolean = false,
    val dataSharingEnabled: Boolean = false
)

data class UserProfile(
    val displayName: String? = null,
    val bio: String? = null,
    val avatarUrl: String? = null,
    val dateOfBirth: LocalDate? = null,
    val location: String? = null,
    val website: String? = null,
    val socialLinks: Map<String, String> = emptyMap(),
    val interests: List<String> = emptyList(),
    val skills: List<String> = emptyList()
) {
    fun addSocialLink(platform: String, url: String): UserProfile {
        return copy(socialLinks = socialLinks + (platform to url))
    }
    
    fun removeSocialLink(platform: String): UserProfile {
        return copy(socialLinks = socialLinks - platform)
    }
    
    fun addInterest(interest: String): UserProfile {
        val normalizedInterest = interest.trim().lowercase()
        return if (normalizedInterest !in interests.map { it.lowercase() }) {
            copy(interests = interests + interest.trim())
        } else this
    }
    
    fun removeInterest(interest: String): UserProfile {
        return copy(interests = interests.filter { 
            !it.equals(interest, ignoreCase = true) 
        })
    }
    
    fun addSkill(skill: String): UserProfile {
        val normalizedSkill = skill.trim().lowercase()
        return if (normalizedSkill !in skills.map { it.lowercase() }) {
            copy(skills = skills + skill.trim())
        } else this
    }
}

enum class UserRole(val permissions: Set<String>) {
    GUEST(setOf("read:public")),
    USER(setOf("read:public", "read:own", "write:own")),
    MODERATOR(setOf("read:public", "read:own", "write:own", "moderate:content", "read:reports")),
    ADMIN(setOf("read:all", "write:all", "delete:all", "manage:users", "manage:system")),
    SUPER_ADMIN(setOf("*"));
    
    fun hasPermission(permission: String): Boolean {
        return permissions.contains("*") || permissions.contains(permission)
    }
    
    fun canManageUser(targetRole: UserRole): Boolean {
        return when (this) {
            SUPER_ADMIN -> true
            ADMIN -> targetRole != SUPER_ADMIN
            MODERATOR -> targetRole in setOf(GUEST, USER)
            else -> false
        }
    }
}

// ============ Repository Pattern with Flows ============

interface Repository<T, ID> {
    suspend fun save(entity: T): T
    suspend fun findById(id: ID): T?
    suspend fun findAll(): List<T>
    suspend fun delete(id: ID): Boolean
    suspend fun count(): Long
    
    fun observeAll(): Flow<List<T>>
    fun observeById(id: ID): Flow<T?>
}

class InMemoryUserRepository : Repository<User, UserId> {
    private val users = mutableMapOf<UserId, User>()
    private val _usersFlow = MutableSharedFlow<List<User>>(replay = 1)
    
    init {
        _usersFlow.tryEmit(emptyList())
    }
    
    override suspend fun save(entity: User): User {
        users[entity.id] = entity
        _usersFlow.emit(users.values.toList())
        return entity
    }
    
    override suspend fun findById(id: UserId): User? {
        return users[id]
    }
    
    override suspend fun findAll(): List<User> {
        return users.values.toList()
    }
    
    override suspend fun delete(id: UserId): Boolean {
        val removed = users.remove(id) != null
        if (removed) {
            _usersFlow.emit(users.values.toList())
        }
        return removed
    }
    
    override suspend fun count(): Long {
        return users.size.toLong()
    }
    
    override fun observeAll(): Flow<List<User>> = _usersFlow.asSharedFlow()
    
    override fun observeById(id: UserId): Flow<User?> {
        return _usersFlow.map { users -> users.find { it.id == id } }
    }
    
    suspend fun findByEmail(email: Email): User? {
        return users.values.find { it.email == email }
    }
    
    suspend fun findByRole(role: UserRole): List<User> {
        return users.values.filter { it.role == role }
    }
    
    suspend fun findActiveUsers(): List<User> {
        return users.values.filter { it.isActive }
    }
    
    suspend fun searchUsers(query: String): List<User> {
        val lowercaseQuery = query.lowercase()
        return users.values.filter { user ->
            user.fullName.lowercase().contains(lowercaseQuery) ||
            user.email.value.lowercase().contains(lowercaseQuery) ||
            user.profile.bio?.lowercase()?.contains(lowercaseQuery) == true
        }
    }
}

// ============ Use Cases / Interactors ============

sealed class UserResult {
    data class Success(val user: User) : UserResult()
    data class Failure(val error: String) : UserResult()
}

class CreateUserUseCase(
    private val userRepository: InMemoryUserRepository,
    private val emailValidator: EmailValidator = EmailValidator()
) {
    
    suspend operator fun invoke(
        firstName: String,
        lastName: String,
        email: String,
        role: UserRole = UserRole.USER
    ): UserResult {
        return try {
            // Validate inputs
            if (firstName.isBlank()) {
                return UserResult.Failure("First name cannot be blank")
            }
            
            if (lastName.isBlank()) {
                return UserResult.Failure("Last name cannot be blank")
            }
            
            if (!emailValidator.isValid(email)) {
                return UserResult.Failure("Invalid email format")
            }
            
            // Check if email already exists
            val emailObj = Email(email.lowercase().trim())
            val existingUser = userRepository.findByEmail(emailObj)
            if (existingUser != null) {
                return UserResult.Failure("User with this email already exists")
            }
            
            // Create and save user
            val user = User.create(firstName, lastName, email, role)
            val savedUser = userRepository.save(user)
            
            UserResult.Success(savedUser)
        } catch (e: Exception) {
            UserResult.Failure("Failed to create user: ${e.message}")
        }
    }
}

class GetUserUseCase(private val userRepository: InMemoryUserRepository) {
    
    suspend operator fun invoke(userId: UserId): UserResult {
        return try {
            val user = userRepository.findById(userId)
            if (user != null) {
                UserResult.Success(user)
            } else {
                UserResult.Failure("User not found")
            }
        } catch (e: Exception) {
            UserResult.Failure("Failed to get user: ${e.message}")
        }
    }
}

class UpdateUserProfileUseCase(
    private val userRepository: InMemoryUserRepository
) {
    
    suspend operator fun invoke(
        userId: UserId,
        profileUpdates: UserProfile.() -> UserProfile
    ): UserResult {
        return try {
            val user = userRepository.findById(userId)
                ?: return UserResult.Failure("User not found")
            
            val updatedUser = user.updateProfile(profileUpdates)
            val savedUser = userRepository.save(updatedUser)
            
            UserResult.Success(savedUser)
        } catch (e: Exception) {
            UserResult.Failure("Failed to update profile: ${e.message}")
        }
    }
}

class SearchUsersUseCase(private val userRepository: InMemoryUserRepository) {
    
    suspend operator fun invoke(query: String): List<User> {
        return if (query.isBlank()) {
            userRepository.findActiveUsers()
        } else {
            userRepository.searchUsers(query)
        }
    }
}

// ============ View Models with StateFlow ============

class UserListViewModel(
    private val searchUsersUseCase: SearchUsersUseCase,
    private val userRepository: InMemoryUserRepository
) {
    
    private val _uiState = MutableStateFlow<UiState<List<User>>>(UiState.Loading)
    val uiState: StateFlow<UiState<List<User>>> = _uiState.asStateFlow()
    
    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()
    
    private val _events = MutableSharedFlow<Event<UserListEvent>>()
    val events: SharedFlow<Event<UserListEvent>> = _events.asSharedFlow()
    
    private val coroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    init {
        observeUsers()
        setupSearch()
    }
    
    private fun observeUsers() {
        coroutineScope.launch {
            userRepository.observeAll()
                .catch { e ->
                    _uiState.value = UiState.Error(e)
                }
                .collect { users ->
                    _uiState.value = if (users.isEmpty()) {
                        UiState.Empty
                    } else {
                        UiState.Success(users)
                    }
                }
        }
    }
    
    private fun setupSearch() {
        coroutineScope.launch {
            _searchQuery
                .debounce(300) // Wait for 300ms after user stops typing
                .distinctUntilChanged()
                .collect { query ->
                    performSearch(query)
                }
        }
    }
    
    fun updateSearchQuery(query: String) {
        _searchQuery.value = query
    }
    
    private suspend fun performSearch(query: String) {
        try {
            _uiState.value = UiState.Loading
            val users = searchUsersUseCase(query)
            _uiState.value = if (users.isEmpty()) {
                UiState.Empty
            } else {
                UiState.Success(users)
            }
        } catch (e: Exception) {
            _uiState.value = UiState.Error(e)
        }
    }
    
    fun refresh() {
        coroutineScope.launch {
            performSearch(_searchQuery.value)
        }
    }
    
    fun onUserClicked(user: User) {
        coroutineScope.launch {
            _events.emit(Event(UserListEvent.NavigateToUserDetail(user.id)))
        }
    }
    
    fun onCreateUserClicked() {
        coroutineScope.launch {
            _events.emit(Event(UserListEvent.NavigateToCreateUser))
        }
    }
    
    fun clear() {
        coroutineScope.cancel()
    }
    
    sealed class UserListEvent {
        data class NavigateToUserDetail(val userId: UserId) : UserListEvent()
        object NavigateToCreateUser : UserListEvent()
    }
}

class UserDetailViewModel(
    private val userId: UserId,
    private val getUserUseCase: GetUserUseCase,
    private val updateUserProfileUseCase: UpdateUserProfileUseCase,
    private val userRepository: InMemoryUserRepository
) {
    
    private val _uiState = MutableStateFlow<UiState<User>>(UiState.Loading)
    val uiState: StateFlow<UiState<User>> = _uiState.asStateFlow()
    
    private val _isUpdating = MutableStateFlow(false)
    val isUpdating: StateFlow<Boolean> = _isUpdating.asStateFlow()
    
    private val _events = MutableSharedFlow<Event<UserDetailEvent>>()
    val events: SharedFlow<Event<UserDetailEvent>> = _events.asSharedFlow()
    
    private val coroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    init {
        loadUser()
        observeUser()
    }
    
    private fun loadUser() {
        coroutineScope.launch {
            try {
                when (val result = getUserUseCase(userId)) {
                    is UserResult.Success -> _uiState.value = UiState.Success(result.user)
                    is UserResult.Failure -> _uiState.value = UiState.Error(
                        RuntimeException(result.error), result.error
                    )
                }
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e)
            }
        }
    }
    
    private fun observeUser() {
        coroutineScope.launch {
            userRepository.observeById(userId)
                .filterNotNull()
                .catch { e ->
                    _uiState.value = UiState.Error(e)
                }
                .collect { user ->
                    _uiState.value = UiState.Success(user)
                }
        }
    }
    
    fun updateBio(bio: String) {
        updateProfile { copy(bio = bio.takeIf { it.isNotBlank() }) }
    }
    
    fun updateDisplayName(displayName: String) {
        updateProfile { copy(displayName = displayName.takeIf { it.isNotBlank() }) }
    }
    
    fun addSkill(skill: String) {
        updateProfile { addSkill(skill) }
    }
    
    fun removeSkill(skill: String) {
        updateProfile { copy(skills = skills - skill) }
    }
    
    fun addInterest(interest: String) {
        updateProfile { addInterest(interest) }
    }
    
    fun removeInterest(interest: String) {
        updateProfile { removeInterest(interest) }
    }
    
    private fun updateProfile(update: UserProfile.() -> UserProfile) {
        coroutineScope.launch {
            try {
                _isUpdating.value = true
                when (val result = updateUserProfileUseCase(userId, update)) {
                    is UserResult.Success -> {
                        _events.emit(Event(UserDetailEvent.ProfileUpdated))
                    }
                    is UserResult.Failure -> {
                        _events.emit(Event(UserDetailEvent.ShowError(result.error)))
                    }
                }
            } catch (e: Exception) {
                _events.emit(Event(UserDetailEvent.ShowError(e.message ?: "Unknown error")))
            } finally {
                _isUpdating.value = false
            }
        }
    }
    
    fun clear() {
        coroutineScope.cancel()
    }
    
    sealed class UserDetailEvent {
        object ProfileUpdated : UserDetailEvent()
        data class ShowError(val message: String) : UserDetailEvent()
    }
}

// ============ Networking with Flows ============

interface ApiService {
    suspend fun getUsers(): NetworkResult<List<User>>
    suspend fun getUserById(id: UserId): NetworkResult<User>
    suspend fun createUser(user: User): NetworkResult<User>
    suspend fun updateUser(user: User): NetworkResult<User>
    suspend fun deleteUser(id: UserId): NetworkResult<Unit>
}

class MockApiService : ApiService {
    
    private val networkDelay = 1000L // Simulate network delay
    
    override suspend fun getUsers(): NetworkResult<List<User>> {
        delay(networkDelay)
        return NetworkResult.success(generateMockUsers())
    }
    
    override suspend fun getUserById(id: UserId): NetworkResult<User> {
        delay(networkDelay)
        return if (Random.nextBoolean()) {
            NetworkResult.success(generateMockUsers().first())
        } else {
            NetworkResult.error(404, "User not found")
        }
    }
    
    override suspend fun createUser(user: User): NetworkResult<User> {
        delay(networkDelay)
        return if (Random.nextDouble() > 0.1) { // 90% success rate
            NetworkResult.success(user)
        } else {
            NetworkResult.error(400, "Email already exists")
        }
    }
    
    override suspend fun updateUser(user: User): NetworkResult<User> {
        delay(networkDelay)
        return NetworkResult.success(user.copy(updatedAt = Instant.now()))
    }
    
    override suspend fun deleteUser(id: UserId): NetworkResult<Unit> {
        delay(networkDelay)
        return NetworkResult.success(Unit)
    }
    
    private fun generateMockUsers(): List<User> {
        return listOf(
            User.create("John", "Doe", "john.doe@example.com", UserRole.USER),
            User.create("Jane", "Smith", "jane.smith@example.com", UserRole.MODERATOR),
            User.create("Admin", "User", "admin@example.com", UserRole.ADMIN)
        )
    }
}

// ============ Reactive Data Layer ============

class UserRepository(
    private val apiService: ApiService,
    private val localRepository: InMemoryUserRepository
) {
    
    fun getUsers(forceRefresh: Boolean = false): Flow<UiState<List<User>>> = flow {
        emit(UiState.Loading)
        
        try {
            // Emit cached data first if available and not forcing refresh
            if (!forceRefresh) {
                val cachedUsers = localRepository.findAll()
                if (cachedUsers.isNotEmpty()) {
                    emit(UiState.Success(cachedUsers))
                }
            }
            
            // Fetch from network
            when (val networkResult = apiService.getUsers()) {
                is NetworkResult.Success -> {
                    // Save to local storage
                    networkResult.data.forEach { user ->
                        localRepository.save(user)
                    }
                    emit(UiState.Success(networkResult.data))
                }
                is NetworkResult.Error -> {
                    emit(UiState.Error(RuntimeException("Network error: ${networkResult.code} - ${networkResult.message}")))
                }
                is NetworkResult.Exception -> {
                    emit(UiState.Error(networkResult.exception))
                }
            }
        } catch (e: Exception) {
            emit(UiState.Error(e))
        }
    }.catch { e ->
        emit(UiState.Error(e))
    }
    
    fun getUserById(id: UserId): Flow<UiState<User>> = flow {
        emit(UiState.Loading)
        
        try {
            // Try to get from local storage first
            val cachedUser = localRepository.findById(id)
            if (cachedUser != null) {
                emit(UiState.Success(cachedUser))
            }
            
            // Fetch from network
            when (val networkResult = apiService.getUserById(id)) {
                is NetworkResult.Success -> {
                    localRepository.save(networkResult.data)
                    emit(UiState.Success(networkResult.data))
                }
                is NetworkResult.Error -> {
                    if (cachedUser == null) {
                        emit(UiState.Error(RuntimeException("User not found")))
                    }
                }
                is NetworkResult.Exception -> {
                    if (cachedUser == null) {
                        emit(UiState.Error(networkResult.exception))
                    }
                }
            }
        } catch (e: Exception) {
            emit(UiState.Error(e))
        }
    }
}

// ============ Coroutines Utilities ============

/**
 * Retry mechanism with exponential backoff
 */
suspend fun <T> retryWithBackoff(
    times: Int = 3,
    initialDelay: Long = 100,
    maxDelay: Long = 1000,
    factor: Double = 2.0,
    block: suspend () -> T
): T {
    var currentDelay = initialDelay
    repeat(times - 1) {
        try {
            return block()
        } catch (e: Exception) {
            delay(currentDelay)
            currentDelay = (currentDelay * factor).toLong().coerceAtMost(maxDelay)
        }
    }
    return block() // Last attempt
}

/**
 * Safe API call wrapper
 */
suspend fun <T> safeApiCall(
    apiCall: suspend () -> T
): NetworkResult<T> {
    return try {
        NetworkResult.success(apiCall())
    } catch (e: Exception) {
        NetworkResult.exception(e)
    }
}

/**
 * Combine multiple flows into a single state
 */
fun <T1, T2, R> Flow<T1>.combineLatest(
    other: Flow<T2>,
    transform: suspend (T1, T2) -> R
): Flow<R> = combine(this, other, transform)

/**
 * Transform flow with loading states
 */
fun <T, R> Flow<T>.mapToUiState(
    transform: suspend (T) -> R
): Flow<UiState<R>> = map { data ->
    try {
        UiState.Success(transform(data))
    } catch (e: Exception) {
        UiState.Error(e)
    }
}.catch { e ->
    emit(UiState.Error(e))
}

// ============ Validation ============

class EmailValidator {
    private val emailRegex = Regex("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$")
    
    fun isValid(email: String): Boolean {
        return email.isNotBlank() && emailRegex.matches(email)
    }
}

class PasswordValidator {
    fun validate(password: String): List<String> {
        val errors = mutableListOf<String>()
        
        if (password.length < 8) {
            errors.add("Password must be at least 8 characters long")
        }
        
        if (!password.any { it.isUpperCase() }) {
            errors.add("Password must contain at least one uppercase letter")
        }
        
        if (!password.any { it.isLowerCase() }) {
            errors.add("Password must contain at least one lowercase letter")
        }
        
        if (!password.any { it.isDigit() }) {
            errors.add("Password must contain at least one digit")
        }
        
        if (!password.any { it in "!@#$%^&*()_+-=[]{}|;:,.<>?" }) {
            errors.add("Password must contain at least one special character")
        }
        
        return errors
    }
    
    fun isValid(password: String): Boolean = validate(password).isEmpty()
}

// ============ Extension Functions ============

fun String.isValidEmail(): Boolean = EmailValidator().isValid(this)

fun String.toUserId(): UserId? = toLongOrNull()?.let { UserId(it) }

fun Instant.toFormattedString(): String {
    return LocalDateTime.ofInstant(this, ZoneId.systemDefault())
        .format(java.time.format.DateTimeFormatter.ofPattern("MMM dd, yyyy 'at' HH:mm"))
}

fun LocalDate.age(): Int = Period.between(this, LocalDate.now()).years

inline fun <T> List<T>.partitionIndexed(predicate: (index: Int, T) -> Boolean): Pair<List<T>, List<T>> {
    val first = mutableListOf<T>()
    val second = mutableListOf<T>()
    
    this.forEachIndexed { index, element ->
        if (predicate(index, element)) {
            first.add(element)
        } else {
            second.add(element)
        }
    }
    
    return Pair(first, second)
}

// ============ Demo Application ============

class DemoApplication {
    private val apiService = MockApiService()
    private val localRepository = InMemoryUserRepository()
    private val userRepository = UserRepository(apiService, localRepository)
    
    private val createUserUseCase = CreateUserUseCase(localRepository)
    private val getUserUseCase = GetUserUseCase(localRepository)
    private val updateUserProfileUseCase = UpdateUserProfileUseCase(localRepository)
    private val searchUsersUseCase = SearchUsersUseCase(localRepository)
    
    suspend fun runDemo() {
        println("=== Kotlin Comprehensive Examples Demo ===\n")
        
        demoCoroutines()
        demoDataClasses()
        demoSealedClasses()
        demoFlows()
        demoUseCases()
        demoViewModels()
        demoNetworking()
        demoExtensions()
        
        println("\n=== Kotlin Features Demonstrated ===")
        println("🚀 Coroutines and structured concurrency")
        println("🌊 Flow and reactive programming")
        println("🏗️  Data classes with business logic")
        println("🔒 Sealed classes and ADTs")
        println("💉 Dependency injection patterns")
        println("🎯 Use case / Clean architecture")
        println("📱 MVVM with ViewModels and StateFlow")
        println("🌐 Networking with error handling")
        println("✨ Extension functions and DSLs")
        println("🔧 Type-safe builders and DSLs")
        println("⚡ Value classes for performance")
        println("🛡️  Null safety and smart casting")
        println("🎨 Functional programming concepts")
        println("📊 State management patterns")
    }
    
    private suspend fun demoCoroutines() {
        println("=== Coroutines Demo ===")
        
        // Concurrent execution
        val startTime = System.currentTimeMillis()
        
        val deferred1 = GlobalScope.async {
            delay(1000)
            "Task 1 completed"
        }
        
        val deferred2 = GlobalScope.async {
            delay(800)
            "Task 2 completed"
        }
        
        val deferred3 = GlobalScope.async {
            delay(1200)
            "Task 3 completed"
        }
        
        val results = awaitAll(deferred1, deferred2, deferred3)
        val totalTime = System.currentTimeMillis() - startTime
        
        println("Concurrent execution results:")
        results.forEach { println("  - $it") }
        println("Total time: ${totalTime}ms (should be ~1200ms, not 3000ms)\n")
        
        // Retry with backoff
        try {
            val result = retryWithBackoff(times = 3) {
                if (Random.nextBoolean()) {
                    "Success on retry!"
                } else {
                    throw RuntimeException("Random failure")
                }
            }
            println("Retry result: $result")
        } catch (e: Exception) {
            println("All retries failed: ${e.message}")
        }
        
        println()
    }
    
    private suspend fun demoDataClasses() {
        println("=== Data Classes Demo ===")
        
        val user = User.create(
            firstName = "Alice",
            lastName = "Johnson",
            email = "alice@example.com",
            role = UserRole.USER
        )
        
        println("Created user: ${user.fullName}")
        println("User permissions: ${user.role.permissions}")
        println("Can manage users: ${user.hasPermission("manage:users")}")
        
        // Update profile using copy and extensions
        val updatedUser = user.updateProfile {
            copy(
                bio = "Software Developer passionate about Kotlin",
                interests = listOf("Kotlin", "Android", "Coroutines"),
                skills = listOf("Kotlin", "Java", "Android")
            )
        }
        
        println("Updated user bio: ${updatedUser.profile.bio}")
        println("User interests: ${updatedUser.profile.interests}")
        
        // Demonstrate value classes
        val userId = UserId.generate()
        println("Generated user ID: ${userId.value}")
        
        val email = Email("test@example.com")
        println("Email domain: ${email.domain}")
        
        println()
    }
    
    private suspend fun demoSealedClasses() {
        println("=== Sealed Classes Demo ===")
        
        val states = listOf(
            UiState.Loading,
            UiState.Success("Hello World"),
            UiState.Error(RuntimeException("Something went wrong")),
            UiState.Empty
        )
        
        states.forEach { state ->
            val message = state.fold(
                onLoading = { "Loading..." },
                onEmpty = { "No data available" },
                onSuccess = { data -> "Success: $data" },
                onError = { _, message -> "Error: $message" }
            )
            println("State: $message")
        }
        
        // Network result demo
        val networkResults = listOf(
            NetworkResult.success("Data loaded"),
            NetworkResult.error(404, "Not found"),
            NetworkResult.exception(RuntimeException("Network error"))
        )
        
        networkResults.forEach { result ->
            val mapped = result.map { data -> data.uppercase() }
            println("Network result: ${mapped.getOrNull() ?: "Failed"}")
        }
        
        println()
    }
    
    private suspend fun demoFlows() {
        println("=== Flows Demo ===")
        
        // Create some test users
        val users = listOf(
            User.create("John", "Doe", "john@example.com"),
            User.create("Jane", "Smith", "jane@example.com"),
            User.create("Bob", "Johnson", "bob@example.com")
        )
        
        users.forEach { localRepository.save(it) }
        
        // Observe users flow
        println("Observing users flow:")
        localRepository.observeAll()
            .take(1)
            .collect { userList ->
                println("  Current users: ${userList.map { it.fullName }}")
            }
        
        // Add another user and observe changes
        val newUser = User.create("Alice", "Brown", "alice@example.com")
        localRepository.save(newUser)
        
        localRepository.observeAll()
            .take(1)
            .collect { userList ->
                println("  After adding Alice: ${userList.map { it.fullName }}")
            }
        
        println()
    }
    
    private suspend fun demoUseCases() {
        println("=== Use Cases Demo ===")
        
        // Create user use case
        val createResult = createUserUseCase(
            firstName = "Charlie",
            lastName = "Wilson",
            email = "charlie@example.com",
            role = UserRole.MODERATOR
        )
        
        when (createResult) {
            is UserResult.Success -> {
                println("Created user: ${createResult.user.fullName}")
                
                // Get user use case
                val getResult = getUserUseCase(createResult.user.id)
                when (getResult) {
                    is UserResult.Success -> {
                        println("Retrieved user: ${getResult.user.fullName}")
                    }
                    is UserResult.Failure -> {
                        println("Failed to get user: ${getResult.error}")
                    }
                }
                
                // Update profile use case
                val updateResult = updateUserProfileUseCase(createResult.user.id) {
                    addSkill("Kotlin").addInterest("Android Development")
                }
                
                when (updateResult) {
                    is UserResult.Success -> {
                        println("Updated user skills: ${updateResult.user.profile.skills}")
                        println("Updated user interests: ${updateResult.user.profile.interests}")
                    }
                    is UserResult.Failure -> {
                        println("Failed to update: ${updateResult.error}")
                    }
                }
            }
            is UserResult.Failure -> {
                println("Failed to create user: ${createResult.error}")
            }
        }
        
        println()
    }
    
    private suspend fun demoViewModels() {
        println("=== ViewModels Demo ===")
        
        val viewModel = UserListViewModel(searchUsersUseCase, localRepository)
        
        // Observe UI state
        println("Initial UI state:")
        viewModel.uiState.take(1).collect { state ->
            when (state) {
                is UiState.Loading -> println("  Loading users...")
                is UiState.Success -> println("  Loaded ${state.data.size} users")
                is UiState.Error -> println("  Error: ${state.message}")
                is UiState.Empty -> println("  No users found")
            }
        }
        
        // Simulate search
        viewModel.updateSearchQuery("alice")
        delay(500) // Wait for debounce
        
        println("After searching for 'alice':")
        viewModel.uiState.take(1).collect { state ->
            when (state) {
                is UiState.Success -> {
                    println("  Found ${state.data.size} users matching 'alice'")
                    state.data.forEach { user ->
                        println("    - ${user.fullName}")
                    }
                }
                else -> println("  Search state: ${state::class.simpleName}")
            }
        }
        
        viewModel.clear()
        println()
    }
    
    private suspend fun demoNetworking() {
        println("=== Networking Demo ===")
        
        // Simulate API calls
        userRepository.getUsers(forceRefresh = true)
            .take(2) // Take loading and success states
            .collect { state ->
                when (state) {
                    is UiState.Loading -> println("  Loading users from API...")
                    is UiState.Success -> {
                        println("  Loaded ${state.data.size} users from API")
                        state.data.forEach { user ->
                            println("    - ${user.fullName} (${user.role})")
                        }
                    }
                    is UiState.Error -> println("  API Error: ${state.message}")
                    is UiState.Empty -> println("  No users from API")
                }
            }
        
        println()
    }
    
    private fun demoExtensions() {
        println("=== Extension Functions Demo ===")
        
        // Email validation
        val emails = listOf("valid@example.com", "invalid.email", "another@test.org")
        emails.forEach { email ->
            println("'$email' is valid: ${email.isValidEmail()}")
        }
        
        // Date formatting
        val now = Instant.now()
        println("Current time: ${now.toFormattedString()}")
        
        // Age calculation
        val birthDate = LocalDate.of(1990, 5, 15)
        println("Age for birth date $birthDate: ${birthDate.age()} years")
        
        // List partitioning
        val numbers = listOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        val (evens, odds) = numbers.partitionIndexed { index, _ -> index % 2 == 0 }
        println("Even indices: $evens")
        println("Odd indices: $odds")
        
        println()
    }
}

// ============ Main Function ============

suspend fun main() {
    val app = DemoApplication()
    app.runDemo()
}