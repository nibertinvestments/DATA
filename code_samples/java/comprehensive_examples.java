/**
 * Comprehensive Java Examples
 * Demonstrates OOP, generics, streams, concurrency, and modern Java patterns
 */

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.stream.*;
import java.util.regex.Pattern;
import java.lang.annotation.*;
import java.lang.reflect.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.*;
import javax.sql.DataSource;
import com.fasterxml.jackson.annotation.*;
import com.fasterxml.jackson.databind.*;

// ========== Domain Models ==========

/**
 * User entity with builder pattern and validation
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class User {
    @JsonProperty("id")
    private Long id;
    
    @JsonProperty("username")
    private String username;
    
    @JsonProperty("email")
    private String email;
    
    @JsonProperty("age")
    private Integer age;
    
    @JsonProperty("active")
    private boolean active = true;
    
    @JsonProperty("created_at")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createdAt;
    
    @JsonProperty("updated_at")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime updatedAt;
    
    @JsonProperty("roles")
    private Set<Role> roles = new HashSet<>();
    
    // Constructors
    public User() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public User(String username, String email) {
        this();
        this.username = username;
        this.email = email;
    }
    
    public User(Long id, String username, String email, Integer age) {
        this(username, email);
        this.id = id;
        this.age = age;
    }
    
    // Builder pattern
    public static class Builder {
        private User user = new User();
        
        public Builder id(Long id) {
            user.id = id;
            return this;
        }
        
        public Builder username(String username) {
            user.username = username;
            return this;
        }
        
        public Builder email(String email) {
            user.email = email;
            return this;
        }
        
        public Builder age(Integer age) {
            user.age = age;
            return this;
        }
        
        public Builder active(boolean active) {
            user.active = active;
            return this;
        }
        
        public Builder role(Role role) {
            user.roles.add(role);
            return this;
        }
        
        public User build() {
            user.validate();
            return user;
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    // Validation
    public void validate() {
        if (username == null || username.trim().isEmpty()) {
            throw new IllegalArgumentException("Username cannot be null or empty");
        }
        if (email == null || !isValidEmail(email)) {
            throw new IllegalArgumentException("Invalid email address");
        }
        if (age != null && (age < 0 || age > 150)) {
            throw new IllegalArgumentException("Age must be between 0 and 150");
        }
    }
    
    private boolean isValidEmail(String email) {
        Pattern pattern = Pattern.compile("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$");
        return pattern.matcher(email).matches();
    }
    
    // Business methods
    public boolean isAdult() {
        return age != null && age >= 18;
    }
    
    public void deactivate() {
        this.active = false;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addRole(Role role) {
        this.roles.add(role);
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean hasRole(Role role) {
        return this.roles.contains(role);
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getUsername() { return username; }
    public void setUsername(String username) { 
        this.username = username; 
        this.updatedAt = LocalDateTime.now();
    }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { 
        this.email = email; 
        this.updatedAt = LocalDateTime.now();
    }
    
    public Integer getAge() { return age; }
    public void setAge(Integer age) { 
        this.age = age; 
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isActive() { return active; }
    public void setActive(boolean active) { 
        this.active = active; 
        this.updatedAt = LocalDateTime.now();
    }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public Set<Role> getRoles() { return new HashSet<>(roles); }
    public void setRoles(Set<Role> roles) { 
        this.roles = new HashSet<>(roles); 
        this.updatedAt = LocalDateTime.now();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return Objects.equals(id, user.id) && Objects.equals(username, user.username);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id, username);
    }
    
    @Override
    public String toString() {
        return String.format("User{id=%d, username='%s', email='%s', age=%d, active=%s}", 
                           id, username, email, age, active);
    }
}

/**
 * Role enumeration with descriptions
 */
public enum Role {
    ADMIN("Administrator", "Full system access"),
    USER("User", "Standard user access"),
    MODERATOR("Moderator", "Content moderation access"),
    GUEST("Guest", "Limited access");
    
    private final String displayName;
    private final String description;
    
    Role(String displayName, String description) {
        this.displayName = displayName;
        this.description = description;
    }
    
    public String getDisplayName() { return displayName; }
    public String getDescription() { return description; }
}

/**
 * Product entity demonstrating inheritance and polymorphism
 */
public abstract class Product {
    protected Long id;
    protected String name;
    protected String description;
    protected double basePrice;
    protected Category category;
    protected LocalDateTime createdAt;
    protected boolean available = true;
    
    public Product(String name, String description, double basePrice, Category category) {
        this.name = name;
        this.description = description;
        this.basePrice = basePrice;
        this.category = category;
        this.createdAt = LocalDateTime.now();
    }
    
    // Abstract methods for polymorphism
    public abstract double calculatePrice();
    public abstract double calculateShippingCost();
    public abstract String getProductType();
    
    // Template method pattern
    public final double getTotalCost() {
        return calculatePrice() + calculateShippingCost();
    }
    
    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public double getBasePrice() { return basePrice; }
    public void setBasePrice(double basePrice) { this.basePrice = basePrice; }
    
    public Category getCategory() { return category; }
    public void setCategory(Category category) { this.category = category; }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public boolean isAvailable() { return available; }
    public void setAvailable(boolean available) { this.available = available; }
}

/**
 * Physical product implementation
 */
public class PhysicalProduct extends Product {
    private double weight;
    private Dimensions dimensions;
    
    public PhysicalProduct(String name, String description, double basePrice, 
                          Category category, double weight, Dimensions dimensions) {
        super(name, description, basePrice, category);
        this.weight = weight;
        this.dimensions = dimensions;
    }
    
    @Override
    public double calculatePrice() {
        // Add weight-based pricing
        return basePrice + (weight * 0.5);
    }
    
    @Override
    public double calculateShippingCost() {
        // Weight and size-based shipping
        double weightCost = weight * 2.0;
        double sizeCost = dimensions.getVolume() * 0.1;
        return weightCost + sizeCost;
    }
    
    @Override
    public String getProductType() {
        return "Physical";
    }
    
    public double getWeight() { return weight; }
    public void setWeight(double weight) { this.weight = weight; }
    
    public Dimensions getDimensions() { return dimensions; }
    public void setDimensions(Dimensions dimensions) { this.dimensions = dimensions; }
}

/**
 * Digital product implementation
 */
public class DigitalProduct extends Product {
    private long fileSizeBytes;
    private String downloadUrl;
    private int downloadLimit;
    
    public DigitalProduct(String name, String description, double basePrice, 
                         Category category, long fileSizeBytes, int downloadLimit) {
        super(name, description, basePrice, category);
        this.fileSizeBytes = fileSizeBytes;
        this.downloadLimit = downloadLimit;
    }
    
    @Override
    public double calculatePrice() {
        // No additional cost for digital products
        return basePrice;
    }
    
    @Override
    public double calculateShippingCost() {
        // No shipping cost for digital products
        return 0.0;
    }
    
    @Override
    public String getProductType() {
        return "Digital";
    }
    
    public long getFileSizeBytes() { return fileSizeBytes; }
    public void setFileSizeBytes(long fileSizeBytes) { this.fileSizeBytes = fileSizeBytes; }
    
    public String getDownloadUrl() { return downloadUrl; }
    public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
    
    public int getDownloadLimit() { return downloadLimit; }
    public void setDownloadLimit(int downloadLimit) { this.downloadLimit = downloadLimit; }
}

/**
 * Dimensions value object
 */
public class Dimensions {
    private final double length;
    private final double width;
    private final double height;
    
    public Dimensions(double length, double width, double height) {
        this.length = length;
        this.width = width;
        this.height = height;
    }
    
    public double getVolume() {
        return length * width * height;
    }
    
    public double getLength() { return length; }
    public double getWidth() { return width; }
    public double getHeight() { return height; }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Dimensions that = (Dimensions) o;
        return Double.compare(that.length, length) == 0 &&
               Double.compare(that.width, width) == 0 &&
               Double.compare(that.height, height) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(length, width, height);
    }
    
    @Override
    public String toString() {
        return String.format("%.2f x %.2f x %.2f", length, width, height);
    }
}

/**
 * Category enumeration
 */
public enum Category {
    ELECTRONICS, BOOKS, CLOTHING, HOME, SPORTS, SOFTWARE, MEDIA
}

// ========== Generic Repository Pattern ==========

/**
 * Generic repository interface
 */
public interface Repository<T, ID> {
    Optional<T> findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
    boolean existsById(ID id);
    long count();
}

/**
 * Searchable repository interface
 */
public interface SearchableRepository<T> {
    List<T> search(String query);
    List<T> findByField(String field, Object value);
}

/**
 * In-memory repository implementation
 */
public class InMemoryRepository<T, ID> implements Repository<T, ID>, SearchableRepository<T> {
    private final Map<ID, T> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);
    private final Function<T, ID> idExtractor;
    private final BiConsumer<T, ID> idSetter;
    
    public InMemoryRepository(Function<T, ID> idExtractor, BiConsumer<T, ID> idSetter) {
        this.idExtractor = idExtractor;
        this.idSetter = idSetter;
    }
    
    @Override
    public Optional<T> findById(ID id) {
        return Optional.ofNullable(storage.get(id));
    }
    
    @Override
    public List<T> findAll() {
        return new ArrayList<>(storage.values());
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public T save(T entity) {
        ID id = idExtractor.apply(entity);
        if (id == null) {
            // Generate new ID for new entities
            id = (ID) Long.valueOf(idGenerator.getAndIncrement());
            idSetter.accept(entity, id);
        }
        storage.put(id, entity);
        return entity;
    }
    
    @Override
    public void deleteById(ID id) {
        storage.remove(id);
    }
    
    @Override
    public boolean existsById(ID id) {
        return storage.containsKey(id);
    }
    
    @Override
    public long count() {
        return storage.size();
    }
    
    @Override
    public List<T> search(String query) {
        return storage.values().stream()
                .filter(entity -> entity.toString().toLowerCase().contains(query.toLowerCase()))
                .collect(Collectors.toList());
    }
    
    @Override
    public List<T> findByField(String field, Object value) {
        return storage.values().stream()
                .filter(entity -> {
                    try {
                        Field f = entity.getClass().getDeclaredField(field);
                        f.setAccessible(true);
                        Object fieldValue = f.get(entity);
                        return Objects.equals(fieldValue, value);
                    } catch (Exception e) {
                        return false;
                    }
                })
                .collect(Collectors.toList());
    }
}

// ========== Service Layer with Annotations ==========

/**
 * Custom annotations for dependency injection and validation
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface Inject {
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Service {
    String value() default "";
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Transactional {
    boolean readOnly() default false;
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Cacheable {
    String value() default "";
    int ttlSeconds() default 300;
}

/**
 * User service with business logic
 */
@Service("userService")
public class UserService {
    @Inject
    private Repository<User, Long> userRepository;
    
    @Inject
    private ValidationService validationService;
    
    private final Map<String, Object> cache = new ConcurrentHashMap<>();
    
    public UserService() {
        // Constructor injection simulation
        this.userRepository = new InMemoryRepository<>(User::getId, User::setId);
        this.validationService = new ValidationService();
    }
    
    @Transactional
    public User createUser(String username, String email, Integer age) {
        // Validate input
        if (username == null || username.trim().isEmpty()) {
            throw new IllegalArgumentException("Username cannot be null or empty");
        }
        
        // Check for duplicates
        List<User> existingUsers = ((SearchableRepository<User>) userRepository).findByField("username", username);
        if (!existingUsers.isEmpty()) {
            throw new IllegalStateException("Username already exists: " + username);
        }
        
        // Create and validate user
        User user = User.builder()
                .username(username)
                .email(email)
                .age(age)
                .role(Role.USER)
                .build();
        
        validationService.validate(user);
        
        // Save user
        return userRepository.save(user);
    }
    
    @Cacheable("users")
    public Optional<User> getUserById(Long id) {
        String cacheKey = "user:" + id;
        User cachedUser = (User) cache.get(cacheKey);
        if (cachedUser != null) {
            return Optional.of(cachedUser);
        }
        
        Optional<User> user = userRepository.findById(id);
        user.ifPresent(u -> cache.put(cacheKey, u));
        return user;
    }
    
    @Transactional
    public User updateUser(Long id, String username, String email, Integer age) {
        User user = getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("User not found: " + id));
        
        if (username != null) user.setUsername(username);
        if (email != null) user.setEmail(email);
        if (age != null) user.setAge(age);
        
        user.validate();
        cache.remove("user:" + id); // Invalidate cache
        
        return userRepository.save(user);
    }
    
    @Transactional
    public void deleteUser(Long id) {
        if (!userRepository.existsById(id)) {
            throw new IllegalArgumentException("User not found: " + id);
        }
        userRepository.deleteById(id);
        cache.remove("user:" + id);
    }
    
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }
    
    public List<User> searchUsers(String query) {
        return ((SearchableRepository<User>) userRepository).search(query);
    }
    
    public List<User> getActiveUsers() {
        return userRepository.findAll().stream()
                .filter(User::isActive)
                .collect(Collectors.toList());
    }
    
    public List<User> getUsersByRole(Role role) {
        return userRepository.findAll().stream()
                .filter(user -> user.hasRole(role))
                .collect(Collectors.toList());
    }
    
    public Map<Role, Long> getUserCountByRole() {
        return userRepository.findAll().stream()
                .flatMap(user -> user.getRoles().stream())
                .collect(Collectors.groupingBy(
                    Function.identity(),
                    Collectors.counting()
                ));
    }
    
    public OptionalDouble getAverageAge() {
        return userRepository.findAll().stream()
                .filter(user -> user.getAge() != null)
                .mapToInt(User::getAge)
                .average();
    }
}

/**
 * Validation service
 */
@Service("validationService")
public class ValidationService {
    
    public void validate(User user) {
        user.validate(); // Delegate to entity validation
        
        // Additional business validation rules
        if (user.getAge() != null && user.getAge() < 13) {
            throw new IllegalArgumentException("Users must be at least 13 years old");
        }
    }
    
    public void validate(Product product) {
        if (product.getName() == null || product.getName().trim().isEmpty()) {
            throw new IllegalArgumentException("Product name cannot be null or empty");
        }
        if (product.getBasePrice() < 0) {
            throw new IllegalArgumentException("Product price cannot be negative");
        }
    }
}

// ========== Functional Programming and Streams ==========

/**
 * Data analysis service using streams and functional programming
 */
@Service("analyticsService")
public class AnalyticsService {
    
    private final UserService userService;
    
    public AnalyticsService(UserService userService) {
        this.userService = userService;
    }
    
    /**
     * Advanced user analytics using streams
     */
    public Map<String, Object> getUserAnalytics() {
        List<User> users = userService.getAllUsers();
        
        Map<String, Object> analytics = new HashMap<>();
        
        // Basic counts
        analytics.put("totalUsers", users.size());
        analytics.put("activeUsers", users.stream().filter(User::isActive).count());
        analytics.put("adultUsers", users.stream().filter(User::isAdult).count());
        
        // Age statistics
        OptionalDouble averageAge = users.stream()
                .filter(user -> user.getAge() != null)
                .mapToInt(User::getAge)
                .average();
        analytics.put("averageAge", averageAge.orElse(0.0));
        
        IntSummaryStatistics ageStats = users.stream()
                .filter(user -> user.getAge() != null)
                .mapToInt(User::getAge)
                .summaryStatistics();
        analytics.put("ageStatistics", Map.of(
            "min", ageStats.getMin(),
            "max", ageStats.getMax(),
            "average", ageStats.getAverage(),
            "count", ageStats.getCount()
        ));
        
        // Role distribution
        Map<Role, Long> roleDistribution = users.stream()
                .flatMap(user -> user.getRoles().stream())
                .collect(Collectors.groupingBy(
                    Function.identity(),
                    Collectors.counting()
                ));
        analytics.put("roleDistribution", roleDistribution);
        
        // Recent users (last 30 days)
        LocalDateTime thirtyDaysAgo = LocalDateTime.now().minusDays(30);
        long recentUsers = users.stream()
                .filter(user -> user.getCreatedAt().isAfter(thirtyDaysAgo))
                .count();
        analytics.put("recentUsers", recentUsers);
        
        // Top email domains
        Map<String, Long> emailDomains = users.stream()
                .map(user -> user.getEmail().substring(user.getEmail().indexOf('@') + 1))
                .collect(Collectors.groupingBy(
                    Function.identity(),
                    Collectors.counting()
                ));
        analytics.put("topEmailDomains", emailDomains.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(10)
                .collect(Collectors.toMap(
                    Map.Entry::getKey,
                    Map.Entry::getValue,
                    (e1, e2) -> e1,
                    LinkedHashMap::new
                )));
        
        return analytics;
    }
    
    /**
     * Functional pipeline for user processing
     */
    public List<String> processUserPipeline() {
        return userService.getAllUsers().stream()
                .filter(User::isActive)                           // Filter active users
                .filter(User::isAdult)                           // Filter adult users
                .sorted(Comparator.comparing(User::getUsername)) // Sort by username
                .map(user -> String.format("%s (%s)", 
                    user.getUsername(), user.getEmail()))       // Transform to string
                .collect(Collectors.toList());                  // Collect results
    }
    
    /**
     * Parallel processing example
     */
    public Map<String, Object> processUsersInParallel() {
        List<User> users = userService.getAllUsers();
        
        // Parallel stream for CPU-intensive operations
        Map<String, Long> results = users.parallelStream()
                .filter(User::isActive)
                .map(user -> {
                    // Simulate CPU-intensive operation
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    return user.getEmail().substring(user.getEmail().indexOf('@') + 1);
                })
                .collect(Collectors.groupingBy(
                    Function.identity(),
                    Collectors.counting()
                ));
        
        return Map.of(
            "processedCount", users.size(),
            "domainCounts", results,
            "processingTime", System.currentTimeMillis()
        );
    }
}

// ========== Concurrency and Async Programming ==========

/**
 * Asynchronous service using CompletableFuture
 */
@Service("asyncService")
public class AsyncUserService {
    
    private final UserService userService;
    private final ExecutorService executorService = Executors.newFixedThreadPool(10);
    
    public AsyncUserService(UserService userService) {
        this.userService = userService;
    }
    
    /**
     * Asynchronous user creation
     */
    public CompletableFuture<User> createUserAsync(String username, String email, Integer age) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(100); // Simulate network delay
                return userService.createUser(username, email, age);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("User creation interrupted", e);
            }
        }, executorService);
    }
    
    /**
     * Asynchronous batch user creation
     */
    public CompletableFuture<List<User>> createUsersAsync(List<Map<String, Object>> userDataList) {
        List<CompletableFuture<User>> futures = userDataList.stream()
                .map(userData -> createUserAsync(
                    (String) userData.get("username"),
                    (String) userData.get("email"),
                    (Integer) userData.get("age")
                ))
                .collect(Collectors.toList());
        
        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> futures.stream()
                        .map(CompletableFuture::join)
                        .collect(Collectors.toList()));
    }
    
    /**
     * Asynchronous user validation with timeout
     */
    public CompletableFuture<Boolean> validateUserAsync(User user, Duration timeout) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(50); // Simulate validation delay
                user.validate();
                return true;
            } catch (Exception e) {
                return false;
            }
        }, executorService)
        .orTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS)
        .exceptionally(throwable -> false);
    }
    
    /**
     * Pipeline processing with error handling
     */
    public CompletableFuture<Map<String, Object>> processUserPipelineAsync(List<User> users) {
        return CompletableFuture.supplyAsync(() -> {
            return users.stream()
                    .filter(User::isActive)
                    .count();
        }, executorService)
        .thenCompose(activeCount -> 
            CompletableFuture.supplyAsync(() -> {
                return users.stream()
                        .filter(User::isAdult)
                        .count();
            }, executorService)
            .thenApply(adultCount -> Map.of(
                "activeUsers", activeCount,
                "adultUsers", adultCount,
                "totalUsers", (long) users.size()
            ))
        )
        .exceptionally(throwable -> {
            return Map.of("error", throwable.getMessage());
        });
    }
    
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}

// ========== File I/O and Serialization ==========

/**
 * File operations service
 */
@Service("fileService")
public class FileService {
    
    private final ObjectMapper objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
    
    /**
     * Save users to JSON file
     */
    public void saveUsersToJson(List<User> users, Path filePath) throws IOException {
        objectMapper.writeValue(filePath.toFile(), users);
    }
    
    /**
     * Load users from JSON file
     */
    public List<User> loadUsersFromJson(Path filePath) throws IOException {
        return objectMapper.readValue(filePath.toFile(), 
                objectMapper.getTypeFactory().constructCollectionType(List.class, User.class));
    }
    
    /**
     * Save users to CSV file
     */
    public void saveUsersToCsv(List<User> users, Path filePath) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(filePath, StandardCharsets.UTF_8)) {
            // Write header
            writer.write("ID,Username,Email,Age,Active,Created,Roles\n");
            
            // Write data
            for (User user : users) {
                String roles = user.getRoles().stream()
                        .map(Role::name)
                        .collect(Collectors.joining(";"));
                
                writer.write(String.format("%d,%s,%s,%s,%s,%s,%s\n",
                        user.getId(),
                        escapeCSV(user.getUsername()),
                        escapeCSV(user.getEmail()),
                        user.getAge() != null ? user.getAge().toString() : "",
                        user.isActive(),
                        user.getCreatedAt().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
                        roles
                ));
            }
        }
    }
    
    private String escapeCSV(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }
    
    /**
     * Process large files with streams
     */
    public long processLargeFile(Path filePath, Predicate<String> lineFilter) throws IOException {
        try (Stream<String> lines = Files.lines(filePath)) {
            return lines
                    .filter(lineFilter)
                    .count();
        }
    }
    
    /**
     * Create backup of data
     */
    public void createBackup(List<User> users, Path backupDir) throws IOException {
        if (!Files.exists(backupDir)) {
            Files.createDirectories(backupDir);
        }
        
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        Path backupFile = backupDir.resolve("users_backup_" + timestamp + ".json");
        
        saveUsersToJson(users, backupFile);
        
        // Compress backup
        Path compressedFile = backupDir.resolve("users_backup_" + timestamp + ".json.gz");
        try (FileInputStream fis = new FileInputStream(backupFile.toFile());
             FileOutputStream fos = new FileOutputStream(compressedFile.toFile());
             GZIPOutputStream gzos = new GZIPOutputStream(fos)) {
            
            byte[] buffer = new byte[1024];
            int length;
            while ((length = fis.read(buffer)) != -1) {
                gzos.write(buffer, 0, length);
            }
        }
        
        // Delete uncompressed file
        Files.delete(backupFile);
    }
}

// ========== Demonstration Classes ==========

/**
 * Main demonstration class
 */
public class JavaExamplesDemo {
    
    public static void main(String[] args) {
        JavaExamplesDemo demo = new JavaExamplesDemo();
        
        try {
            demo.demonstrateBasicOperations();
            demo.demonstrateStreamProcessing();
            demo.demonstrateAsyncProgramming();
            demo.demonstrateFileOperations();
            demo.demonstrateReflectionAndAnnotations();
            demo.printFeaturesSummary();
        } catch (Exception e) {
            System.err.println("Demo failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void demonstrateBasicOperations() {
        System.out.println("=== Basic Operations Demo ===");
        
        UserService userService = new UserService();
        
        // Create users
        User user1 = userService.createUser("alice", "alice@example.com", 25);
        User user2 = userService.createUser("bob", "bob@example.com", 17);
        User user3 = userService.createUser("charlie", "charlie@example.com", 30);
        
        System.out.println("Created users:");
        System.out.println("- " + user1);
        System.out.println("- " + user2);
        System.out.println("- " + user3);
        
        // Update user
        User updatedUser = userService.updateUser(user1.getId(), "alice_updated", null, 26);
        System.out.println("Updated user: " + updatedUser);
        
        // Search users
        List<User> searchResults = userService.searchUsers("alice");
        System.out.println("Search results for 'alice': " + searchResults.size() + " users");
        
        // Get active users
        List<User> activeUsers = userService.getActiveUsers();
        System.out.println("Active users: " + activeUsers.size());
        
        // Role-based queries
        Map<Role, Long> roleDistribution = userService.getUserCountByRole();
        System.out.println("Role distribution: " + roleDistribution);
        
        System.out.println();
    }
    
    private void demonstrateStreamProcessing() {
        System.out.println("=== Stream Processing Demo ===");
        
        UserService userService = new UserService();
        AnalyticsService analyticsService = new AnalyticsService(userService);
        
        // Create test data
        for (int i = 1; i <= 20; i++) {
            userService.createUser("user" + i, "user" + i + "@example.com", 
                                 18 + (i % 50));
        }
        
        // Analytics
        Map<String, Object> analytics = analyticsService.getUserAnalytics();
        System.out.println("User Analytics:");
        analytics.forEach((key, value) -> 
            System.out.println("- " + key + ": " + value));
        
        // Pipeline processing
        List<String> processedUsers = analyticsService.processUserPipeline();
        System.out.println("Processed users (first 5): " + 
                          processedUsers.stream().limit(5).collect(Collectors.toList()));
        
        // Parallel processing
        Map<String, Object> parallelResults = analyticsService.processUsersInParallel();
        System.out.println("Parallel processing results: " + parallelResults);
        
        System.out.println();
    }
    
    private void demonstrateAsyncProgramming() {
        System.out.println("=== Async Programming Demo ===");
        
        UserService userService = new UserService();
        AsyncUserService asyncService = new AsyncUserService(userService);
        
        try {
            // Single async operation
            CompletableFuture<User> futureUser = asyncService.createUserAsync(
                "async_user", "async@example.com", 25);
            User asyncUser = futureUser.get(5, TimeUnit.SECONDS);
            System.out.println("Async created user: " + asyncUser);
            
            // Batch async operations
            List<Map<String, Object>> userData = Arrays.asList(
                Map.of("username", "batch1", "email", "batch1@example.com", "age", 22),
                Map.of("username", "batch2", "email", "batch2@example.com", "age", 28),
                Map.of("username", "batch3", "email", "batch3@example.com", "age", 35)
            );
            
            CompletableFuture<List<User>> futureUsers = asyncService.createUsersAsync(userData);
            List<User> batchUsers = futureUsers.get(10, TimeUnit.SECONDS);
            System.out.println("Batch created users: " + batchUsers.size());
            
            // Async validation with timeout
            CompletableFuture<Boolean> validationResult = asyncService.validateUserAsync(
                asyncUser, Duration.ofSeconds(1));
            boolean isValid = validationResult.get(2, TimeUnit.SECONDS);
            System.out.println("Async validation result: " + isValid);
            
            // Pipeline processing
            List<User> allUsers = userService.getAllUsers();
            CompletableFuture<Map<String, Object>> pipelineResult = 
                asyncService.processUserPipelineAsync(allUsers);
            Map<String, Object> result = pipelineResult.get(5, TimeUnit.SECONDS);
            System.out.println("Pipeline processing result: " + result);
            
        } catch (Exception e) {
            System.err.println("Async operation failed: " + e.getMessage());
        } finally {
            asyncService.shutdown();
        }
        
        System.out.println();
    }
    
    private void demonstrateFileOperations() {
        System.out.println("=== File Operations Demo ===");
        
        UserService userService = new UserService();
        FileService fileService = new FileService();
        
        try {
            // Create test users
            List<User> users = Arrays.asList(
                userService.createUser("file_user1", "file1@example.com", 25),
                userService.createUser("file_user2", "file2@example.com", 30),
                userService.createUser("file_user3", "file3@example.com", 22)
            );
            
            // Save to JSON
            Path jsonFile = Paths.get("/tmp/users.json");
            fileService.saveUsersToJson(users, jsonFile);
            System.out.println("Saved " + users.size() + " users to JSON file");
            
            // Load from JSON
            List<User> loadedUsers = fileService.loadUsersFromJson(jsonFile);
            System.out.println("Loaded " + loadedUsers.size() + " users from JSON file");
            
            // Save to CSV
            Path csvFile = Paths.get("/tmp/users.csv");
            fileService.saveUsersToCsv(users, csvFile);
            System.out.println("Saved users to CSV file");
            
            // Create backup
            Path backupDir = Paths.get("/tmp/backups");
            fileService.createBackup(users, backupDir);
            System.out.println("Created compressed backup");
            
            // Process large file (simulate)
            Files.write(Paths.get("/tmp/large_file.txt"), 
                       Arrays.asList("line1", "line2", "line3", "important_line", "line5"));
            
            long importantLines = fileService.processLargeFile(
                Paths.get("/tmp/large_file.txt"),
                line -> line.contains("important")
            );
            System.out.println("Found " + importantLines + " important lines");
            
        } catch (IOException e) {
            System.err.println("File operation failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    private void demonstrateReflectionAndAnnotations() {
        System.out.println("=== Reflection and Annotations Demo ===");
        
        // Analyze UserService class
        Class<?> userServiceClass = UserService.class;
        
        // Check class annotations
        Service serviceAnnotation = userServiceClass.getAnnotation(Service.class);
        if (serviceAnnotation != null) {
            System.out.println("Service name: " + serviceAnnotation.value());
        }
        
        // Analyze methods
        Method[] methods = userServiceClass.getDeclaredMethods();
        for (Method method : methods) {
            if (method.isAnnotationPresent(Transactional.class)) {
                Transactional tx = method.getAnnotation(Transactional.class);
                System.out.println("Transactional method: " + method.getName() + 
                                 " (readOnly: " + tx.readOnly() + ")");
            }
            
            if (method.isAnnotationPresent(Cacheable.class)) {
                Cacheable cache = method.getAnnotation(Cacheable.class);
                System.out.println("Cacheable method: " + method.getName() + 
                                 " (TTL: " + cache.ttlSeconds() + "s)");
            }
        }
        
        // Analyze fields
        Field[] fields = userServiceClass.getDeclaredFields();
        for (Field field : fields) {
            if (field.isAnnotationPresent(Inject.class)) {
                System.out.println("Injectable field: " + field.getName() + 
                                 " (" + field.getType().getSimpleName() + ")");
            }
        }
        
        System.out.println();
    }
    
    private void printFeaturesSummary() {
        System.out.println("=== Java Features Demonstrated ===");
        System.out.println("- Object-Oriented Programming (inheritance, polymorphism, encapsulation)");
        System.out.println("- Generics and type safety");
        System.out.println("- Builder pattern and fluent interfaces");
        System.out.println("- Repository pattern with in-memory implementation");
        System.out.println("- Service layer with business logic");
        System.out.println("- Stream API and functional programming");
        System.out.println("- Lambda expressions and method references");
        System.out.println("- CompletableFuture for asynchronous programming");
        System.out.println("- Custom annotations and reflection");
        System.out.println("- File I/O with NIO.2");
        System.out.println("- JSON serialization with Jackson");
        System.out.println("- Exception handling and validation");
        System.out.println("- Collections and concurrent collections");
        System.out.println("- Enum types with behavior");
        System.out.println("- Time API (java.time)");
        System.out.println("- Optional for null safety");
        System.out.println("- Parallel streams for concurrent processing");
    }
}