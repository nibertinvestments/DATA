# Basic PHP Dataset - Server-Side Web Development

## Dataset 1: Hello World and Basic Syntax
```php
<?php
// Simple Hello World
echo "Hello, World!";
?>

<?php
// Hello World with functions
function greet() {
    echo "Hello, World!";
}

function greetWithName($name) {
    echo "Hello, $name!";
}

greet();
echo "\n";
greetWithName("PHP");
?>
```

## Dataset 2: Variables and Data Types
```php
<?php
// Variables (start with $)
$message = "Hello PHP";
$count = 42;
$price = 3.14;
$isActive = true;

// String interpolation
$name = "John";
$age = 30;

// Arrays
$numbers = [1, 2, 3, 4, 5];
$fruits = array("apple", "banana", "cherry");

// Associative arrays
$person = [
    "name" => "Alice",
    "age" => 25,
    "email" => "alice@example.com"
];

// Multi-dimensional arrays
$users = [
    ["name" => "John", "age" => 30],
    ["name" => "Jane", "age" => 25]
];

// Output
echo "Message: $message\n";
echo "Count: $count\n";
echo "Price: $price\n";
echo "Is Active: " . ($isActive ? "true" : "false") . "\n";
echo "Name: $name, Age: $age\n";

// Array output
echo "Numbers: " . implode(", ", $numbers) . "\n";
echo "Fruits: " . implode(", ", $fruits) . "\n";

// Associative array access
echo "Person name: " . $person["name"] . "\n";
echo "Person age: " . $person["age"] . "\n";

// Variable types
echo "Type of message: " . gettype($message) . "\n";
echo "Type of count: " . gettype($count) . "\n";
echo "Type of price: " . gettype($price) . "\n";
echo "Type of isActive: " . gettype($isActive) . "\n";
?>
```

## Dataset 3: Control Structures
```php
<?php
// If-else statements
$age = 18;
if ($age >= 18) {
    echo "Adult\n";
} elseif ($age >= 13) {
    echo "Teenager\n";
} else {
    echo "Child\n";
}

// Ternary operator
$status = ($age >= 18) ? "adult" : "minor";
echo "Status: $status\n";

// For loops
for ($i = 0; $i < 5; $i++) {
    echo "Number: $i\n";
}

// Foreach loops
$numbers = [1, 2, 3, 4, 5];
foreach ($numbers as $number) {
    echo "Foreach: $number\n";
}

// Foreach with key-value pairs
$person = ["name" => "Alice", "age" => 25, "city" => "New York"];
foreach ($person as $key => $value) {
    echo "$key: $value\n";
}

// While loop
$count = 0;
while ($count < 5) {
    echo "Count: $count\n";
    $count++;
}

// Do-while loop
$doCount = 0;
do {
    echo "Do-while: $doCount\n";
    $doCount++;
} while ($doCount < 3);

// Switch statement
$day = "Monday";
switch ($day) {
    case "Monday":
        echo "Start of the week\n";
        break;
    case "Friday":
        echo "TGIF!\n";
        break;
    default:
        echo "Regular day\n";
        break;
}

// Match expression (PHP 8.0+)
$dayMessage = match($day) {
    "Monday" => "Start of the week",
    "Friday" => "TGIF!",
    default => "Regular day"
};
echo "Match expression: $dayMessage\n";
?>
```

## Dataset 4: Functions and Scope
```php
<?php
// Basic function
function add($a, $b) {
    return $a + $b;
}

// Function with default parameters
function greet($name, $greeting = "Hello") {
    return "$greeting, $name!";
}

// Function with variable number of arguments
function sum(...$numbers) {
    $total = 0;
    foreach ($numbers as $number) {
        $total += $number;
    }
    return $total;
}

// Function with type declarations (PHP 7+)
function divide(float $a, float $b): float {
    if ($b == 0) {
        throw new InvalidArgumentException("Division by zero");
    }
    return $a / $b;
}

// Anonymous function (closure)
$multiply = function($a, $b) {
    return $a * $b;
};

// Closure with use keyword
$factor = 10;
$multiplyByFactor = function($number) use ($factor) {
    return $number * $factor;
};

// Arrow functions (PHP 7.4+)
$square = fn($x) => $x * $x;

// Global and local scope
$globalVar = "I'm global";

function testScope() {
    $localVar = "I'm local";
    global $globalVar;
    echo "Inside function: $globalVar\n";
    echo "Inside function: $localVar\n";
}

// Function calls
echo "Add: " . add(5, 3) . "\n";
echo "Greet: " . greet("Alice") . "\n";
echo "Greet with custom: " . greet("Bob", "Hi") . "\n";
echo "Sum: " . sum(1, 2, 3, 4, 5) . "\n";

try {
    echo "Divide: " . divide(10, 2) . "\n";
    // echo "Divide by zero: " . divide(10, 0) . "\n"; // Would throw exception
} catch (InvalidArgumentException $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

echo "Multiply: " . $multiply(4, 5) . "\n";
echo "Multiply by factor: " . $multiplyByFactor(3) . "\n";
echo "Square: " . $square(5) . "\n";

testScope();
?>
```

## Dataset 5: Classes and Objects
```php
<?php
// Basic class
class Person {
    // Properties
    private $name;
    private $age;
    public $email;
    
    // Constructor
    public function __construct($name, $age) {
        $this->name = $name;
        $this->age = $age;
    }
    
    // Getter methods
    public function getName() {
        return $this->name;
    }
    
    public function getAge() {
        return $this->age;
    }
    
    // Setter methods
    public function setName($name) {
        $this->name = $name;
    }
    
    public function setAge($age) {
        if ($age >= 0) {
            $this->age = $age;
        }
    }
    
    // Methods
    public function introduce() {
        return "Hi, I'm {$this->name} and I'm {$this->age} years old";
    }
    
    public function haveBirthday() {
        $this->age++;
        return "Happy birthday! Now I'm {$this->age}";
    }
    
    // Static method
    public static function getSpecies() {
        return "Homo sapiens";
    }
}

// Inheritance
class Student extends Person {
    private $studentId;
    private $courses;
    
    public function __construct($name, $age, $studentId) {
        parent::__construct($name, $age);
        $this->studentId = $studentId;
        $this->courses = [];
    }
    
    public function getStudentId() {
        return $this->studentId;
    }
    
    public function addCourse($course) {
        $this->courses[] = $course;
    }
    
    public function getCourses() {
        return $this->courses;
    }
    
    // Override method
    public function introduce() {
        return parent::introduce() . ". My student ID is {$this->studentId}";
    }
}

// Using classes
$person1 = new Person("Alice", 25);
$person1->email = "alice@example.com";

$person2 = new Person("Bob", 30);

echo $person1->introduce() . "\n";
echo $person1->haveBirthday() . "\n";

echo $person2->introduce() . "\n";

// Static method
echo "Species: " . Person::getSpecies() . "\n";

// Inheritance
$student = new Student("Carol", 20, "S12345");
$student->addCourse("PHP Programming");
$student->addCourse("Web Development");

echo $student->introduce() . "\n";
echo "Courses: " . implode(", ", $student->getCourses()) . "\n";
?>
```

## Dataset 6: Arrays and Array Functions
```php
<?php
// Indexed arrays
$fruits = ["apple", "banana", "cherry", "date"];
$numbers = range(1, 10);

// Associative arrays
$person = [
    "name" => "John",
    "age" => 30,
    "city" => "New York",
    "country" => "USA"
];

// Multi-dimensional arrays
$users = [
    ["name" => "Alice", "age" => 25, "role" => "admin"],
    ["name" => "Bob", "age" => 30, "role" => "user"],
    ["name" => "Carol", "age" => 35, "role" => "moderator"]
];

// Array functions
echo "Array functions:\n";
echo "Count of fruits: " . count($fruits) . "\n";
echo "Is array: " . (is_array($fruits) ? "yes" : "no") . "\n";

// Adding elements
array_push($fruits, "elderberry");
$fruits[] = "fig"; // Alternative way

// Removing elements
$lastFruit = array_pop($fruits);
$firstFruit = array_shift($fruits);

echo "Last fruit: $lastFruit\n";
echo "First fruit: $firstFruit\n";
echo "Remaining fruits: " . implode(", ", $fruits) . "\n";

// Array search and manipulation
if (in_array("cherry", $fruits)) {
    echo "Cherry found in array\n";
}

$position = array_search("cherry", $fruits);
echo "Cherry is at position: $position\n";

// Array slicing
$slice = array_slice($fruits, 1, 2);
echo "Slice [1,2]: " . implode(", ", $slice) . "\n";

// Array mapping and filtering
$uppercaseFruits = array_map('strtoupper', $fruits);
echo "Uppercase fruits: " . implode(", ", $uppercaseFruits) . "\n";

$longFruits = array_filter($fruits, function($fruit) {
    return strlen($fruit) > 5;
});
echo "Long fruits: " . implode(", ", $longFruits) . "\n";

// Array reduce
$totalLength = array_reduce($fruits, function($carry, $fruit) {
    return $carry + strlen($fruit);
}, 0);
echo "Total length of all fruits: $totalLength\n";

// Sorting
$numbers = [3, 1, 4, 1, 5, 9, 2, 6];
sort($numbers);
echo "Sorted numbers: " . implode(", ", $numbers) . "\n";

// Associative array functions
echo "Person keys: " . implode(", ", array_keys($person)) . "\n";
echo "Person values: " . implode(", ", array_values($person)) . "\n";

// Array merge
$moreInfo = ["occupation" => "Developer", "salary" => 75000];
$fullPerson = array_merge($person, $moreInfo);
echo "Full person info:\n";
foreach ($fullPerson as $key => $value) {
    echo "  $key: $value\n";
}
?>
```

## Dataset 7: String Manipulation
```php
<?php
$text = "Hello, World!";

// String functions
echo "Original: $text\n";
echo "Uppercase: " . strtoupper($text) . "\n";
echo "Lowercase: " . strtolower($text) . "\n";
echo "Length: " . strlen($text) . "\n";
echo "Substring: " . substr($text, 0, 5) . "\n";
echo "Replace: " . str_replace("World", "PHP", $text) . "\n";
echo "Position of 'World': " . strpos($text, "World") . "\n";

// String interpolation and concatenation
$name = "Alice";
$age = 25;

echo "Name: $name, Age: $age\n"; // Variable interpolation
echo "Name: " . $name . ", Age: " . $age . "\n"; // Concatenation
echo sprintf("Name: %s, Age: %d\n", $name, $age); // sprintf formatting

// String splitting and joining
$sentence = "The quick brown fox jumps";
$words = explode(" ", $sentence);
echo "Words: " . implode(", ", $words) . "\n";

$joined = implode(" ", $words);
echo "Joined: $joined\n";

// String trimming
$paddedText = "   Hello World   ";
echo "Original: '$paddedText'\n";
echo "Trimmed: '" . trim($paddedText) . "'\n";
echo "Left trim: '" . ltrim($paddedText) . "'\n";
echo "Right trim: '" . rtrim($paddedText) . "'\n";

// String comparison
$str1 = "Hello";
$str2 = "hello";
echo "Case sensitive comparison: " . ($str1 === $str2 ? "equal" : "not equal") . "\n";
echo "Case insensitive comparison: " . (strcasecmp($str1, $str2) === 0 ? "equal" : "not equal") . "\n";

// String validation
$email = "user@example.com";
if (filter_var($email, FILTER_VALIDATE_EMAIL)) {
    echo "Valid email: $email\n";
} else {
    echo "Invalid email: $email\n";
}

// Regular expressions
$pattern = "/\d+/"; // Match numbers
$subject = "There are 123 apples and 456 oranges";
if (preg_match_all($pattern, $subject, $matches)) {
    echo "Numbers found: " . implode(", ", $matches[0]) . "\n";
}

// String formatting
$price = 1234.567;
echo "Formatted price: $" . number_format($price, 2) . "\n";

// Heredoc and Nowdoc
$heredoc = <<<EOD
This is a heredoc string.
It can span multiple lines.
Variables like $name are interpolated.
EOD;

$nowdoc = <<<'EOD'
This is a nowdoc string.
It can span multiple lines.
Variables like $name are NOT interpolated.
EOD;

echo "Heredoc:\n$heredoc\n";
echo "Nowdoc:\n$nowdoc\n";
?>
```

## Dataset 8: Error Handling and Exceptions
```php
<?php
// Custom exception class
class ValidationException extends Exception {
    public function __construct($message, $code = 0, Throwable $previous = null) {
        parent::__construct($message, $code, $previous);
    }
}

// Function that throws exceptions
function divide($a, $b) {
    if ($b == 0) {
        throw new InvalidArgumentException("Division by zero is not allowed");
    }
    return $a / $b;
}

function validateAge($age) {
    if ($age < 0) {
        throw new ValidationException("Age cannot be negative");
    }
    if ($age > 150) {
        throw new ValidationException("Age cannot be greater than 150");
    }
    return true;
}

// Basic try-catch
try {
    $result = divide(10, 2);
    echo "Division result: $result\n";
} catch (InvalidArgumentException $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

// Multiple catch blocks
try {
    $age = -5;
    validateAge($age);
    echo "Valid age: $age\n";
} catch (ValidationException $e) {
    echo "Validation error: " . $e->getMessage() . "\n";
} catch (Exception $e) {
    echo "General error: " . $e->getMessage() . "\n";
} finally {
    echo "This always executes\n";
}

// Error handling with set_error_handler
function customErrorHandler($errno, $errstr, $errfile, $errline) {
    echo "Custom error handler: [$errno] $errstr in $errfile on line $errline\n";
    return true; // Don't execute PHP internal error handler
}

set_error_handler("customErrorHandler");

// Trigger an error
// echo $undefinedVariable; // This would trigger a notice

// Restore default error handler
restore_error_handler();

// Exception handling in functions
function processUser($name, $age) {
    try {
        if (empty($name)) {
            throw new InvalidArgumentException("Name cannot be empty");
        }
        
        validateAge($age);
        
        echo "User '$name' (age: $age) processed successfully\n";
        return true;
    } catch (Exception $e) {
        echo "Failed to process user: " . $e->getMessage() . "\n";
        return false;
    }
}

// Test error handling
$testCases = [
    ["Alice", 25],
    ["", 30],
    ["Bob", -5],
    ["Carol", 200]
];

foreach ($testCases as [$name, $age]) {
    processUser($name, $age);
}

// File operations with error handling
function readFileContent($filename) {
    try {
        if (!file_exists($filename)) {
            throw new Exception("File does not exist: $filename");
        }
        
        $content = file_get_contents($filename);
        if ($content === false) {
            throw new Exception("Failed to read file: $filename");
        }
        
        return $content;
    } catch (Exception $e) {
        echo "File error: " . $e->getMessage() . "\n";
        return null;
    }
}

// Test file reading
$content = readFileContent("nonexistent.txt");
if ($content !== null) {
    echo "File content: $content\n";
}
?>
```

## Dataset 9: File I/O and JSON
```php
<?php
// Writing to file
$content = "Hello, File!\nThis is written from PHP\nLine 3";
file_put_contents("output.txt", $content);
echo "File written successfully\n";

// Reading from file
$readContent = file_get_contents("output.txt");
echo "File content:\n$readContent\n";

// Reading file line by line
$lines = file("output.txt", FILE_IGNORE_NEW_LINES);
echo "Reading line by line:\n";
foreach ($lines as $lineNumber => $line) {
    echo "Line " . ($lineNumber + 1) . ": $line\n";
}

// Using file handles
$handle = fopen("handle_output.txt", "w");
if ($handle) {
    fwrite($handle, "Line 1\n");
    fwrite($handle, "Line 2\n");
    fwrite($handle, "Current time: " . date("Y-m-d H:i:s") . "\n");
    fclose($handle);
    echo "File written using handle\n";
}

// Reading with file handle
$handle = fopen("handle_output.txt", "r");
if ($handle) {
    echo "Reading with handle:\n";
    while (($line = fgets($handle)) !== false) {
        echo $line;
    }
    fclose($handle);
}

// CSV file operations
$csvData = [
    ["Name", "Age", "City"],
    ["Alice", 25, "New York"],
    ["Bob", 30, "London"],
    ["Carol", 35, "Paris"]
];

// Writing CSV
$csvHandle = fopen("data.csv", "w");
foreach ($csvData as $row) {
    fputcsv($csvHandle, $row);
}
fclose($csvHandle);
echo "CSV file written\n";

// Reading CSV
$csvHandle = fopen("data.csv", "r");
echo "CSV data:\n";
while (($row = fgetcsv($csvHandle)) !== false) {
    echo implode(", ", $row) . "\n";
}
fclose($csvHandle);

// JSON operations
$person = [
    "name" => "Alice",
    "age" => 30,
    "email" => "alice@example.com",
    "hobbies" => ["reading", "coding", "hiking"]
];

// Encoding JSON
$jsonString = json_encode($person, JSON_PRETTY_PRINT);
echo "JSON encoded:\n$jsonString\n";

// Writing JSON to file
file_put_contents("person.json", $jsonString);
echo "JSON file written\n";

// Reading and decoding JSON
$jsonFromFile = file_get_contents("person.json");
$decodedPerson = json_decode($jsonFromFile, true); // true for associative array

echo "Decoded JSON:\n";
echo "Name: " . $decodedPerson["name"] . "\n";
echo "Age: " . $decodedPerson["age"] . "\n";
echo "Hobbies: " . implode(", ", $decodedPerson["hobbies"]) . "\n";

// JSON error handling
$invalidJson = '{"name": "Alice", "age": }'; // Invalid JSON
$decoded = json_decode($invalidJson);

if (json_last_error() !== JSON_ERROR_NONE) {
    echo "JSON error: " . json_last_error_msg() . "\n";
}

// Directory operations
$directory = "test_directory";
if (!is_dir($directory)) {
    mkdir($directory);
    echo "Directory created: $directory\n";
}

// List directory contents
$files = scandir(".");
echo "Current directory contents:\n";
foreach ($files as $file) {
    if ($file !== "." && $file !== "..") {
        echo "  $file\n";
    }
}

// File information
$filename = "output.txt";
if (file_exists($filename)) {
    echo "File info for $filename:\n";
    echo "  Size: " . filesize($filename) . " bytes\n";
    echo "  Modified: " . date("Y-m-d H:i:s", filemtime($filename)) . "\n";
    echo "  Readable: " . (is_readable($filename) ? "yes" : "no") . "\n";
    echo "  Writable: " . (is_writable($filename) ? "yes" : "no") . "\n";
}
?>
```

## Dataset 10: Database Operations (MySQL with PDO)
```php
<?php
// Database configuration
$host = 'localhost';
$dbname = 'test_database';
$username = 'username';
$password = 'password';

// PDO connection with error handling
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8mb4", $username, $password, [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC
    ]);
    echo "Database connected successfully\n";
} catch (PDOException $e) {
    echo "Connection failed: " . $e->getMessage() . "\n";
    exit;
}

// Create table
try {
    $createTable = "
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            age INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ";
    $pdo->exec($createTable);
    echo "Table created successfully\n";
} catch (PDOException $e) {
    echo "Error creating table: " . $e->getMessage() . "\n";
}

// Insert data (prepared statements)
function insertUser($pdo, $name, $email, $age) {
    try {
        $stmt = $pdo->prepare("INSERT INTO users (name, email, age) VALUES (?, ?, ?)");
        $stmt->execute([$name, $email, $age]);
        echo "User inserted: $name\n";
        return $pdo->lastInsertId();
    } catch (PDOException $e) {
        echo "Error inserting user: " . $e->getMessage() . "\n";
        return false;
    }
}

// Insert sample data
$users = [
    ["Alice Smith", "alice@example.com", 25],
    ["Bob Johnson", "bob@example.com", 30],
    ["Carol Davis", "carol@example.com", 35]
];

foreach ($users as [$name, $email, $age]) {
    insertUser($pdo, $name, $email, $age);
}

// Select data
function getAllUsers($pdo) {
    try {
        $stmt = $pdo->query("SELECT * FROM users ORDER BY created_at");
        return $stmt->fetchAll();
    } catch (PDOException $e) {
        echo "Error fetching users: " . $e->getMessage() . "\n";
        return [];
    }
}

// Select with parameters
function getUsersByAge($pdo, $minAge) {
    try {
        $stmt = $pdo->prepare("SELECT * FROM users WHERE age >= ? ORDER BY age");
        $stmt->execute([$minAge]);
        return $stmt->fetchAll();
    } catch (PDOException $e) {
        echo "Error fetching users by age: " . $e->getMessage() . "\n";
        return [];
    }
}

// Display all users
echo "\nAll users:\n";
$allUsers = getAllUsers($pdo);
foreach ($allUsers as $user) {
    printf("ID: %d, Name: %s, Email: %s, Age: %d\n", 
           $user['id'], $user['name'], $user['email'], $user['age']);
}

// Display users by age filter
echo "\nUsers aged 30 or older:\n";
$olderUsers = getUsersByAge($pdo, 30);
foreach ($olderUsers as $user) {
    printf("Name: %s, Age: %d\n", $user['name'], $user['age']);
}

// Update data
function updateUserAge($pdo, $userId, $newAge) {
    try {
        $stmt = $pdo->prepare("UPDATE users SET age = ? WHERE id = ?");
        $stmt->execute([$newAge, $userId]);
        echo "User $userId age updated to $newAge\n";
        return $stmt->rowCount();
    } catch (PDOException $e) {
        echo "Error updating user: " . $e->getMessage() . "\n";
        return false;
    }
}

// Update a user
updateUserAge($pdo, 1, 26);

// Delete data
function deleteUser($pdo, $userId) {
    try {
        $stmt = $pdo->prepare("DELETE FROM users WHERE id = ?");
        $stmt->execute([$userId]);
        echo "User $userId deleted\n";
        return $stmt->rowCount();
    } catch (PDOException $e) {
        echo "Error deleting user: " . $e->getMessage() . "\n";
        return false;
    }
}

// Transactions
function transferUser($pdo, $fromId, $toId) {
    try {
        $pdo->beginTransaction();
        
        // Get user data
        $stmt = $pdo->prepare("SELECT * FROM users WHERE id = ?");
        $stmt->execute([$fromId]);
        $user = $stmt->fetch();
        
        if (!$user) {
            throw new Exception("User not found");
        }
        
        // Create backup entry
        $stmt = $pdo->prepare("INSERT INTO user_backup (original_id, name, email, age) VALUES (?, ?, ?, ?)");
        $stmt->execute([$user['id'], $user['name'], $user['email'], $user['age']]);
        
        // Delete original
        deleteUser($pdo, $fromId);
        
        $pdo->commit();
        echo "User transfer completed successfully\n";
    } catch (Exception $e) {
        $pdo->rollBack();
        echo "Transaction failed: " . $e->getMessage() . "\n";
    }
}

// Count records
function getUserCount($pdo) {
    try {
        $stmt = $pdo->query("SELECT COUNT(*) as count FROM users");
        $result = $stmt->fetch();
        return $result['count'];
    } catch (PDOException $e) {
        echo "Error counting users: " . $e->getMessage() . "\n";
        return 0;
    }
}

echo "\nTotal users: " . getUserCount($pdo) . "\n";

// Close connection (PDO automatically closes on script end)
$pdo = null;
echo "Database connection closed\n";
?>
```