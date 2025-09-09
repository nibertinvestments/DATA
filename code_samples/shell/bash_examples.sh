#!/bin/bash

# Comprehensive Shell/Bash scripting examples for ML/AI training

# Script metadata
SCRIPT_NAME="bash_examples.sh"
SCRIPT_VERSION="1.0"
SCRIPT_AUTHOR="AI Training Dataset"

echo "==============================================="
echo "  Shell/Bash Programming Examples"
echo "  Script: $SCRIPT_NAME v$SCRIPT_VERSION"
echo "  Author: $SCRIPT_AUTHOR"
echo "==============================================="
echo

# Basic variables and data types
echo "=== Variables and Data Types ==="
STRING_VAR="Hello, World!"
NUMBER_VAR=42
ARRAY_VAR=("apple" "banana" "cherry" "date")
READONLY_VAR="This cannot be changed"
readonly READONLY_VAR

echo "String variable: $STRING_VAR"
echo "Number variable: $NUMBER_VAR"
echo "Array variable: ${ARRAY_VAR[@]}"
echo "Array length: ${#ARRAY_VAR[@]}"
echo "First element: ${ARRAY_VAR[0]}"
echo "Last element: ${ARRAY_VAR[-1]}"
echo

# String manipulation
echo "=== String Manipulation ==="
TEXT="  Hello, Shell Programming!  "
echo "Original: '$TEXT'"
echo "Length: ${#TEXT}"
echo "Uppercase: ${TEXT^^}"
echo "Lowercase: ${TEXT,,}"
echo "Trimmed: $(echo "$TEXT" | xargs)"
echo "Substring (7-12): ${TEXT:7:5}"
echo "Replace 'Shell' with 'Bash': ${TEXT/Shell/Bash}"
echo

# Arithmetic operations
echo "=== Arithmetic Operations ==="
A=10
B=3

echo "A = $A, B = $B"
echo "Addition: $((A + B))"
echo "Subtraction: $((A - B))"
echo "Multiplication: $((A * B))"
echo "Division: $((A / B))"
echo "Modulo: $((A % B))"
echo "Power: $((A ** 2))"

# Using bc for floating point arithmetic
FLOAT_RESULT=$(echo "scale=2; $A / $B" | bc)
echo "Float division: $FLOAT_RESULT"
echo

# Conditional statements
echo "=== Conditional Statements ==="

# Function to check number properties
check_number() {
    local num=$1
    
    if [ $num -gt 0 ]; then
        echo "$num is positive"
    elif [ $num -lt 0 ]; then
        echo "$num is negative"
    else
        echo "$num is zero"
    fi
    
    if [ $((num % 2)) -eq 0 ]; then
        echo "$num is even"
    else
        echo "$num is odd"
    fi
}

# Test with different numbers
for number in -5 0 7 12; do
    echo "Checking $number:"
    check_number $number
    echo
done

# File and directory operations
echo "=== File and Directory Operations ==="

# Create temporary directory
TEMP_DIR="/tmp/shell_examples_$$"
mkdir -p "$TEMP_DIR"
echo "Created temporary directory: $TEMP_DIR"

# Create sample files
echo "Sample content 1" > "$TEMP_DIR/file1.txt"
echo "Sample content 2" > "$TEMP_DIR/file2.txt"
echo "Different content" > "$TEMP_DIR/file3.log"

echo "Files in $TEMP_DIR:"
ls -la "$TEMP_DIR"

# File testing
for file in "$TEMP_DIR"/*; do
    if [ -f "$file" ]; then
        echo "$(basename "$file") is a regular file"
        echo "  Size: $(stat -f%z "$file" 2>/dev/null || stat -c%s "$file") bytes"
        echo "  Last modified: $(stat -f%Sm "$file" 2>/dev/null || stat -c%y "$file")"
    fi
done
echo

# Loops
echo "=== Loop Examples ==="

# For loop with range
echo "For loop (1-5):"
for i in {1..5}; do
    echo "  Iteration $i"
done

# For loop with array
echo "For loop with fruits:"
FRUITS=("apple" "banana" "orange" "grape")
for fruit in "${FRUITS[@]}"; do
    echo "  Fruit: $fruit"
done

# While loop
echo "While loop (countdown):"
count=5
while [ $count -gt 0 ]; do
    echo "  Countdown: $count"
    ((count--))
done

# Until loop
echo "Until loop (count up):"
counter=1
until [ $counter -gt 3 ]; do
    echo "  Counter: $counter"
    ((counter++))
done
echo

# Functions
echo "=== Functions ==="

# Simple function
greet() {
    local name=${1:-"World"}
    echo "Hello, $name!"
}

# Function with multiple parameters
calculate() {
    local operation=$1
    local num1=$2
    local num2=$3
    
    case $operation in
        "add")
            echo $((num1 + num2))
            ;;
        "subtract")
            echo $((num1 - num2))
            ;;
        "multiply")
            echo $((num1 * num2))
            ;;
        "divide")
            if [ $num2 -ne 0 ]; then
                echo $((num1 / num2))
            else
                echo "Error: Division by zero"
                return 1
            fi
            ;;
        *)
            echo "Unknown operation: $operation"
            return 1
            ;;
    esac
}

# Function that returns multiple values via global variables
get_file_info() {
    local filename=$1
    
    if [ -f "$filename" ]; then
        FILE_SIZE=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename")
        FILE_LINES=$(wc -l < "$filename")
        FILE_WORDS=$(wc -w < "$filename")
        return 0
    else
        return 1
    fi
}

# Test functions
greet
greet "Alice"

echo "Calculator examples:"
echo "10 + 5 = $(calculate add 10 5)"
echo "10 - 5 = $(calculate subtract 10 5)"
echo "10 * 5 = $(calculate multiply 10 5)"
echo "10 / 5 = $(calculate divide 10 5)"

# Test file info function
if get_file_info "$TEMP_DIR/file1.txt"; then
    echo "File info for file1.txt:"
    echo "  Size: $FILE_SIZE bytes"
    echo "  Lines: $FILE_LINES"
    echo "  Words: $FILE_WORDS"
fi
echo

# Text processing
echo "=== Text Processing ==="

# Create sample data file
DATA_FILE="$TEMP_DIR/data.csv"
cat > "$DATA_FILE" << EOF
Name,Age,City,Salary
Alice,25,New York,50000
Bob,30,San Francisco,75000
Charlie,35,Chicago,60000
Diana,28,Boston,55000
Eve,32,Seattle,70000
EOF

echo "Sample CSV data:"
cat "$DATA_FILE"
echo

# Process the CSV file
echo "Processing CSV data:"

# Count lines
echo "Total lines: $(wc -l < "$DATA_FILE")"

# Extract specific columns (using awk)
echo "Names and ages:"
awk -F',' 'NR>1 {print $1 " is " $2 " years old"}' "$DATA_FILE"

# Filter data (using grep and awk)
echo "People older than 30:"
awk -F',' 'NR>1 && $2>30 {print $1 " (" $2 ")"}' "$DATA_FILE"

# Calculate average salary
echo "Average salary:"
awk -F',' 'NR>1 {sum+=$4; count++} END {print sum/count}' "$DATA_FILE"
echo

# Command substitution and pipes
echo "=== Command Substitution and Pipes ==="

# Get system information
echo "System information:"
echo "  Current user: $(whoami)"
echo "  Current directory: $(pwd)"
echo "  Current date: $(date)"
echo "  Disk usage of current directory:"
du -sh . 2>/dev/null || echo "  Cannot determine disk usage"

# Process list with pipes
echo "Top 5 processes by memory usage:"
ps aux 2>/dev/null | head -n 1  # Header
ps aux 2>/dev/null | tail -n +2 | sort -k4 -nr | head -n 5 || echo "  Cannot retrieve process information"
echo

# Error handling
echo "=== Error Handling ==="

# Function with error handling
safe_divide() {
    local num1=$1
    local num2=$2
    
    # Check if parameters are provided
    if [ $# -ne 2 ]; then
        echo "Error: Exactly two parameters required" >&2
        return 1
    fi
    
    # Check if parameters are numbers
    if ! [[ "$num1" =~ ^-?[0-9]+$ ]] || ! [[ "$num2" =~ ^-?[0-9]+$ ]]; then
        echo "Error: Parameters must be integers" >&2
        return 1
    fi
    
    # Check for division by zero
    if [ "$num2" -eq 0 ]; then
        echo "Error: Division by zero" >&2
        return 1
    fi
    
    echo $((num1 / num2))
    return 0
}

# Test error handling
echo "Testing error handling:"
safe_divide 10 2 && echo "Success: 10/2"
safe_divide 10 0 || echo "Caught error: division by zero"
safe_divide 10 || echo "Caught error: missing parameter"
safe_divide "abc" 2 || echo "Caught error: invalid parameter"
echo

# Pattern matching and case statements
echo "=== Pattern Matching ==="

identify_file_type() {
    local filename=$1
    
    case "$filename" in
        *.txt)
            echo "Text file"
            ;;
        *.log)
            echo "Log file"
            ;;
        *.csv)
            echo "CSV data file"
            ;;
        *.sh)
            echo "Shell script"
            ;;
        *.py)
            echo "Python script"
            ;;
        *)
            echo "Unknown file type"
            ;;
    esac
}

# Test pattern matching
for file in "document.txt" "error.log" "data.csv" "script.sh" "program.py" "unknown.xyz"; do
    echo "$file: $(identify_file_type "$file")"
done
echo

# Regular expressions with grep
echo "=== Regular Expressions ==="

# Create sample log file
LOG_FILE="$TEMP_DIR/sample.log"
cat > "$LOG_FILE" << EOF
2023-01-15 10:30:00 INFO User alice logged in
2023-01-15 10:35:12 ERROR Failed to connect to database
2023-01-15 10:40:25 INFO User bob logged in
2023-01-15 10:45:33 WARNING Disk space running low
2023-01-15 10:50:45 ERROR Connection timeout
2023-01-15 10:55:15 INFO User alice logged out
EOF

echo "Sample log file:"
cat "$LOG_FILE"
echo

echo "Filtering log entries:"
echo "ERROR messages:"
grep "ERROR" "$LOG_FILE"

echo "User activities:"
grep -E "(logged in|logged out)" "$LOG_FILE"

echo "Messages between 10:40 and 10:50:"
grep -E "10:4[0-9]:" "$LOG_FILE"
echo

# Advanced arrays and associative arrays
echo "=== Advanced Arrays ==="

# Indexed arrays
declare -a indexed_array=("first" "second" "third")
indexed_array[5]="sixth"  # Sparse array

echo "Indexed array:"
for i in "${!indexed_array[@]}"; do
    echo "  [$i] = ${indexed_array[$i]}"
done

# Associative arrays (Bash 4+)
if [ "${BASH_VERSION%%.*}" -ge 4 ]; then
    declare -A assoc_array
    assoc_array["name"]="John"
    assoc_array["age"]="30"
    assoc_array["city"]="New York"
    
    echo "Associative array:"
    for key in "${!assoc_array[@]}"; do
        echo "  $key = ${assoc_array[$key]}"
    done
else
    echo "Associative arrays require Bash 4 or later"
fi
echo

# Cleanup
echo "=== Cleanup ==="
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "Cleanup completed"
echo

# Summary
echo "=== Summary ==="
echo "Shell scripting examples completed successfully!"
echo "Topics covered:"
echo "  ✓ Variables and data types"
echo "  ✓ String manipulation"
echo "  ✓ Arithmetic operations"
echo "  ✓ Conditional statements"
echo "  ✓ File operations"
echo "  ✓ Loops"
echo "  ✓ Functions"
echo "  ✓ Text processing"
echo "  ✓ Command substitution"
echo "  ✓ Error handling"
echo "  ✓ Pattern matching"
echo "  ✓ Regular expressions"
echo "  ✓ Arrays"
echo

exit 0