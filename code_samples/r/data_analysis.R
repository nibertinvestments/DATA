# R Programming Examples for Data Science and ML

# Basic data types and structures
print("=== R Programming Examples ===")

# Vectors
numeric_vector <- c(1, 2, 3, 4, 5)
character_vector <- c("apple", "banana", "cherry")
logical_vector <- c(TRUE, FALSE, TRUE, TRUE)

cat("Numeric vector:", numeric_vector, "\n")
cat("Character vector:", character_vector, "\n")
cat("Logical vector:", logical_vector, "\n\n")

# Matrices
matrix_data <- matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)
print("Matrix:")
print(matrix_data)

# Data frames (most important data structure in R)
df <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana", "Eve"),
  age = c(25, 30, 35, 28, 32),
  salary = c(50000, 60000, 75000, 55000, 68000),
  department = c("IT", "Finance", "IT", "HR", "Finance"),
  stringsAsFactors = FALSE
)

print("Data Frame:")
print(df)

# Lists (can contain different data types)
my_list <- list(
  numbers = 1:5,
  strings = c("a", "b", "c"),
  matrix = matrix(1:6, nrow = 2),
  dataframe = df[1:3, ]
)

print("List structure:")
str(my_list)

# Statistical functions
print("\n=== Statistical Analysis ===")

# Basic statistics
numbers <- c(23, 45, 67, 89, 12, 34, 56, 78, 90, 11)
cat("Numbers:", numbers, "\n")
cat("Mean:", mean(numbers), "\n")
cat("Median:", median(numbers), "\n")
cat("Standard deviation:", sd(numbers), "\n")
cat("Variance:", var(numbers), "\n")
cat("Range:", range(numbers), "\n")
cat("Quantiles:", quantile(numbers), "\n\n")

# Data manipulation with built-in functions
print("=== Data Manipulation ===")

# Subset data
subset_df <- subset(df, age > 30)
print("Employees older than 30:")
print(subset_df)

# Aggregate data
agg_salary <- aggregate(salary ~ department, data = df, FUN = mean)
print("Average salary by department:")
print(agg_salary)

# Apply functions
apply_result <- apply(matrix_data, 2, sum)  # Sum by columns
cat("Column sums of matrix:", apply_result, "\n")

# Custom functions
calculate_bmi <- function(weight, height) {
  bmi <- weight / (height^2)
  status <- ifelse(bmi < 18.5, "Underweight",
                   ifelse(bmi < 25, "Normal",
                          ifelse(bmi < 30, "Overweight", "Obese")))
  return(list(bmi = bmi, status = status))
}

# Example usage
weight <- 70  # kg
height <- 1.75  # meters
bmi_result <- calculate_bmi(weight, height)
cat("BMI:", round(bmi_result$bmi, 2), "- Status:", bmi_result$status, "\n\n")

# Data visualization (if libraries are available)
print("=== Data Visualization ===")

# Basic plotting
png("histogram.png", width = 800, height = 600)
hist(numbers, 
     main = "Histogram of Numbers", 
     xlab = "Values", 
     ylab = "Frequency",
     col = "lightblue",
     border = "black")
dev.off()

png("boxplot.png", width = 800, height = 600)
boxplot(salary ~ department, 
        data = df,
        main = "Salary Distribution by Department",
        xlab = "Department",
        ylab = "Salary",
        col = c("lightblue", "lightgreen", "lightcoral"))
dev.off()

cat("Plots saved as histogram.png and boxplot.png\n\n")

# String manipulation
print("=== String Processing ===")

text <- "This is a sample text for string manipulation"
cat("Original text:", text, "\n")
cat("Upper case:", toupper(text), "\n")
cat("Lower case:", tolower(text), "\n")
cat("Number of characters:", nchar(text), "\n")

# Split and combine strings
words <- strsplit(text, " ")[[1]]
cat("Split into words:", words, "\n")
combined <- paste(words[1:3], collapse = "-")
cat("Combined first 3 words:", combined, "\n\n")

# Regular expressions
email_pattern <- "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
emails <- c("user@example.com", "invalid-email", "test@domain.org", "not_an_email")
valid_emails <- grep(email_pattern, emails, value = TRUE)
cat("Valid emails:", valid_emails, "\n\n")

# Date and time handling
print("=== Date and Time ===")

current_date <- Sys.Date()
current_time <- Sys.time()
cat("Current date:", as.character(current_date), "\n")
cat("Current time:", as.character(current_time), "\n")

# Create date sequence
date_sequence <- seq(from = as.Date("2023-01-01"), 
                     to = as.Date("2023-01-10"), 
                     by = "day")
cat("Date sequence:", as.character(date_sequence), "\n\n")

# Control structures
print("=== Control Structures ===")

# For loop
cat("For loop (squares of 1-5):")
for (i in 1:5) {
  cat(" ", i^2)
}
cat("\n")

# While loop
counter <- 1
cat("While loop (factorial of 5):")
factorial <- 1
while (counter <= 5) {
  factorial <- factorial * counter
  counter <- counter + 1
}
cat(" ", factorial, "\n")

# If-else statements
for (num in c(5, 12, 7, 20)) {
  if (num %% 2 == 0) {
    cat(num, "is even\n")
  } else {
    cat(num, "is odd\n")
  }
}

# Statistical modeling
print("\n=== Statistical Modeling ===")

# Generate sample data
set.seed(123)  # For reproducibility
x <- rnorm(100, mean = 50, sd = 10)
y <- 2 * x + rnorm(100, mean = 0, sd = 5)

# Linear regression
model <- lm(y ~ x)
summary_model <- summary(model)
cat("Linear regression summary:\n")
cat("R-squared:", summary_model$r.squared, "\n")
cat("Coefficients:\n")
print(summary_model$coefficients)

# Correlation
correlation <- cor(x, y)
cat("Correlation between x and y:", correlation, "\n\n")

# Advanced data structures and operations
print("=== Advanced Operations ===")

# Working with factors
departments <- factor(c("IT", "Finance", "HR", "IT", "Finance", "HR"))
cat("Factor levels:", levels(departments), "\n")
cat("Factor summary:\n")
print(summary(departments))

# Working with missing data
data_with_na <- c(1, 2, NA, 4, 5, NA, 7, 8)
cat("Data with NA:", data_with_na, "\n")
cat("Mean (with NA):", mean(data_with_na), "\n")
cat("Mean (NA removed):", mean(data_with_na, na.rm = TRUE), "\n")

# Complete cases
complete_data <- data_with_na[complete.cases(data_with_na)]
cat("Complete cases:", complete_data, "\n\n")

# Functional programming in R
print("=== Functional Programming ===")

# lapply (returns list)
list_result <- lapply(list(1:3, 4:6, 7:9), sum)
cat("lapply result (sums):", unlist(list_result), "\n")

# sapply (returns vector/matrix)
vector_result <- sapply(list(1:3, 4:6, 7:9), mean)
cat("sapply result (means):", vector_result, "\n")

# mapply (multivariate apply)
multivariate_result <- mapply(function(x, y) x + y, 1:3, 4:6)
cat("mapply result (pairwise sums):", multivariate_result, "\n\n")

# Custom higher-order functions
apply_twice <- function(f, x) {
  f(f(x))
}

square <- function(x) x^2
result <- apply_twice(square, 2)  # ((2^2)^2) = 16
cat("Apply square function twice to 2:", result, "\n")

# Creating and working with time series
print("=== Time Series ===")

# Create time series data
ts_data <- ts(1:24, start = c(2023, 1), frequency = 12)
cat("Time series (monthly data for 2 years):\n")
print(ts_data)

# Time series operations
cat("Start of time series:", start(ts_data), "\n")
cat("End of time series:", end(ts_data), "\n")
cat("Frequency:", frequency(ts_data), "\n\n")

# Matrix operations
print("=== Matrix Operations ===")

A <- matrix(c(1, 2, 3, 4), nrow = 2)
B <- matrix(c(5, 6, 7, 8), nrow = 2)

cat("Matrix A:\n")
print(A)
cat("Matrix B:\n")
print(B)

cat("Matrix multiplication A %*% B:\n")
print(A %*% B)

cat("Element-wise multiplication A * B:\n")
print(A * B)

cat("Transpose of A:\n")
print(t(A))

cat("Determinant of A:", det(A), "\n")
cat("Inverse of A:\n")
print(solve(A))

print("\n=== Summary ===")
cat("R programming examples completed successfully!\n")
cat("Covered topics:\n")
cat("- Basic data types and structures\n")
cat("- Statistical analysis\n")
cat("- Data manipulation\n")
cat("- String processing\n")
cat("- Date/time handling\n")
cat("- Control structures\n")
cat("- Statistical modeling\n")
cat("- Functional programming\n")
cat("- Time series\n")
cat("- Matrix operations\n")