# Statistical Modeling - Regression Analysis in R
# Comprehensive linear and non-linear regression implementations

# Linear Regression Model
linear_model <- function(x, y) {
  # Fit linear model
  model <- lm(y ~ x)
  
  # Get summary statistics
  summary_stats <- summary(model)
  
  # Calculate predictions
  predictions <- predict(model, newdata = data.frame(x = x))
  
  # Calculate residuals
  residuals <- y - predictions
  
  # Return comprehensive results
  list(
    model = model,
    coefficients = coef(model),
    r_squared = summary_stats$r.squared,
    predictions = predictions,
    residuals = residuals
  )
}

# Multiple Regression
multiple_regression <- function(data, formula_str) {
  # Parse formula
  formula_obj <- as.formula(formula_str)
  
  # Fit model
  model <- lm(formula_obj, data = data)
  
  # Get diagnostics
  diagnostics <- list(
    summary = summary(model),
    anova = anova(model),
    residuals = resid(model),
    fitted = fitted(model)
  )
  
  return(diagnostics)
}

# Polynomial Regression
polynomial_regression <- function(x, y, degree = 2) {
  # Create polynomial features
  x_poly <- poly(x, degree)
  
  # Fit polynomial model
  model <- lm(y ~ x_poly)
  
  # Generate predictions
  predictions <- predict(model)
  
  list(
    model = model,
    predictions = predictions,
    degree = degree
  )
}

# Logistic Regression
logistic_regression <- function(x, y) {
  # Fit logistic model
  model <- glm(y ~ x, family = binomial(link = "logit"))
  
  # Get predictions (probabilities)
  predictions <- predict(model, type = "response")
  
  # Classification (threshold 0.5)
  classifications <- ifelse(predictions > 0.5, 1, 0)
  
  list(
    model = model,
    probabilities = predictions,
    classifications = classifications
  )
}

# Demonstration
cat("Statistical Modeling in R\n")
cat("=========================\n\n")

# Sample data
set.seed(42)
x <- seq(1, 100, by = 1)
y <- 2 * x + rnorm(100, mean = 0, sd = 10)

# Run linear regression
result <- linear_model(x, y)
cat("Linear Model R-squared:", result$r_squared, "\n")
cat("Coefficients:", result$coefficients, "\n")
