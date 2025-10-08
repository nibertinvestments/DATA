# Time Series Analysis in R
# Comprehensive time series modeling and forecasting

# ARIMA Model Implementation
arima_model <- function(ts_data, p = 1, d = 1, q = 1) {
  # Fit ARIMA model
  model <- arima(ts_data, order = c(p, d, q))
  
  # Get fitted values
  fitted_values <- ts_data - residuals(model)
  
  # Forecast
  forecast_horizon <- 10
  forecasts <- predict(model, n.ahead = forecast_horizon)
  
  list(
    model = model,
    fitted = fitted_values,
    forecasts = forecasts$pred,
    forecast_se = forecasts$se,
    aic = AIC(model),
    bic = BIC(model)
  )
}

# Moving Average Smoothing
moving_average <- function(x, window_size = 3) {
  n <- length(x)
  result <- rep(NA, n)
  
  for (i in window_size:n) {
    result[i] <- mean(x[(i-window_size+1):i])
  }
  
  return(result)
}

# Exponential Smoothing
exponential_smoothing <- function(x, alpha = 0.3) {
  n <- length(x)
  smoothed <- numeric(n)
  smoothed[1] <- x[1]
  
  for (i in 2:n) {
    smoothed[i] <- alpha * x[i] + (1 - alpha) * smoothed[i-1]
  }
  
  return(smoothed)
}

# Seasonal Decomposition
seasonal_decompose <- function(ts_data, frequency) {
  # Create time series object
  ts_obj <- ts(ts_data, frequency = frequency)
  
  # Decompose
  decomposed <- decompose(ts_obj)
  
  list(
    trend = as.numeric(decomposed$trend),
    seasonal = as.numeric(decomposed$seasonal),
    random = as.numeric(decomposed$random)
  )
}

# Autocorrelation Function
calculate_acf <- function(x, max_lag = 20) {
  n <- length(x)
  x_mean <- mean(x)
  
  acf_values <- numeric(max_lag + 1)
  
  # Calculate ACF
  for (lag in 0:max_lag) {
    numerator <- sum((x[1:(n-lag)] - x_mean) * (x[(1+lag):n] - x_mean))
    denominator <- sum((x - x_mean)^2)
    acf_values[lag + 1] <- numerator / denominator
  }
  
  return(acf_values)
}

# Dickey-Fuller Test for Stationarity
dickey_fuller_test <- function(x) {
  n <- length(x)
  
  # Lag the series
  x_lag <- c(NA, x[1:(n-1)])
  x_diff <- diff(x)
  x_lag_diff <- x_lag[2:n]
  
  # Run regression
  model <- lm(x_diff ~ x_lag_diff)
  
  # Calculate test statistic
  coef_estimate <- coef(model)[2]
  std_error <- summary(model)$coefficients[2, 2]
  t_stat <- coef_estimate / std_error
  
  list(
    statistic = t_stat,
    coefficient = coef_estimate,
    std_error = std_error
  )
}

cat("Time Series Analysis in R\n")
cat("==========================\n")
