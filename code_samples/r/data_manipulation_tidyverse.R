# Data Manipulation with Tidyverse Concepts
# Advanced data wrangling and transformation patterns

# Data frame manipulation
manipulate_dataframe <- function(df) {
  # Sort by column
  df_sorted <- df[order(df$value, decreasing = TRUE), ]
  
  # Filter rows
  df_filtered <- df[df$value > 50, ]
  
  # Add new column
  df$squared <- df$value^2
  
  # Aggregate by group
  aggregated <- aggregate(value ~ category, data = df, FUN = mean)
  
  list(
    sorted = df_sorted,
    filtered = df_filtered,
    with_new_col = df,
    aggregated = aggregated
  )
}

# Reshape data
reshape_data <- function(df) {
  # Wide to long format
  long_format <- reshape(df, 
                        varying = list(names(df)[-1]),
                        v.names = "value",
                        timevar = "variable",
                        times = names(df)[-1],
                        direction = "long")
  
  return(long_format)
}

# Join operations
join_dataframes <- function(df1, df2, by_col) {
  # Inner join
  inner_joined <- merge(df1, df2, by = by_col)
  
  # Left join
  left_joined <- merge(df1, df2, by = by_col, all.x = TRUE)
  
  # Right join  
  right_joined <- merge(df1, df2, by = by_col, all.y = TRUE)
  
  # Full outer join
  full_joined <- merge(df1, df2, by = by_col, all = TRUE)
  
  list(
    inner = inner_joined,
    left = left_joined,
    right = right_joined,
    full = full_joined
  )
}

# Group operations
group_operations <- function(df, group_col, value_col) {
  # Split-apply-combine pattern
  grouped <- split(df, df[[group_col]])
  
  # Calculate statistics per group
  results <- lapply(grouped, function(g) {
    list(
      mean = mean(g[[value_col]]),
      median = median(g[[value_col]]),
      sd = sd(g[[value_col]]),
      min = min(g[[value_col]]),
      max = max(g[[value_col]]),
      count = nrow(g)
    )
  })
  
  return(results)
}

# Window functions
window_functions <- function(df, value_col) {
  # Cumulative sum
  df$cumsum <- cumsum(df[[value_col]])
  
  # Moving average (3-period)
  n <- length(df[[value_col]])
  df$moving_avg <- c(rep(NA, 2), sapply(3:n, function(i) {
    mean(df[[value_col]][(i-2):i])
  }))
  
  # Rank
  df$rank <- rank(df[[value_col]])
  
  # Percent rank
  df$percent_rank <- (rank(df[[value_col]]) - 1) / (length(df[[value_col]]) - 1)
  
  return(df)
}

cat("Data Manipulation Patterns in R\n")
cat("================================\n")
