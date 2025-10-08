# Advanced Data Visualization in R
# Comprehensive plotting and visualization techniques

# Create scatter plot with regression line
scatter_with_regression <- function(x, y, main = "Scatter Plot") {
  # Fit linear model
  model <- lm(y ~ x)
  
  # Create plot
  plot(x, y, 
       main = main,
       xlab = "X", 
       ylab = "Y",
       pch = 19,
       col = "blue")
  
  # Add regression line
  abline(model, col = "red", lwd = 2)
  
  # Add confidence interval
  predictions <- predict(model, interval = "confidence")
  lines(x[order(x)], predictions[order(x), "lwr"], 
        col = "gray", lty = 2)
  lines(x[order(x)], predictions[order(x), "upr"], 
        col = "gray", lty = 2)
  
  # Add legend
  legend("topleft", 
         legend = c("Data", "Regression", "95% CI"),
         col = c("blue", "red", "gray"),
         lty = c(NA, 1, 2),
         pch = c(19, NA, NA))
}

# Create histogram with density curve
histogram_with_density <- function(data, main = "Histogram") {
  # Create histogram
  hist(data, 
       prob = TRUE,
       main = main,
       xlab = "Value",
       col = "lightblue",
       border = "white")
  
  # Add density curve
  lines(density(data), col = "red", lwd = 2)
  
  # Add normal distribution overlay
  x_vals <- seq(min(data), max(data), length.out = 100)
  y_vals <- dnorm(x_vals, mean = mean(data), sd = sd(data))
  lines(x_vals, y_vals, col = "blue", lwd = 2, lty = 2)
  
  # Add legend
  legend("topright",
         legend = c("Density", "Normal"),
         col = c("red", "blue"),
         lty = c(1, 2),
         lwd = 2)
}

# Box plot with individual points
boxplot_with_points <- function(data, groups, main = "Box Plot") {
  # Create box plot
  boxplot(data ~ groups,
          main = main,
          xlab = "Group",
          ylab = "Value",
          col = "lightgreen",
          border = "darkgreen")
  
  # Add individual points with jitter
  for (i in 1:length(unique(groups))) {
    group_data <- data[groups == unique(groups)[i]]
    points(rep(i, length(group_data)) + runif(length(group_data), -0.1, 0.1),
           group_data,
           pch = 19,
           col = rgb(0, 0, 0, 0.3),
           cex = 0.5)
  }
}

# Heatmap visualization
create_heatmap <- function(matrix_data, main = "Heatmap") {
  # Create color palette
  colors <- colorRampPalette(c("blue", "white", "red"))(100)
  
  # Create heatmap
  image(1:ncol(matrix_data), 
        1:nrow(matrix_data),
        t(matrix_data),
        col = colors,
        main = main,
        xlab = "Column",
        ylab = "Row",
        axes = FALSE)
  
  # Add axes
  axis(1, at = 1:ncol(matrix_data), labels = colnames(matrix_data))
  axis(2, at = 1:nrow(matrix_data), labels = rownames(matrix_data))
  
  # Add color scale
  legend_image <- as.raster(matrix(colors, ncol = 1))
  rasterImage(legend_image, 
              ncol(matrix_data) + 0.5, 1,
              ncol(matrix_data) + 1, nrow(matrix_data))
}

# Time series plot with trend
time_series_plot <- function(dates, values, main = "Time Series") {
  # Create plot
  plot(dates, values,
       type = "l",
       main = main,
       xlab = "Date",
       ylab = "Value",
       col = "blue",
       lwd = 2)
  
  # Add trend line
  time_numeric <- as.numeric(dates)
  trend_model <- lm(values ~ time_numeric)
  trend_line <- predict(trend_model)
  lines(dates, trend_line, col = "red", lwd = 2, lty = 2)
  
  # Add grid
  grid()
  
  # Add legend
  legend("topleft",
         legend = c("Actual", "Trend"),
         col = c("blue", "red"),
         lty = c(1, 2),
         lwd = 2)
}

cat("Advanced Visualization Techniques in R\n")
cat("======================================\n")
