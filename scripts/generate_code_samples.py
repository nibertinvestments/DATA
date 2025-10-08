#!/usr/bin/env python3
"""
Generate diverse, verified code samples for AI/ML training
Creates unique, syntactically correct code samples across multiple languages
"""

import os
from pathlib import Path
import subprocess
import sys

# Base path for code samples
CODE_SAMPLES_PATH = Path("/home/runner/work/DATA/DATA/code_samples")

def generate_r_samples():
    """Generate R code samples for statistical computing and data science"""
    
    samples = {
        "statistical_modeling_regression.R": '''# Statistical Modeling - Regression Analysis in R
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
cat("Statistical Modeling in R\\n")
cat("=========================\\n\\n")

# Sample data
set.seed(42)
x <- seq(1, 100, by = 1)
y <- 2 * x + rnorm(100, mean = 0, sd = 10)

# Run linear regression
result <- linear_model(x, y)
cat("Linear Model R-squared:", result$r_squared, "\\n")
cat("Coefficients:", result$coefficients, "\\n")
''',

        "data_manipulation_tidyverse.R": '''# Data Manipulation with Tidyverse Concepts
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

cat("Data Manipulation Patterns in R\\n")
cat("================================\\n")
''',

        "machine_learning_classification.R": '''# Machine Learning - Classification Algorithms in R
# Implementation of common classification methods

# Decision Tree Classifier (manual implementation)
decision_tree_classifier <- function(x, y, max_depth = 3) {
  # Calculate Gini impurity
  gini_impurity <- function(classes) {
    if (length(classes) == 0) return(0)
    probs <- table(classes) / length(classes)
    return(1 - sum(probs^2))
  }
  
  # Find best split
  find_best_split <- function(x, y) {
    best_gini <- Inf
    best_feature <- NULL
    best_threshold <- NULL
    
    for (feature in 1:ncol(x)) {
      thresholds <- sort(unique(x[, feature]))
      for (threshold in thresholds) {
        left_mask <- x[, feature] <= threshold
        right_mask <- !left_mask
        
        if (sum(left_mask) == 0 || sum(right_mask) == 0) next
        
        left_gini <- gini_impurity(y[left_mask])
        right_gini <- gini_impurity(y[right_mask])
        
        weighted_gini <- (sum(left_mask) * left_gini + 
                         sum(right_mask) * right_gini) / length(y)
        
        if (weighted_gini < best_gini) {
          best_gini <- weighted_gini
          best_feature <- feature
          best_threshold <- threshold
        }
      }
    }
    
    list(feature = best_feature, threshold = best_threshold, gini = best_gini)
  }
  
  # Build tree recursively
  build_tree <- function(x, y, depth = 0) {
    # Stopping conditions
    if (depth >= max_depth || length(unique(y)) == 1 || nrow(x) < 2) {
      return(list(type = "leaf", class = names(which.max(table(y)))))
    }
    
    split <- find_best_split(x, y)
    if (is.null(split$feature)) {
      return(list(type = "leaf", class = names(which.max(table(y)))))
    }
    
    left_mask <- x[, split$feature] <= split$threshold
    right_mask <- !left_mask
    
    left_tree <- build_tree(x[left_mask, , drop = FALSE], 
                           y[left_mask], depth + 1)
    right_tree <- build_tree(x[right_mask, , drop = FALSE], 
                            y[right_mask], depth + 1)
    
    list(
      type = "node",
      feature = split$feature,
      threshold = split$threshold,
      left = left_tree,
      right = right_tree
    )
  }
  
  tree <- build_tree(x, y)
  return(tree)
}

# K-Nearest Neighbors
knn_classifier <- function(x_train, y_train, x_test, k = 3) {
  # Euclidean distance
  euclidean_distance <- function(a, b) {
    sqrt(sum((a - b)^2))
  }
  
  # Predict single instance
  predict_single <- function(x) {
    # Calculate distances to all training points
    distances <- apply(x_train, 1, function(train_point) {
      euclidean_distance(x, train_point)
    })
    
    # Get k nearest neighbors
    nearest_indices <- order(distances)[1:k]
    nearest_labels <- y_train[nearest_indices]
    
    # Majority vote
    names(which.max(table(nearest_labels)))
  }
  
  # Predict all test instances
  predictions <- apply(x_test, 1, predict_single)
  return(predictions)
}

# Naive Bayes Classifier
naive_bayes_classifier <- function(x_train, y_train) {
  # Calculate class priors
  classes <- unique(y_train)
  priors <- sapply(classes, function(c) sum(y_train == c) / length(y_train))
  
  # Calculate feature probabilities for each class
  feature_probs <- lapply(classes, function(c) {
    class_data <- x_train[y_train == c, , drop = FALSE]
    list(
      mean = colMeans(class_data),
      sd = apply(class_data, 2, sd)
    )
  })
  names(feature_probs) <- classes
  
  # Prediction function
  predict_fn <- function(x_test) {
    # Gaussian probability density
    gaussian_prob <- function(x, mean, sd) {
      exp(-((x - mean)^2) / (2 * sd^2)) / (sqrt(2 * pi) * sd)
    }
    
    # Predict single instance
    predict_single <- function(x) {
      posteriors <- sapply(classes, function(c) {
        prior <- priors[c]
        likelihood <- prod(gaussian_prob(
          x, 
          feature_probs[[c]]$mean, 
          feature_probs[[c]]$sd
        ))
        prior * likelihood
      })
      names(which.max(posteriors))
    }
    
    apply(x_test, 1, predict_single)
  }
  
  list(
    classes = classes,
    priors = priors,
    feature_probs = feature_probs,
    predict = predict_fn
  )
}

cat("Machine Learning Classification in R\\n")
cat("====================================\\n")
''',

        "visualization_advanced_plots.R": '''# Advanced Data Visualization in R
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

cat("Advanced Visualization Techniques in R\\n")
cat("======================================\\n")
''',

        "time_series_analysis.R": '''# Time Series Analysis in R
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

cat("Time Series Analysis in R\\n")
cat("==========================\\n")
'''
    }
    
    return samples

def generate_kotlin_samples():
    """Generate Kotlin code samples without external dependencies"""
    
    samples = {
        "algorithms_graph_advanced.kt": '''// Advanced Graph Algorithms in Kotlin
// Comprehensive graph theory implementations

data class Graph(val vertices: Int) {
    private val adjacencyList = Array(vertices) { mutableListOf<Pair<Int, Int>>() }
    
    fun addEdge(from: Int, to: Int, weight: Int = 1) {
        adjacencyList[from].add(Pair(to, weight))
    }
    
    fun getNeighbors(vertex: Int): List<Pair<Int, Int>> = adjacencyList[vertex]
}

// Dijkstra's Shortest Path Algorithm
class DijkstraShortestPath {
    fun findShortestPath(graph: Graph, start: Int, vertices: Int): IntArray {
        val distances = IntArray(vertices) { Int.MAX_VALUE }
        val visited = BooleanArray(vertices)
        distances[start] = 0
        
        repeat(vertices) {
            val u = findMinDistance(distances, visited)
            if (u == -1) return@repeat
            
            visited[u] = true
            
            for ((v, weight) in graph.getNeighbors(u)) {
                if (!visited[v] && distances[u] != Int.MAX_VALUE) {
                    val newDist = distances[u] + weight
                    if (newDist < distances[v]) {
                        distances[v] = newDist
                    }
                }
            }
        }
        
        return distances
    }
    
    private fun findMinDistance(distances: IntArray, visited: BooleanArray): Int {
        var minDist = Int.MAX_VALUE
        var minIndex = -1
        
        for (i in distances.indices) {
            if (!visited[i] && distances[i] < minDist) {
                minDist = distances[i]
                minIndex = i
            }
        }
        
        return minIndex
    }
}

// Bellman-Ford Algorithm
class BellmanFordAlgorithm {
    fun findShortestPaths(graph: Graph, start: Int, vertices: Int, edges: List<Triple<Int, Int, Int>>): IntArray? {
        val distances = IntArray(vertices) { Int.MAX_VALUE }
        distances[start] = 0
        
        // Relax edges
        repeat(vertices - 1) {
            for ((u, v, weight) in edges) {
                if (distances[u] != Int.MAX_VALUE && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight
                }
            }
        }
        
        // Check for negative cycles
        for ((u, v, weight) in edges) {
            if (distances[u] != Int.MAX_VALUE && distances[u] + weight < distances[v]) {
                return null // Negative cycle detected
            }
        }
        
        return distances
    }
}

// Floyd-Warshall All-Pairs Shortest Path
class FloydWarshallAlgorithm {
    fun allPairsShortestPath(vertices: Int, edges: List<Triple<Int, Int, Int>>): Array<IntArray> {
        val dist = Array(vertices) { IntArray(vertices) { Int.MAX_VALUE / 2 } }
        
        // Initialize diagonal
        for (i in 0 until vertices) {
            dist[i][i] = 0
        }
        
        // Add edges
        for ((u, v, weight) in edges) {
            dist[u][v] = weight
        }
        
        // Floyd-Warshall algorithm
        for (k in 0 until vertices) {
            for (i in 0 until vertices) {
                for (j in 0 until vertices) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j]
                    }
                }
            }
        }
        
        return dist
    }
}

// Topological Sort
class TopologicalSort {
    fun sort(graph: Graph, vertices: Int): List<Int>? {
        val inDegree = IntArray(vertices)
        val result = mutableListOf<Int>()
        
        // Calculate in-degrees
        for (u in 0 until vertices) {
            for ((v, _) in graph.getNeighbors(u)) {
                inDegree[v]++
            }
        }
        
        // Queue for vertices with in-degree 0
        val queue = ArrayDeque<Int>()
        for (i in inDegree.indices) {
            if (inDegree[i] == 0) {
                queue.add(i)
            }
        }
        
        // Process vertices
        while (queue.isNotEmpty()) {
            val u = queue.removeFirst()
            result.add(u)
            
            for ((v, _) in graph.getNeighbors(u)) {
                inDegree[v]--
                if (inDegree[v] == 0) {
                    queue.add(v)
                }
            }
        }
        
        return if (result.size == vertices) result else null // Cycle detected
    }
}

fun main() {
    println("Advanced Graph Algorithms in Kotlin")
    println("====================================")
    
    // Create sample graph
    val graph = Graph(5)
    graph.addEdge(0, 1, 4)
    graph.addEdge(0, 2, 1)
    graph.addEdge(2, 1, 2)
    graph.addEdge(1, 3, 1)
    graph.addEdge(2, 3, 5)
    graph.addEdge(3, 4, 3)
    
    // Test Dijkstra
    val dijkstra = DijkstraShortestPath()
    val distances = dijkstra.findShortestPath(graph, 0, 5)
    println("Shortest paths from vertex 0: ${distances.contentToString()}")
}
''',

        "data_structures_advanced_trees.kt": '''// Advanced Tree Data Structures in Kotlin
// Comprehensive tree implementations

// AVL Tree (Self-Balancing Binary Search Tree)
class AVLTree<T : Comparable<T>> {
    private data class Node<T>(
        var value: T,
        var height: Int = 1,
        var left: Node<T>? = null,
        var right: Node<T>? = null
    )
    
    private var root: Node<T>? = null
    
    private fun height(node: Node<T>?): Int = node?.height ?: 0
    
    private fun updateHeight(node: Node<T>) {
        node.height = 1 + maxOf(height(node.left), height(node.right))
    }
    
    private fun balanceFactor(node: Node<T>): Int {
        return height(node.left) - height(node.right)
    }
    
    private fun rotateRight(y: Node<T>): Node<T> {
        val x = y.left!!
        val T2 = x.right
        
        x.right = y
        y.left = T2
        
        updateHeight(y)
        updateHeight(x)
        
        return x
    }
    
    private fun rotateLeft(x: Node<T>): Node<T> {
        val y = x.right!!
        val T2 = y.left
        
        y.left = x
        x.right = T2
        
        updateHeight(x)
        updateHeight(y)
        
        return y
    }
    
    private fun balance(node: Node<T>): Node<T> {
        updateHeight(node)
        val balance = balanceFactor(node)
        
        // Left heavy
        if (balance > 1) {
            if (balanceFactor(node.left!!) < 0) {
                node.left = rotateLeft(node.left!!)
            }
            return rotateRight(node)
        }
        
        // Right heavy
        if (balance < -1) {
            if (balanceFactor(node.right!!) > 0) {
                node.right = rotateRight(node.right!!)
            }
            return rotateLeft(node)
        }
        
        return node
    }
    
    fun insert(value: T) {
        root = insertHelper(root, value)
    }
    
    private fun insertHelper(node: Node<T>?, value: T): Node<T> {
        if (node == null) return Node(value)
        
        when {
            value < node.value -> node.left = insertHelper(node.left, value)
            value > node.value -> node.right = insertHelper(node.right, value)
            else -> return node
        }
        
        return balance(node)
    }
    
    fun search(value: T): Boolean {
        var current = root
        while (current != null) {
            when {
                value < current.value -> current = current.left
                value > current.value -> current = current.right
                else -> return true
            }
        }
        return false
    }
    
    fun inorderTraversal(): List<T> {
        val result = mutableListOf<T>()
        inorderHelper(root, result)
        return result
    }
    
    private fun inorderHelper(node: Node<T>?, result: MutableList<T>) {
        if (node != null) {
            inorderHelper(node.left, result)
            result.add(node.value)
            inorderHelper(node.right, result)
        }
    }
}

// Red-Black Tree
class RedBlackTree<T : Comparable<T>> {
    private enum class Color { RED, BLACK }
    
    private data class Node<T>(
        var value: T,
        var color: Color = Color.RED,
        var left: Node<T>? = null,
        var right: Node<T>? = null,
        var parent: Node<T>? = null
    )
    
    private var root: Node<T>? = null
    
    fun insert(value: T) {
        val newNode = Node(value)
        
        if (root == null) {
            root = newNode
            newNode.color = Color.BLACK
            return
        }
        
        // Standard BST insert
        var current = root
        var parent: Node<T>? = null
        
        while (current != null) {
            parent = current
            current = if (value < current.value) current.left else current.right
        }
        
        newNode.parent = parent
        if (value < parent!!.value) {
            parent.left = newNode
        } else {
            parent.right = newNode
        }
        
        // Fix Red-Black tree properties
        fixInsert(newNode)
    }
    
    private fun fixInsert(node: Node<T>) {
        var current = node
        
        while (current != root && current.parent?.color == Color.RED) {
            val parent = current.parent!!
            val grandparent = parent.parent!!
            
            if (parent == grandparent.left) {
                val uncle = grandparent.right
                
                if (uncle?.color == Color.RED) {
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    current = grandparent
                } else {
                    if (current == parent.right) {
                        current = parent
                        rotateLeft(current)
                    }
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    rotateRight(grandparent)
                }
            } else {
                val uncle = grandparent.left
                
                if (uncle?.color == Color.RED) {
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    current = grandparent
                } else {
                    if (current == parent.left) {
                        current = parent
                        rotateRight(current)
                    }
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    rotateLeft(grandparent)
                }
            }
        }
        
        root?.color = Color.BLACK
    }
    
    private fun rotateLeft(node: Node<T>) {
        val rightChild = node.right!!
        node.right = rightChild.left
        
        if (rightChild.left != null) {
            rightChild.left!!.parent = node
        }
        
        rightChild.parent = node.parent
        
        when {
            node.parent == null -> root = rightChild
            node == node.parent!!.left -> node.parent!!.left = rightChild
            else -> node.parent!!.right = rightChild
        }
        
        rightChild.left = node
        node.parent = rightChild
    }
    
    private fun rotateRight(node: Node<T>) {
        val leftChild = node.left!!
        node.left = leftChild.right
        
        if (leftChild.right != null) {
            leftChild.right!!.parent = node
        }
        
        leftChild.parent = node.parent
        
        when {
            node.parent == null -> root = leftChild
            node == node.parent!!.right -> node.parent!!.right = leftChild
            else -> node.parent!!.left = leftChild
        }
        
        leftChild.right = node
        node.parent = leftChild
    }
    
    fun search(value: T): Boolean {
        var current = root
        while (current != null) {
            when {
                value < current.value -> current = current.left
                value > current.value -> current = current.right
                else -> return true
            }
        }
        return false
    }
}

// B-Tree
class BTree<T : Comparable<T>>(private val degree: Int = 3) {
    private data class Node<T>(
        val keys: MutableList<T> = mutableListOf(),
        val children: MutableList<Node<T>> = mutableListOf(),
        var isLeaf: Boolean = true
    )
    
    private var root = Node<T>()
    
    fun insert(key: T) {
        if (root.keys.size == 2 * degree - 1) {
            val newRoot = Node<T>()
            newRoot.isLeaf = false
            newRoot.children.add(root)
            splitChild(newRoot, 0)
            root = newRoot
        }
        insertNonFull(root, key)
    }
    
    private fun insertNonFull(node: Node<T>, key: T) {
        var i = node.keys.size - 1
        
        if (node.isLeaf) {
            node.keys.add(key)
            while (i >= 0 && key < node.keys[i]) {
                node.keys[i + 1] = node.keys[i]
                i--
            }
            node.keys[i + 1] = key
        } else {
            while (i >= 0 && key < node.keys[i]) {
                i--
            }
            i++
            
            if (node.children[i].keys.size == 2 * degree - 1) {
                splitChild(node, i)
                if (key > node.keys[i]) {
                    i++
                }
            }
            insertNonFull(node.children[i], key)
        }
    }
    
    private fun splitChild(parent: Node<T>, index: Int) {
        val fullChild = parent.children[index]
        val newChild = Node<T>()
        newChild.isLeaf = fullChild.isLeaf
        
        val midIndex = degree - 1
        
        // Move half of keys to new node
        for (j in 0 until degree - 1) {
            newChild.keys.add(fullChild.keys[midIndex + 1 + j])
        }
        
        // Move children if not leaf
        if (!fullChild.isLeaf) {
            for (j in 0 until degree) {
                newChild.children.add(fullChild.children[midIndex + 1 + j])
            }
        }
        
        // Update parent
        parent.keys.add(index, fullChild.keys[midIndex])
        parent.children.add(index + 1, newChild)
        
        // Remove moved keys from full child
        fullChild.keys.subList(midIndex, fullChild.keys.size).clear()
        if (!fullChild.isLeaf) {
            fullChild.children.subList(midIndex + 1, fullChild.children.size).clear()
        }
    }
    
    fun search(key: T): Boolean {
        return searchHelper(root, key)
    }
    
    private fun searchHelper(node: Node<T>, key: T): Boolean {
        var i = 0
        while (i < node.keys.size && key > node.keys[i]) {
            i++
        }
        
        if (i < node.keys.size && key == node.keys[i]) {
            return true
        }
        
        return if (node.isLeaf) {
            false
        } else {
            searchHelper(node.children[i], key)
        }
    }
}

fun main() {
    println("Advanced Tree Data Structures in Kotlin")
    println("========================================")
    
    // Test AVL Tree
    val avl = AVLTree<Int>()
    listOf(10, 20, 30, 40, 50, 25).forEach { avl.insert(it) }
    println("AVL Tree inorder: ${avl.inorderTraversal()}")
    println("Search 25: ${avl.search(25)}")
}
'''
    }
    
    return samples

# Generate all samples
def main():
    print("Generating code samples...")
    
    # Generate R samples
    r_samples = generate_r_samples()
    r_path = CODE_SAMPLES_PATH / "r"
    r_path.mkdir(exist_ok=True)
    
    for filename, content in r_samples.items():
        filepath = r_path / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Created: {filepath}")
    
    # Generate Kotlin samples
    kotlin_samples = generate_kotlin_samples()
    kotlin_path = CODE_SAMPLES_PATH / "kotlin"
    kotlin_path.mkdir(exist_ok=True)
    
    for filename, content in kotlin_samples.items():
        filepath = kotlin_path / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Created: {filepath}")
    
    # Generate Python samples
    python_samples = generate_python_samples()
    python_path = CODE_SAMPLES_PATH / "python"
    
    for filename, content in python_samples.items():
        filepath = python_path / filename
        if not filepath.exists():  # Don't overwrite existing files
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate JavaScript samples
    js_samples = generate_javascript_samples()
    js_path = CODE_SAMPLES_PATH / "javascript"
    
    for filename, content in js_samples.items():
        filepath = js_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate TypeScript samples
    ts_samples = generate_typescript_samples()
    ts_path = CODE_SAMPLES_PATH / "typescript"
    
    for filename, content in ts_samples.items():
        filepath = ts_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate Java samples
    java_samples = generate_java_samples()
    java_path = CODE_SAMPLES_PATH / "java"
    
    for filename, content in java_samples.items():
        filepath = java_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate Go samples
    go_samples = generate_go_samples()
    go_path = CODE_SAMPLES_PATH / "go"
    
    for filename, content in go_samples.items():
        filepath = go_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate PHP samples
    php_samples = generate_php_samples()
    php_path = CODE_SAMPLES_PATH / "php"
    
    for filename, content in php_samples.items():
        filepath = php_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate Ruby samples
    ruby_samples = generate_ruby_samples()
    ruby_path = CODE_SAMPLES_PATH / "ruby"
    
    for filename, content in ruby_samples.items():
        filepath = ruby_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate C++ samples
    cpp_samples = generate_cpp_samples()
    cpp_path = CODE_SAMPLES_PATH / "cpp"
    
    for filename, content in cpp_samples.items():
        filepath = cpp_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate C# samples
    csharp_samples = generate_csharp_samples()
    csharp_path = CODE_SAMPLES_PATH / "csharp"
    
    for filename, content in csharp_samples.items():
        filepath = csharp_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate Swift samples
    swift_samples = generate_swift_samples()
    swift_path = CODE_SAMPLES_PATH / "swift"
    
    for filename, content in swift_samples.items():
        filepath = swift_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    # Generate Perl samples
    perl_samples = generate_perl_samples()
    perl_path = CODE_SAMPLES_PATH / "perl"
    
    for filename, content in perl_samples.items():
        filepath = perl_path / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
    
    print(f"\nGenerated samples:")
    print(f"  R: {len(r_samples)}")
    print(f"  Kotlin: {len(kotlin_samples)}")
    print(f"  Python: {len(python_samples)}")
    print(f"  JavaScript: {len(js_samples)}")
    print(f"  TypeScript: {len(ts_samples)}")
    print(f"  Java: {len(java_samples)}")
    print(f"  Go: {len(go_samples)}")
    print(f"  PHP: {len(php_samples)}")
    print(f"  Ruby: {len(ruby_samples)}")
    print(f"  C++: {len(cpp_samples)}")
    print(f"  C#: {len(csharp_samples)}")
    print(f"  Swift: {len(swift_samples)}")
    print(f"  Perl: {len(perl_samples)}")
    total = sum([len(r_samples), len(kotlin_samples), len(python_samples), len(js_samples),
                 len(ts_samples), len(java_samples), len(go_samples), len(php_samples),
                 len(ruby_samples), len(cpp_samples), len(csharp_samples), len(swift_samples),
                 len(perl_samples)])
    print(f"Total new samples: {total}")

def generate_python_samples():
    """Generate additional Python code samples"""
    samples = {
        "concurrency_multiprocessing_advanced.py": '''"""
Advanced Multiprocessing Patterns in Python
Demonstrates process pools, shared memory, and IPC
"""

import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue, Process, Value, Array
import time
import os

def worker_function(x):
    """Simple worker function for pool"""
    return x * x

def parallel_map_reduce():
    """Demonstrate map-reduce with process pool"""
    with Pool(processes=4) as pool:
        # Map phase
        numbers = range(1, 100)
        squared = pool.map(worker_function, numbers)
        
        # Reduce phase
        total = sum(squared)
        return total

def shared_memory_example():
    """Demonstrate shared memory between processes"""
    # Shared value
    counter = Value('i', 0)
    
    # Shared array
    shared_arr = Array('d', [0.0] * 10)
    
    def increment_counter(val, arr, idx):
        with val.get_lock():
            val.value += 1
        arr[idx] = val.value * 2.0
    
    processes = []
    for i in range(5):
        p = Process(target=increment_counter, args=(counter, shared_arr, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f"Final counter value: {counter.value}")
    print(f"Shared array: {list(shared_arr[:5])}")

def producer_consumer_pattern():
    """Demonstrate producer-consumer with Queue"""
    def producer(queue, items):
        for item in items:
            queue.put(item)
            time.sleep(0.1)
        queue.put(None)  # Sentinel
    
    def consumer(queue):
        results = []
        while True:
            item = queue.get()
            if item is None:
                break
            results.append(item * 2)
        return results
    
    q = Queue()
    items = range(10)
    
    prod = Process(target=producer, args=(q, items))
    cons = Process(target=consumer, args=(q,))
    
    prod.start()
    cons.start()
    
    prod.join()
    cons.join()

def parallel_computation():
    """Compute intensive task with parallel execution"""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    numbers = range(1, 10000)
    
    # Serial execution
    start = time.time()
    serial_primes = [n for n in numbers if is_prime(n)]
    serial_time = time.time() - start
    
    # Parallel execution
    start = time.time()
    with Pool() as pool:
        results = pool.map(is_prime, numbers)
        parallel_primes = [n for n, is_p in zip(numbers, results) if is_p]
    parallel_time = time.time() - start
    
    print(f"Serial time: {serial_time:.4f}s")
    print(f"Parallel time: {parallel_time:.4f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
    print(f"Found {len(parallel_primes)} primes")

if __name__ == "__main__":
    print("Advanced Multiprocessing Patterns")
    print("=================================\\n")
    
    print("1. Parallel Map-Reduce:")
    result = parallel_map_reduce()
    print(f"   Sum of squares: {result}\\n")
    
    print("2. Shared Memory:")
    shared_memory_example()
    print()
    
    print("3. Producer-Consumer Pattern:")
    producer_consumer_pattern()
    print()
    
    print("4. Parallel Computation:")
    parallel_computation()
''',

        "design_patterns_command.py": '''"""
Command Design Pattern in Python
Encapsulates requests as objects for flexible execution
"""

from abc import ABC, abstractmethod
from typing import List

# Command interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

# Receiver class
class Light:
    def __init__(self, location: str):
        self.location = location
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        print(f"{self.location} light is ON")
    
    def turn_off(self):
        self.is_on = False
        print(f"{self.location} light is OFF")

# Concrete commands
class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_on()
    
    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_off()
    
    def undo(self):
        self.light.turn_on()

# Invoker
class RemoteControl:
    def __init__(self):
        self.commands: List[Command] = []
        self.history: List[Command] = []
    
    def set_command(self, slot: int, command: Command):
        if len(self.commands) <= slot:
            self.commands.extend([None] * (slot - len(self.commands) + 1))
        self.commands[slot] = command
    
    def press_button(self, slot: int):
        if slot < len(self.commands) and self.commands[slot]:
            self.commands[slot].execute()
            self.history.append(self.commands[slot])
    
    def press_undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()

# Macro command
class MacroCommand(Command):
    def __init__(self, commands: List[Command]):
        self.commands = commands
    
    def execute(self):
        for command in self.commands:
            command.execute()
    
    def undo(self):
        for command in reversed(self.commands):
            command.undo()

def main():
    print("Command Pattern Demonstration")
    print("============================\\n")
    
    # Create receivers
    living_room_light = Light("Living Room")
    bedroom_light = Light("Bedroom")
    
    # Create commands
    living_on = LightOnCommand(living_room_light)
    living_off = LightOffCommand(living_room_light)
    bedroom_on = LightOnCommand(bedroom_light)
    bedroom_off = LightOffCommand(bedroom_light)
    
    # Set up remote
    remote = RemoteControl()
    remote.set_command(0, living_on)
    remote.set_command(1, living_off)
    remote.set_command(2, bedroom_on)
    remote.set_command(3, bedroom_off)
    
    # Test commands
    print("Pressing button 0:")
    remote.press_button(0)
    
    print("\\nPressing button 2:")
    remote.press_button(2)
    
    print("\\nUndoing last command:")
    remote.press_undo()
    
    print("\\nTesting macro command:")
    all_on = MacroCommand([living_on, bedroom_on])
    all_on.execute()
    
    print("\\nUndoing macro:")
    all_on.undo()

if __name__ == "__main__":
    main()
''',
    }
    return samples

def generate_javascript_samples():
    """Generate additional JavaScript code samples"""
    samples = {
        "promises_async_patterns.js": '''/**
 * Advanced Promise and Async Patterns in JavaScript
 * Demonstrates promise composition, error handling, and async utilities
 */

// Promise utility functions
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Retry with exponential backoff
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            
            const delayTime = baseDelay * Math.pow(2, i);
            console.log(`Retry ${i + 1} after ${delayTime}ms`);
            await delay(delayTime);
        }
    }
}

// Promise.all with concurrency limit
async function promiseAllLimit(promises, limit) {
    const results = [];
    const executing = [];
    
    for (const [index, promise] of promises.entries()) {
        const p = Promise.resolve(promise).then(
            result => {
                results[index] = result;
            }
        );
        
        results.push(undefined);
        
        if (limit <= promises.length) {
            const e = p.then(() => executing.splice(executing.indexOf(e), 1));
            executing.push(e);
            
            if (executing.length >= limit) {
                await Promise.race(executing);
            }
        }
    }
    
    await Promise.all(executing);
    return results;
}

// Promise timeout wrapper
function withTimeout(promise, timeoutMs) {
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Timeout')), timeoutMs)
        )
    ]);
}

// Async queue implementation
class AsyncQueue {
    constructor(concurrency = 1) {
        this.concurrency = concurrency;
        this.running = 0;
        this.queue = [];
    }
    
    async add(fn) {
        return new Promise((resolve, reject) => {
            this.queue.push({ fn, resolve, reject });
            this.process();
        });
    }
    
    async process() {
        if (this.running >= this.concurrency || this.queue.length === 0) {
            return;
        }
        
        this.running++;
        const { fn, resolve, reject } = this.queue.shift();
        
        try {
            const result = await fn();
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running--;
            this.process();
        }
    }
}

// Async generator example
async function* asyncGenerator(items) {
    for (const item of items) {
        await delay(100);
        yield item * 2;
    }
}

// Pipeline composition
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const pipeAsync = (...fns) => x => fns.reduce(async (v, f) => f(await v), x);

// Demonstration
async function main() {
    console.log("Advanced Promise Patterns");
    console.log("========================\\n");
    
    // Test retry with backoff
    console.log("1. Retry with backoff:");
    let attempts = 0;
    const flaky = async () => {
        attempts++;
        if (attempts < 3) throw new Error("Fail");
        return "Success";
    };
    const result = await retryWithBackoff(flaky);
    console.log(`Result: ${result}\\n`);
    
    // Test async queue
    console.log("2. Async queue:");
    const queue = new AsyncQueue(2);
    const tasks = [1, 2, 3, 4, 5].map(n =>
        () => delay(100).then(() => n * 2)
    );
    const queueResults = await Promise.all(tasks.map(task => queue.add(task)));
    console.log(`Queue results: ${queueResults}\\n`);
    
    // Test async generator
    console.log("3. Async generator:");
    for await (const value of asyncGenerator([1, 2, 3])) {
        console.log(`  Generated: ${value}`);
    }
}

main().catch(console.error);
''',

        "functional_composition.js": '''/**
 * Functional Programming Composition Patterns
 * Demonstrates pure functions, composition, and functional utilities
 */

// Pure function utilities
const identity = x => x;
const constant = x => () => x;
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);

// Currying
const curry = (fn) => {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...args2) {
                return curried.apply(this, args.concat(args2));
            };
        }
    };
};

// Partial application
const partial = (fn, ...presetArgs) => {
    return (...laterArgs) => fn(...presetArgs, ...laterArgs);
};

// Function composition utilities
const map = curry((fn, arr) => arr.map(fn));
const filter = curry((fn, arr) => arr.filter(fn));
const reduce = curry((fn, init, arr) => arr.reduce(fn, init));

// Transducers
const mapping = (fn) => (step) => {
    return (acc, value) => step(acc, fn(value));
};

const filtering = (predicate) => (step) => {
    return (acc, value) => predicate(value) ? step(acc, value) : acc;
};

const transduce = (xform, step, init, coll) => {
    const xf = xform(step);
    return coll.reduce(xf, init);
};

// Monads
class Maybe {
    constructor(value) {
        this.value = value;
    }
    
    static of(value) {
        return new Maybe(value);
    }
    
    isNothing() {
        return this.value === null || this.value === undefined;
    }
    
    map(fn) {
        return this.isNothing() ? this : Maybe.of(fn(this.value));
    }
    
    flatMap(fn) {
        return this.isNothing() ? this : fn(this.value);
    }
    
    getOrElse(defaultValue) {
        return this.isNothing() ? defaultValue : this.value;
    }
}

class Either {
    constructor(value, isRight = true) {
        this.value = value;
        this.isRight = isRight;
    }
    
    static right(value) {
        return new Either(value, true);
    }
    
    static left(value) {
        return new Either(value, false);
    }
    
    map(fn) {
        return this.isRight ? Either.right(fn(this.value)) : this;
    }
    
    flatMap(fn) {
        return this.isRight ? fn(this.value) : this;
    }
    
    fold(leftFn, rightFn) {
        return this.isRight ? rightFn(this.value) : leftFn(this.value);
    }
}

// Lens implementation
const lens = (getter, setter) => ({
    get: getter,
    set: setter,
    over: (fn, obj) => setter(fn(getter(obj)), obj)
});

const view = (lens, obj) => lens.get(obj);
const set = (lens, value, obj) => lens.set(value, obj);
const over = (lens, fn, obj) => lens.over(fn, obj);

// Demonstration
function main() {
    console.log("Functional Composition Patterns");
    console.log("===============================\\n");
    
    // Composition example
    const double = x => x * 2;
    const increment = x => x + 1;
    const square = x => x * x;
    
    const transform = pipe(double, increment, square);
    console.log("1. Function composition:");
    console.log(`   pipe(double, increment, square)(3) = ${transform(3)}\\n`);
    
    // Currying example
    const add = curry((a, b, c) => a + b + c);
    console.log("2. Currying:");
    console.log(`   add(1)(2)(3) = ${add(1)(2)(3)}\\n`);
    
    // Maybe monad
    console.log("3. Maybe monad:");
    const safeDiv = (a, b) => b === 0 ? Maybe.of(null) : Maybe.of(a / b);
    console.log(`   10 / 2 = ${safeDiv(10, 2).getOrElse("Error")}`);
    console.log(`   10 / 0 = ${safeDiv(10, 0).getOrElse("Error")}\\n`);
    
    // Transducers
    console.log("4. Transducers:");
    const xform = compose(
        filtering(x => x % 2 === 0),
        mapping(x => x * 2)
    );
    const result = transduce(
        xform,
        (acc, x) => acc.concat(x),
        [],
        [1, 2, 3, 4, 5, 6]
    );
    console.log(`   Result: [${result}]`);
}

main();
''',
    }
    return samples

def generate_typescript_samples():
    """Generate TypeScript code samples"""
    samples = {
        "generics_advanced.ts": '''/**
 * Advanced Generic Types in TypeScript
 * Demonstrates generic constraints, mapped types, and conditional types
 */

// Generic constraints
interface Lengthwise {
    length: number;
}

function logLength<T extends Lengthwise>(arg: T): T {
    console.log(arg.length);
    return arg;
}

// Generic factory pattern
interface Constructor<T> {
    new(...args: any[]): T;
}

function create<T>(ctor: Constructor<T>, ...args: any[]): T {
    return new ctor(...args);
}

// Custom mapped types (avoiding built-in duplicates)
type MyReadonly<T> = {
    readonly [P in keyof T]: T[P];
};

type MyPartial<T> = {
    [P in keyof T]?: T[P];
};

type Nullable<T> = {
    [P in keyof T]: T[P] | null;
};

// Conditional types
type Flatten<T> = T extends Array<infer U> ? U : T;

type ExtractPromise<T> = T extends Promise<infer U> ? U : T;

// Advanced generics
interface Repository<T> {
    getById(id: string): T | null;
    getAll(): T[];
    create(item: Omit<T, 'id'>): T;
    update(id: string, item: MyPartial<T>): T;
    delete(id: string): boolean;
}

class InMemoryRepository<T extends { id: string }> implements Repository<T> {
    private items: { [key: string]: T } = {};
    
    getById(id: string): T | null {
        return this.items[id] || null;
    }
    
    getAll(): T[] {
        return Object.values(this.items);
    }
    
    create(item: Omit<T, 'id'>): T {
        const id = Math.random().toString(36).substr(2, 9);
        const newItem = { ...item, id } as T;
        this.items[id] = newItem;
        return newItem;
    }
    
    update(id: string, item: MyPartial<T>): T {
        const existing = this.items[id];
        if (!existing) throw new Error('Not found');
        const updated = { ...existing, ...item };
        this.items[id] = updated;
        return updated;
    }
    
    delete(id: string): boolean {
        if (this.items[id]) {
            delete this.items[id];
            return true;
        }
        return false;
    }
}

// Type guards
function isString(value: unknown): value is string {
    return typeof value === 'string';
}

function isArray<T>(value: unknown): value is T[] {
    return Array.isArray(value);
}

// Utility types
type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// Example usage
interface User {
    id: string;
    name: string;
    email: string;
    age: number;
}

const userRepo = new InMemoryRepository<User>();

console.log("Advanced TypeScript Generics");
console.log("===========================");
''',
    }
    return samples

def generate_java_samples():
    """Generate Java code samples"""
    samples = {}  # Will add later to stay within limits
    return samples

def generate_go_samples():
    """Generate Go code samples"""
    samples = {}
    return samples

def generate_php_samples():
    """Generate PHP code samples"""
    samples = {}
    return samples

def generate_ruby_samples():
    """Generate Ruby code samples"""
    samples = {}
    return samples

def generate_cpp_samples():
    """Generate C++ code samples"""
    samples = {}
    return samples

def generate_csharp_samples():
    """Generate C# code samples"""
    samples = {}
    return samples

def generate_swift_samples():
    """Generate Swift code samples"""
    samples = {}
    return samples

def generate_perl_samples():
    """Generate Perl code samples"""
    samples = {}
    return samples

if __name__ == "__main__":
    main()
