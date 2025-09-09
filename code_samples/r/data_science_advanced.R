# Advanced R Data Science and Statistical Analysis
# This file demonstrates intermediate to advanced R programming concepts for
# data science, statistical modeling, and machine learning applications

# Load required libraries
library(tidyverse)      # Data manipulation and visualization
library(dplyr)          # Data wrangling
library(ggplot2)        # Advanced plotting
library(caret)          # Machine learning
library(randomForest)   # Random forest algorithm
library(e1071)          # SVM and other algorithms
library(cluster)        # Clustering algorithms
library(factoextra)     # Factor analysis and PCA
library(corrplot)       # Correlation plots
library(plotly)         # Interactive plots
library(shiny)          # Web applications
library(lubridate)      # Date/time manipulation
library(stringr)        # String manipulation
library(purrr)          # Functional programming

# Advanced Data Manipulation Functions
# ====================================

#' Clean and preprocess a dataset
#' @param data Raw dataset
#' @param numeric_cols Columns to treat as numeric
#' @param categorical_cols Columns to treat as categorical
#' @return Cleaned dataset
preprocess_data <- function(data, numeric_cols = NULL, categorical_cols = NULL) {
  # Handle missing values
  data_clean <- data %>%
    # Remove rows with all NA values
    filter(!if_all(everything(), is.na)) %>%
    # Convert specified columns to appropriate types
    {
      if (!is.null(numeric_cols)) {
        mutate_at(., vars(all_of(numeric_cols)), as.numeric)
      } else {
        .
      }
    } %>%
    {
      if (!is.null(categorical_cols)) {
        mutate_at(., vars(all_of(categorical_cols)), as.factor)
      } else {
        .
      }
    }
  
  # Handle outliers using IQR method
  numeric_columns <- data_clean %>% 
    select_if(is.numeric) %>% 
    names()
  
  for (col in numeric_columns) {
    Q1 <- quantile(data_clean[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(data_clean[[col]], 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    
    # Cap outliers at 1.5 * IQR
    data_clean[[col]][data_clean[[col]] < (Q1 - 1.5 * IQR_val)] <- Q1 - 1.5 * IQR_val
    data_clean[[col]][data_clean[[col]] > (Q3 + 1.5 * IQR_val)] <- Q3 + 1.5 * IQR_val
  }
  
  return(data_clean)
}

#' Perform feature engineering on a dataset
#' @param data Input dataset
#' @return Dataset with engineered features
feature_engineering <- function(data) {
  data %>%
    # Create interaction terms for numeric variables
    {
      numeric_cols <- select_if(., is.numeric) %>% names()
      if (length(numeric_cols) >= 2) {
        mutate(.,
               interaction_1_2 = .[[numeric_cols[1]]] * .[[numeric_cols[2]]],
               ratio_1_2 = ifelse(.[[numeric_cols[2]]] != 0, 
                                  .[[numeric_cols[1]]] / .[[numeric_cols[2]]], 0)
        )
      } else {
        .
      }
    } %>%
    # Create polynomial features
    mutate_if(is.numeric, list(squared = ~ .^2, cubed = ~ .^3)) %>%
    # Create binned versions of continuous variables
    mutate_if(is.numeric, list(binned = ~ cut(., breaks = 5, labels = FALSE)))
}

# Advanced Statistical Functions
# ==============================

#' Perform comprehensive statistical analysis
#' @param data Dataset to analyze
#' @param target_var Target variable for analysis
#' @return List containing various statistical results
comprehensive_statistical_analysis <- function(data, target_var = NULL) {
  results <- list()
  
  # Descriptive statistics
  results$descriptive <- data %>%
    select_if(is.numeric) %>%
    summarise_all(list(
      mean = ~ mean(., na.rm = TRUE),
      median = ~ median(., na.rm = TRUE),
      sd = ~ sd(., na.rm = TRUE),
      min = ~ min(., na.rm = TRUE),
      max = ~ max(., na.rm = TRUE),
      q25 = ~ quantile(., 0.25, na.rm = TRUE),
      q75 = ~ quantile(., 0.75, na.rm = TRUE),
      skewness = ~ moments::skewness(., na.rm = TRUE),
      kurtosis = ~ moments::kurtosis(., na.rm = TRUE)
    ))
  
  # Correlation analysis
  numeric_data <- data %>% select_if(is.numeric)
  if (ncol(numeric_data) > 1) {
    results$correlation <- cor(numeric_data, use = "complete.obs")
  }
  
  # Distribution tests
  results$normality_tests <- numeric_data %>%
    map_dfr(~ {
      if (length(na.omit(.)) > 3) {
        shapiro_test <- shapiro.test(sample(na.omit(.), min(5000, length(na.omit(.)))))
        tibble(
          p_value = shapiro_test$p.value,
          is_normal = shapiro_test$p.value > 0.05
        )
      } else {
        tibble(p_value = NA, is_normal = NA)
      }
    }, .id = "variable")
  
  # If target variable is specified, perform additional analysis
  if (!is.null(target_var) && target_var %in% names(data)) {
    # ANOVA for categorical predictors
    categorical_vars <- data %>% 
      select_if(is.factor) %>% 
      names()
    
    if (length(categorical_vars) > 0 && is.numeric(data[[target_var]])) {
      results$anova <- map_dfr(categorical_vars, ~ {
        formula_str <- paste(target_var, "~", .)
        anova_result <- aov(as.formula(formula_str), data = data)
        summary_result <- summary(anova_result)
        
        tibble(
          variable = .,
          f_statistic = summary_result[[1]]$`F value`[1],
          p_value = summary_result[[1]]$`Pr(>F)`[1],
          significant = summary_result[[1]]$`Pr(>F)`[1] < 0.05
        )
      })
    }
  }
  
  return(results)
}

#' Perform time series analysis
#' @param data Time series data
#' @param date_col Column containing dates
#' @param value_col Column containing values
#' @return Time series analysis results
time_series_analysis <- function(data, date_col, value_col) {
  # Convert to time series object
  ts_data <- data %>%
    arrange(!!sym(date_col)) %>%
    mutate(date = as.Date(!!sym(date_col))) %>%
    select(date, value = !!sym(value_col))
  
  # Create ts object
  ts_obj <- ts(ts_data$value, frequency = 12)  # Assuming monthly data
  
  # Decomposition
  decomp <- decompose(ts_obj)
  
  # Trend analysis
  trend_model <- lm(value ~ as.numeric(date), data = ts_data)
  
  # Seasonality detection
  seasonal_strength <- var(decomp$seasonal, na.rm = TRUE) / var(decomp$x, na.rm = TRUE)
  
  # ARIMA modeling
  arima_model <- auto.arima(ts_obj)
  
  # Forecasting
  forecast_result <- forecast(arima_model, h = 12)
  
  return(list(
    decomposition = decomp,
    trend_model = trend_model,
    seasonal_strength = seasonal_strength,
    arima_model = arima_model,
    forecast = forecast_result,
    original_data = ts_data
  ))
}

# Machine Learning Pipeline
# =========================

#' Create a comprehensive machine learning pipeline
#' @param data Training data
#' @param target_var Target variable name
#' @param test_size Proportion of data for testing
#' @param algorithms Vector of algorithms to try
#' @return Trained models and evaluation results
ml_pipeline <- function(data, target_var, test_size = 0.2, 
                       algorithms = c("rf", "svm", "glm", "knn")) {
  
  # Data splitting
  set.seed(123)
  train_indices <- createDataPartition(data[[target_var]], p = 1 - test_size, list = FALSE)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Preprocessing
  preprocess_recipe <- preProcess(train_data[, !names(train_data) %in% target_var], 
                                  method = c("center", "scale", "nzv"))
  
  train_processed <- predict(preprocess_recipe, train_data)
  test_processed <- predict(preprocess_recipe, test_data)
  
  # Model training with cross-validation
  ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
  
  models <- list()
  
  if ("rf" %in% algorithms) {
    models$random_forest <- train(
      as.formula(paste(target_var, "~ .")),
      data = train_processed,
      method = "rf",
      trControl = ctrl,
      tuneLength = 3
    )
  }
  
  if ("svm" %in% algorithms) {
    models$svm <- train(
      as.formula(paste(target_var, "~ .")),
      data = train_processed,
      method = "svmRadial",
      trControl = ctrl,
      tuneLength = 3
    )
  }
  
  if ("glm" %in% algorithms) {
    models$glm <- train(
      as.formula(paste(target_var, "~ .")),
      data = train_processed,
      method = "glm",
      trControl = ctrl
    )
  }
  
  if ("knn" %in% algorithms) {
    models$knn <- train(
      as.formula(paste(target_var, "~ .")),
      data = train_processed,
      method = "knn",
      trControl = ctrl,
      tuneLength = 5
    )
  }
  
  # Model evaluation
  predictions <- map(models, ~ predict(., test_processed))
  
  # Calculate metrics based on problem type
  if (is.numeric(test_data[[target_var]])) {
    # Regression metrics
    metrics <- map_dfr(predictions, ~ {
      tibble(
        rmse = RMSE(., test_data[[target_var]]),
        mae = MAE(., test_data[[target_var]]),
        r_squared = R2(., test_data[[target_var]])
      )
    }, .id = "model")
  } else {
    # Classification metrics
    metrics <- map_dfr(predictions, ~ {
      cm <- confusionMatrix(factor(.), factor(test_data[[target_var]]))
      tibble(
        accuracy = cm$overall["Accuracy"],
        kappa = cm$overall["Kappa"],
        sensitivity = cm$byClass["Sensitivity"],
        specificity = cm$byClass["Specificity"]
      )
    }, .id = "model")
  }
  
  return(list(
    models = models,
    predictions = predictions,
    metrics = metrics,
    preprocess_recipe = preprocess_recipe,
    test_data = test_data
  ))
}

# Advanced Visualization Functions
# ================================

#' Create comprehensive data visualization dashboard
#' @param data Dataset to visualize
#' @param target_var Optional target variable
#' @return List of ggplot objects
create_visualization_dashboard <- function(data, target_var = NULL) {
  plots <- list()
  
  # Distribution plots for numeric variables
  numeric_vars <- data %>% select_if(is.numeric) %>% names()
  
  if (length(numeric_vars) > 0) {
    plots$distributions <- data %>%
      select(all_of(numeric_vars)) %>%
      gather(key = "variable", value = "value") %>%
      ggplot(aes(x = value)) +
      geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
      facet_wrap(~ variable, scales = "free") +
      theme_minimal() +
      labs(title = "Distribution of Numeric Variables")
  }
  
  # Correlation heatmap
  if (length(numeric_vars) > 1) {
    corr_matrix <- cor(data[numeric_vars], use = "complete.obs")
    plots$correlation <- corrplot(corr_matrix, method = "color", type = "upper",
                                  order = "hclust", tl.cex = 0.8, tl.col = "black")
  }
  
  # Boxplots for outlier detection
  if (length(numeric_vars) > 0) {
    plots$boxplots <- data %>%
      select(all_of(numeric_vars)) %>%
      gather(key = "variable", value = "value") %>%
      ggplot(aes(x = variable, y = value)) +
      geom_boxplot(fill = "lightblue", alpha = 0.7) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Boxplots for Outlier Detection")
  }
  
  # Categorical variable analysis
  categorical_vars <- data %>% select_if(is.factor) %>% names()
  
  if (length(categorical_vars) > 0) {
    plots$categorical <- data %>%
      select(all_of(categorical_vars)) %>%
      gather(key = "variable", value = "value") %>%
      count(variable, value) %>%
      ggplot(aes(x = value, y = n, fill = variable)) +
      geom_col() +
      facet_wrap(~ variable, scales = "free") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Categorical Variable Frequencies")
  }
  
  # Target variable analysis if provided
  if (!is.null(target_var) && target_var %in% names(data)) {
    if (is.numeric(data[[target_var]])) {
      # Scatter plots against target
      if (length(numeric_vars) > 1) {
        other_numeric <- setdiff(numeric_vars, target_var)
        plots$target_relationships <- data %>%
          select(!!sym(target_var), all_of(other_numeric[1:min(4, length(other_numeric))])) %>%
          gather(key = "variable", value = "value", -!!sym(target_var)) %>%
          ggplot(aes(x = value, y = !!sym(target_var))) +
          geom_point(alpha = 0.6) +
          geom_smooth(method = "lm", se = FALSE, color = "red") +
          facet_wrap(~ variable, scales = "free") +
          theme_minimal() +
          labs(title = paste("Relationships with", target_var))
      }
    }
  }
  
  return(plots)
}

# Clustering Analysis
# ==================

#' Perform comprehensive clustering analysis
#' @param data Dataset for clustering
#' @param k_range Range of k values to try
#' @return Clustering results and optimal k
clustering_analysis <- function(data, k_range = 2:8) {
  # Prepare data (only numeric variables)
  numeric_data <- data %>% 
    select_if(is.numeric) %>%
    na.omit() %>%
    scale()
  
  # Determine optimal number of clusters
  # Elbow method
  wss <- map_dbl(k_range, ~ {
    kmeans(numeric_data, centers = ., nstart = 20)$tot.withinss
  })
  
  elbow_plot <- tibble(k = k_range, wss = wss) %>%
    ggplot(aes(x = k, y = wss)) +
    geom_line() +
    geom_point() +
    theme_minimal() +
    labs(title = "Elbow Method for Optimal K",
         x = "Number of Clusters",
         y = "Within-cluster Sum of Squares")
  
  # Silhouette method
  silhouette_scores <- map_dbl(k_range, ~ {
    km_result <- kmeans(numeric_data, centers = ., nstart = 20)
    silhouette(km_result$cluster, dist(numeric_data)) %>%
      summary() %>%
      .$avg.width
  })
  
  optimal_k <- k_range[which.max(silhouette_scores)]
  
  # Perform clustering with optimal k
  final_kmeans <- kmeans(numeric_data, centers = optimal_k, nstart = 20)
  
  # Hierarchical clustering
  hc_result <- hclust(dist(numeric_data), method = "ward.D2")
  
  # PCA for visualization
  pca_result <- prcomp(numeric_data)
  pca_data <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    cluster = as.factor(final_kmeans$cluster)
  )
  
  cluster_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
    geom_point(alpha = 0.7) +
    theme_minimal() +
    labs(title = "Cluster Visualization (PCA)",
         x = paste("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 1), "% variance)"),
         y = paste("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 1), "% variance)"))
  
  return(list(
    optimal_k = optimal_k,
    kmeans_result = final_kmeans,
    hierarchical_result = hc_result,
    silhouette_scores = silhouette_scores,
    elbow_plot = elbow_plot,
    cluster_plot = cluster_plot,
    pca_result = pca_result
  ))
}

# Text Analytics Functions
# =======================

#' Perform text analysis on character columns
#' @param text_data Vector of text data
#' @return Text analysis results
text_analytics <- function(text_data) {
  # Clean text data
  clean_text <- text_data %>%
    str_to_lower() %>%
    str_remove_all("[^a-zA-Z\\s]") %>%
    str_squish()
  
  # Word frequency analysis
  words <- clean_text %>%
    str_split("\\s+") %>%
    unlist() %>%
    table() %>%
    sort(decreasing = TRUE)
  
  # Sentiment analysis (simplified)
  positive_words <- c("good", "great", "excellent", "amazing", "wonderful", "fantastic")
  negative_words <- c("bad", "terrible", "awful", "horrible", "disappointing", "poor")
  
  sentiment_scores <- map_dbl(clean_text, ~ {
    words_in_text <- str_split(., "\\s+")[[1]]
    positive_count <- sum(words_in_text %in% positive_words)
    negative_count <- sum(words_in_text %in% negative_words)
    positive_count - negative_count
  })
  
  # Text statistics
  text_stats <- tibble(
    text = text_data,
    clean_text = clean_text,
    word_count = str_count(clean_text, "\\S+"),
    char_count = str_length(clean_text),
    sentiment_score = sentiment_scores
  )
  
  return(list(
    word_frequencies = head(words, 20),
    text_statistics = text_stats,
    avg_sentiment = mean(sentiment_scores, na.rm = TRUE)
  ))
}

# Example Usage and Testing
# ========================

# Generate sample data for demonstration
generate_sample_data <- function(n = 1000) {
  set.seed(123)
  
  tibble(
    id = 1:n,
    age = rnorm(n, 35, 10) %>% pmax(18) %>% pmin(80),
    income = rlnorm(n, 10, 0.5),
    education = sample(c("High School", "Bachelor", "Master", "PhD"), n, 
                      replace = TRUE, prob = c(0.4, 0.35, 0.2, 0.05)),
    experience = age - 22 + rnorm(n, 0, 2),
    satisfaction = rbinom(n, 1, 0.7),
    department = sample(c("Sales", "Engineering", "Marketing", "HR"), n, replace = TRUE),
    performance_score = 60 + 20 * satisfaction + rnorm(n, 0, 10),
    city = sample(c("New York", "Los Angeles", "Chicago", "Houston"), n, replace = TRUE),
    join_date = sample(seq(as.Date("2015-01-01"), as.Date("2023-12-31"), by = "day"), n),
    comments = sample(c("Great company to work for", "Could be better", 
                       "Excellent work environment", "Needs improvement",
                       "Very satisfied with my role"), n, replace = TRUE)
  ) %>%
    mutate(
      education = factor(education, levels = c("High School", "Bachelor", "Master", "PhD")),
      department = factor(department),
      city = factor(city),
      satisfaction = factor(satisfaction, labels = c("Unsatisfied", "Satisfied"))
    )
}

# Main analysis function
main_analysis <- function() {
  cat("=== Advanced R Data Science Pipeline ===\n\n")
  
  # Generate and preprocess data
  cat("1. Generating and preprocessing data...\n")
  raw_data <- generate_sample_data(1000)
  clean_data <- preprocess_data(raw_data, 
                               numeric_cols = c("age", "income", "experience", "performance_score"),
                               categorical_cols = c("education", "department", "satisfaction", "city"))
  
  # Feature engineering
  cat("2. Performing feature engineering...\n")
  engineered_data <- feature_engineering(clean_data)
  
  # Statistical analysis
  cat("3. Conducting statistical analysis...\n")
  stats_results <- comprehensive_statistical_analysis(clean_data, "performance_score")
  
  cat("   - Descriptive statistics calculated\n")
  cat("   - Correlation analysis completed\n")
  cat("   - Normality tests performed\n")
  
  # Machine learning pipeline
  cat("4. Running machine learning pipeline...\n")
  ml_results <- ml_pipeline(clean_data, "performance_score", 
                           algorithms = c("rf", "glm"))
  
  cat("   - Models trained:", paste(names(ml_results$models), collapse = ", "), "\n")
  cat("   - Best model performance:\n")
  best_model <- ml_results$metrics[which.min(ml_results$metrics$rmse), ]
  cat("     Model:", best_model$model, "RMSE:", round(best_model$rmse, 3), "\n")
  
  # Clustering analysis
  cat("5. Performing clustering analysis...\n")
  cluster_data <- clean_data %>% select_if(is.numeric)
  clustering_results <- clustering_analysis(cluster_data)
  cat("   - Optimal number of clusters:", clustering_results$optimal_k, "\n")
  
  # Text analytics
  cat("6. Analyzing text data...\n")
  text_results <- text_analytics(clean_data$comments)
  cat("   - Average sentiment score:", round(text_results$avg_sentiment, 3), "\n")
  cat("   - Most common words:", paste(names(head(text_results$word_frequencies, 5)), collapse = ", "), "\n")
  
  # Create visualizations
  cat("7. Creating visualizations...\n")
  viz_results <- create_visualization_dashboard(clean_data, "performance_score")
  cat("   - Generated", length(viz_results), "visualization plots\n")
  
  cat("\n=== Analysis Complete ===\n")
  
  return(list(
    data = clean_data,
    statistics = stats_results,
    ml_results = ml_results,
    clustering = clustering_results,
    text_analysis = text_results,
    visualizations = viz_results
  ))
}

# Run the main analysis
if (interactive()) {
  results <- main_analysis()
  
  # Display some key results
  cat("\nKey Insights:\n")
  cat("- Dataset contains", nrow(results$data), "observations\n")
  cat("- Strongest correlation with performance:", 
      names(which.max(abs(results$statistics$correlation["performance_score", ])[-1])), "\n")
  cat("- Best ML model:", results$ml_results$metrics[which.min(results$ml_results$metrics$rmse), ]$model, "\n")
  cat("- Identified", results$clustering$optimal_k, "distinct employee clusters\n")
}

# Additional utility functions for specific analyses
# =================================================

#' Perform A/B test analysis
#' @param control_group Control group data
#' @param treatment_group Treatment group data
#' @param metric Metric to compare
#' @return A/B test results
ab_test_analysis <- function(control_group, treatment_group, metric) {
  # Descriptive statistics
  control_stats <- summary(control_group[[metric]])
  treatment_stats <- summary(treatment_group[[metric]])
  
  # T-test
  t_test_result <- t.test(treatment_group[[metric]], control_group[[metric]])
  
  # Effect size (Cohen's d)
  pooled_sd <- sqrt(((length(control_group[[metric]]) - 1) * var(control_group[[metric]]) + 
                     (length(treatment_group[[metric]]) - 1) * var(treatment_group[[metric]])) /
                    (length(control_group[[metric]]) + length(treatment_group[[metric]]) - 2))
  
  cohens_d <- (mean(treatment_group[[metric]]) - mean(control_group[[metric]])) / pooled_sd
  
  # Power analysis
  power_result <- power.t.test(n = length(control_group[[metric]]), 
                              delta = abs(cohens_d), 
                              sd = pooled_sd)
  
  return(list(
    control_stats = control_stats,
    treatment_stats = treatment_stats,
    t_test = t_test_result,
    effect_size = cohens_d,
    power = power_result$power,
    significant = t_test_result$p.value < 0.05
  ))
}

#' Create an automated report
#' @param analysis_results Results from main_analysis()
#' @param output_file Output file path
create_automated_report <- function(analysis_results, output_file = "analysis_report.html") {
  # This would typically use R Markdown to create a comprehensive report
  # For now, we'll create a simple summary
  
  report_content <- paste(
    "# Data Science Analysis Report",
    "",
    "## Dataset Overview",
    paste("- Observations:", nrow(analysis_results$data)),
    paste("- Variables:", ncol(analysis_results$data)),
    "",
    "## Key Findings",
    paste("- Best ML Model RMSE:", round(min(analysis_results$ml_results$metrics$rmse), 3)),
    paste("- Optimal Clusters:", analysis_results$clustering$optimal_k),
    paste("- Average Sentiment:", round(analysis_results$text_analysis$avg_sentiment, 3)),
    "",
    "## Recommendations",
    "- Consider implementing the best performing model for predictions",
    "- Use clustering results for customer segmentation",
    "- Monitor sentiment scores for continuous improvement",
    sep = "\n"
  )
  
  writeLines(report_content, output_file)
  cat("Report saved to:", output_file, "\n")
}

cat("Advanced R Data Science toolkit loaded successfully!\n")
cat("Run main_analysis() to execute the complete pipeline.\n")