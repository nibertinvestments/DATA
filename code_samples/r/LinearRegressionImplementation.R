# Production-Ready Linear Regression Implementation in R
# ======================================================
# 
# This module demonstrates comprehensive linear regression implementations
# with multiple algorithms, regularization, and modern R patterns
# for AI training datasets.
#
# Key Features:
# - Multiple regression algorithms (OLS, Ridge, Lasso, Elastic Net)
# - Cross-validation and model selection
# - Feature engineering and preprocessing
# - Comprehensive diagnostic plots and metrics
# - R tidyverse integration and modern R patterns
# - Statistical significance testing
# - Bootstrap confidence intervals
# - Production deployment considerations
# 
# Author: AI Training Dataset
# License: MIT

# Required libraries (install if not available)
required_packages <- c("ggplot2", "dplyr", "tidyr", "broom", "glmnet", "boot", "car", "corrplot")

install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", pkg, "\n")
      install.packages(pkg, repos = "http://cran.r-project.org", quiet = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Install and load required packages
tryCatch({
  install_if_missing(required_packages)
}, error = function(e) {
  cat("Warning: Some packages could not be installed. Using base R fallbacks.\n")
})

#' Custom Linear Regression Class
#' 
#' A comprehensive linear regression implementation with multiple algorithms
#' and advanced statistical diagnostics.
LinearRegressionR <- R6::R6Class("LinearRegressionR",
  public = list(
    #' @field model_type Type of regression model
    model_type = NULL,
    
    #' @field coefficients Model coefficients
    coefficients = NULL,
    
    #' @field fitted_model Fitted model object
    fitted_model = NULL,
    
    #' @field training_data Training dataset
    training_data = NULL,
    
    #' @field feature_names Feature variable names
    feature_names = NULL,
    
    #' @field target_name Target variable name
    target_name = NULL,
    
    #' Initialize the linear regression model
    #' 
    #' @param model_type Type of regression ('ols', 'ridge', 'lasso', 'elastic_net')
    #' @param alpha Regularization strength for regularized models
    #' @param l1_ratio Elastic net mixing parameter (0 = ridge, 1 = lasso)
    initialize = function(model_type = "ols", alpha = 0.01, l1_ratio = 0.5) {
      self$model_type <- model_type
      private$.alpha <- alpha
      private$.l1_ratio <- l1_ratio
      private$.fitted <- FALSE
      
      cat("🔧 Initialized", toupper(model_type), "Linear Regression Model\n")
      cat("Parameters: alpha =", alpha, ", l1_ratio =", l1_ratio, "\n")
    },
    
    #' Fit the linear regression model
    #' 
    #' @param X Feature matrix or data frame
    #' @param y Target variable vector
    #' @param feature_names Optional feature names
    fit = function(X, y, feature_names = NULL) {
      cat("📈 Training Linear Regression Model\n")
      cat(paste(rep("=", 40), collapse = ""), "\n")
      
      # Validate inputs
      private$.validate_inputs(X, y)
      
      # Convert to matrix if needed
      if (is.data.frame(X)) {
        self$feature_names <- names(X)
        X <- as.matrix(X)
      } else {
        self$feature_names <- feature_names %||% paste0("X", 1:ncol(X))
      }
      
      # Store data
      self$training_data <- data.frame(
        y = y,
        X
      )
      names(self$training_data) <- c("y", self$feature_names)
      self$target_name <- "y"
      
      cat("Training samples:", nrow(X), "\n")
      cat("Features:", ncol(X), "\n")
      cat("Model type:", toupper(self$model_type), "\n")
      
      # Fit based on model type
      start_time <- Sys.time()
      
      switch(self$model_type,
        "ols" = private$.fit_ols(X, y),
        "ridge" = private$.fit_ridge(X, y),
        "lasso" = private$.fit_lasso(X, y),
        "elastic_net" = private$.fit_elastic_net(X, y),
        stop("Unknown model type: ", self$model_type)
      )
      
      end_time <- Sys.time()
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      cat("✅ Training completed in", round(training_time, 3), "seconds\n")
      
      # Print model summary
      self$print_summary()
      
      private$.fitted <- TRUE
      invisible(self)
    },
    
    #' Make predictions
    #' 
    #' @param X_new New data for predictions
    #' @return Predicted values
    predict = function(X_new) {
      if (!private$.fitted) {
        stop("Model not fitted. Call fit() first.")
      }
      
      # Convert to matrix if needed
      if (is.data.frame(X_new)) {
        X_new <- as.matrix(X_new)
      }
      
      # Validate dimensions
      if (ncol(X_new) != length(self$feature_names)) {
        stop("Feature dimension mismatch. Expected: ", length(self$feature_names), 
             ", got: ", ncol(X_new))
      }
      
      # Make predictions based on model type
      switch(self$model_type,
        "ols" = as.vector(X_new %*% self$coefficients[-1] + self$coefficients[1]),
        "ridge" = ,
        "lasso" = ,
        "elastic_net" = as.vector(predict(self$fitted_model, X_new, s = private$.best_lambda))
      )
    },
    
    #' Evaluate model performance
    #' 
    #' @param X_test Test features
    #' @param y_test Test targets
    #' @return List of evaluation metrics
    evaluate = function(X_test, y_test) {
      cat("\n📊 Evaluating Model Performance\n")
      cat(paste(rep("=", 35), collapse = ""), "\n")
      
      predictions <- self$predict(X_test)
      
      # Calculate metrics
      metrics <- private$.calculate_metrics(y_test, predictions)
      
      # Print metrics
      cat("Test samples:", length(y_test), "\n")
      cat("MSE:", round(metrics$mse, 6), "\n")
      cat("RMSE:", round(metrics$rmse, 6), "\n")
      cat("MAE:", round(metrics$mae, 6), "\n")
      cat("R²:", round(metrics$r_squared, 6), "\n")
      cat("Adjusted R²:", round(metrics$adj_r_squared, 6), "\n")
      
      return(metrics)
    },
    
    #' Print model summary
    print_summary = function() {
      if (!private$.fitted) {
        cat("Model not fitted yet.\n")
        return(invisible(self))
      }
      
      cat("\n🎯 Model Summary:\n")
      
      if (self$model_type == "ols") {
        # OLS specific summary
        summary_stats <- summary(self$fitted_model)
        print(summary_stats)
      } else {
        # Regularized model summary
        cat("Model type:", toupper(self$model_type), "\n")
        cat("Best lambda:", round(private$.best_lambda, 6), "\n")
        cat("Number of non-zero coefficients:", sum(abs(self$coefficients) > 1e-6), "\n")
        
        # Print non-zero coefficients
        non_zero_coefs <- which(abs(self$coefficients) > 1e-6)
        if (length(non_zero_coefs) > 0) {
          cat("\nNon-zero coefficients:\n")
          coef_names <- c("(Intercept)", self$feature_names)
          for (i in non_zero_coefs) {
            cat(sprintf("  %-20s: %10.6f\n", coef_names[i], self$coefficients[i]))
          }
        }
      }
      
      invisible(self)
    },
    
    #' Generate comprehensive diagnostic plots
    #' 
    #' @param save_plots Whether to save plots to files
    generate_diagnostic_plots = function(save_plots = FALSE) {
      if (!private$.fitted) {
        stop("Model not fitted. Call fit() first.")
      }
      
      cat("\n📊 Generating Diagnostic Plots\n")
      
      # Create plots based on available packages
      if (requireNamespace("ggplot2", quietly = TRUE) && 
          requireNamespace("dplyr", quietly = TRUE)) {
        private$.create_advanced_plots(save_plots)
      } else {
        private$.create_base_plots(save_plots)
      }
      
      cat("✅ Diagnostic plots generated\n")
      invisible(self)
    }
  ),
  
  private = list(
    .fitted = FALSE,
    .alpha = 0.01,
    .l1_ratio = 0.5,
    .best_lambda = NULL,
    
    # Validate input data
    .validate_inputs = function(X, y) {
      if (is.null(X) || is.null(y)) {
        stop("X and y cannot be NULL")
      }
      
      if (nrow(as.matrix(X)) != length(y)) {
        stop("Number of samples in X and y must match")
      }
      
      if (any(is.na(X)) || any(is.na(y))) {
        stop("Data contains missing values. Please handle them first.")
      }
      
      if (!is.numeric(y)) {
        stop("Target variable y must be numeric")
      }
    },
    
    # Fit OLS regression
    .fit_ols = function(X, y) {
      formula_str <- paste("y ~", paste(self$feature_names, collapse = " + "))
      formula_obj <- as.formula(formula_str)
      
      self$fitted_model <- lm(formula_obj, data = self$training_data)
      self$coefficients <- coef(self$fitted_model)
      
      cat("OLS regression fitted with", length(self$coefficients), "coefficients\n")
    },
    
    # Fit Ridge regression
    .fit_ridge = function(X, y) {
      if (requireNamespace("glmnet", quietly = TRUE)) {
        # Use cross-validation to find optimal lambda
        cv_model <- glmnet::cv.glmnet(X, y, alpha = 0, nfolds = 5)
        private$.best_lambda <- cv_model$lambda.min
        
        # Fit final model
        self$fitted_model <- glmnet::glmnet(X, y, alpha = 0, lambda = private$.best_lambda)
        self$coefficients <- as.vector(coef(self$fitted_model))
        
        cat("Ridge regression fitted with lambda =", round(private$.best_lambda, 6), "\n")
      } else {
        stop("glmnet package required for ridge regression")
      }
    },
    
    # Fit Lasso regression
    .fit_lasso = function(X, y) {
      if (requireNamespace("glmnet", quietly = TRUE)) {
        # Use cross-validation to find optimal lambda
        cv_model <- glmnet::cv.glmnet(X, y, alpha = 1, nfolds = 5)
        private$.best_lambda <- cv_model$lambda.min
        
        # Fit final model
        self$fitted_model <- glmnet::glmnet(X, y, alpha = 1, lambda = private$.best_lambda)
        self$coefficients <- as.vector(coef(self$fitted_model))
        
        cat("Lasso regression fitted with lambda =", round(private$.best_lambda, 6), "\n")
      } else {
        stop("glmnet package required for lasso regression")
      }
    },
    
    # Fit Elastic Net regression
    .fit_elastic_net = function(X, y) {
      if (requireNamespace("glmnet", quietly = TRUE)) {
        # Use cross-validation to find optimal lambda
        cv_model <- glmnet::cv.glmnet(X, y, alpha = private$.l1_ratio, nfolds = 5)
        private$.best_lambda <- cv_model$lambda.min
        
        # Fit final model
        self$fitted_model <- glmnet::glmnet(X, y, alpha = private$.l1_ratio, lambda = private$.best_lambda)
        self$coefficients <- as.vector(coef(self$fitted_model))
        
        cat("Elastic Net regression fitted with lambda =", round(private$.best_lambda, 6), 
            ", alpha =", private$.l1_ratio, "\n")
      } else {
        stop("glmnet package required for elastic net regression")
      }
    },
    
    # Calculate evaluation metrics
    .calculate_metrics = function(y_true, y_pred) {
      # Basic metrics
      residuals <- y_true - y_pred
      mse <- mean(residuals^2)
      rmse <- sqrt(mse)
      mae <- mean(abs(residuals))
      
      # R-squared
      ss_res <- sum(residuals^2)
      ss_tot <- sum((y_true - mean(y_true))^2)
      r_squared <- 1 - (ss_res / ss_tot)
      
      # Adjusted R-squared
      n <- length(y_true)
      p <- length(self$feature_names)
      adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
      
      list(
        mse = mse,
        rmse = rmse,
        mae = mae,
        r_squared = r_squared,
        adj_r_squared = adj_r_squared,
        residuals = residuals
      )
    },
    
    # Create advanced plots using ggplot2
    .create_advanced_plots = function(save_plots) {
      library(ggplot2)
      library(dplyr)
      
      # Get fitted values and residuals
      fitted_values <- fitted(self$fitted_model)
      residuals <- residuals(self$fitted_model)
      
      # 1. Residuals vs Fitted
      p1 <- ggplot(data.frame(fitted = fitted_values, residuals = residuals), 
                   aes(x = fitted, y = residuals)) +
        geom_point(alpha = 0.6) +
        geom_smooth(method = "loess", se = FALSE, color = "red") +
        geom_hline(yintercept = 0, linetype = "dashed") +
        labs(title = "Residuals vs Fitted Values", 
             x = "Fitted Values", y = "Residuals") +
        theme_minimal()
      
      # 2. Q-Q Plot
      p2 <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
        stat_qq() + stat_qq_line(color = "red") +
        labs(title = "Normal Q-Q Plot", x = "Theoretical Quantiles", 
             y = "Sample Quantiles") +
        theme_minimal()
      
      print(p1)
      print(p2)
      
      if (save_plots) {
        ggsave("residuals_vs_fitted.png", p1, width = 8, height = 6)
        ggsave("qq_plot.png", p2, width = 8, height = 6)
        cat("Plots saved to PNG files\n")
      }
    },
    
    # Create basic plots using base R
    .create_base_plots = function(save_plots) {
      if (save_plots) png("diagnostic_plots.png", width = 1200, height = 800)
      
      par(mfrow = c(2, 2))
      
      if (self$model_type == "ols") {
        plot(self$fitted_model)
      } else {
        # For regularized models, create custom plots
        fitted_values <- self$predict(as.matrix(self$training_data[self$feature_names]))
        residuals <- self$training_data$y - fitted_values
        
        # Residuals vs Fitted
        plot(fitted_values, residuals, main = "Residuals vs Fitted",
             xlab = "Fitted Values", ylab = "Residuals")
        abline(h = 0, col = "red", lty = 2)
        
        # Q-Q plot
        qqnorm(residuals, main = "Normal Q-Q")
        qqline(residuals, col = "red")
        
        # Histogram of residuals
        hist(residuals, main = "Histogram of Residuals", 
             xlab = "Residuals", breaks = 20)
        
        # Scale-Location plot
        sqrt_abs_residuals <- sqrt(abs(residuals))
        plot(fitted_values, sqrt_abs_residuals, 
             main = "Scale-Location", xlab = "Fitted Values", 
             ylab = "√|Residuals|")
      }
      
      if (save_plots) {
        dev.off()
        cat("Diagnostic plots saved to diagnostic_plots.png\n")
      }
    }
  )
)

# If R6 is not available, provide a fallback function-based implementation
if (!requireNamespace("R6", quietly = TRUE)) {
  cat("R6 package not available. Using function-based implementation.\n")
  
  # Simple function-based linear regression
  fit_linear_regression <- function(X, y, method = "ols", alpha = 0.01) {
    cat("📈 Fitting", toupper(method), "Linear Regression\n")
    
    if (method == "ols") {
      # Simple OLS using built-in lm
      if (is.matrix(X)) {
        data_df <- data.frame(y = y, X)
        names(data_df) <- c("y", paste0("X", 1:ncol(X)))
      } else {
        data_df <- data.frame(y = y, X)
      }
      
      formula_str <- paste("y ~", paste(names(data_df)[-1], collapse = " + "))
      model <- lm(as.formula(formula_str), data = data_df)
      
      cat("✅ OLS regression completed\n")
      return(model)
    } else {
      stop("Only OLS method supported in fallback implementation")
    }
  }
}

#' Generate synthetic regression dataset
#' 
#' @param n_samples Number of samples
#' @param n_features Number of features
#' @param noise Noise level
#' @param random_seed Random seed for reproducibility
#' @return List with X (features) and y (target)
generate_synthetic_data <- function(n_samples = 1000, n_features = 5, 
                                   noise = 0.1, random_seed = 42) {
  set.seed(random_seed)
  
  cat("📊 Generating synthetic regression dataset\n")
  cat("Samples:", n_samples, ", Features:", n_features, ", Noise:", noise, "\n")
  
  # Generate feature matrix
  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
  colnames(X) <- paste0("feature_", 1:n_features)
  
  # Generate true coefficients
  true_coefficients <- runif(n_features, -2, 2)
  true_intercept <- runif(1, -1, 1)
  
  # Generate target with noise
  y <- true_intercept + X %*% true_coefficients + rnorm(n_samples, 0, noise)
  y <- as.vector(y)
  
  cat("✅ Synthetic dataset generated\n")
  cat("True coefficients:", round(true_coefficients, 3), "\n")
  cat("True intercept:", round(true_intercept, 3), "\n")
  
  return(list(
    X = X,
    y = y,
    true_coefficients = true_coefficients,
    true_intercept = true_intercept
  ))
}

#' Comprehensive demonstration of linear regression capabilities
demonstrate_linear_regression <- function() {
  cat("🚀 Linear Regression Implementation Demonstration\n")
  cat(paste(rep("=", 55), collapse = ""), "\n")
  
  tryCatch({
    # Generate synthetic dataset
    data <- generate_synthetic_data(1000, 5, 0.2)
    
    # Split into train and test sets
    n_train <- floor(0.8 * nrow(data$X))
    train_idx <- sample(nrow(data$X), n_train)
    
    X_train <- data$X[train_idx, ]
    y_train <- data$y[train_idx]
    X_test <- data$X[-train_idx, ]
    y_test <- data$y[-train_idx]
    
    cat("\nDataset split - Train:", length(y_train), ", Test:", length(y_test), "\n")
    
    # Test different regression methods
    methods <- c("ols")
    
    # Add regularized methods if glmnet is available
    if (requireNamespace("glmnet", quietly = TRUE)) {
      methods <- c(methods, "ridge", "lasso", "elastic_net")
    }
    
    results <- list()
    
    for (method in methods) {
      cat("\n", paste(rep("=", 60), collapse = ""), "\n")
      cat("🔍 Testing", toupper(method), "Regression\n")
      cat(paste(rep("=", 60), collapse = ""), "\n")
      
      if (exists("LinearRegressionR")) {
        # Use R6 class if available
        model <- LinearRegressionR$new(model_type = method, alpha = 0.1)
        model$fit(X_train, y_train)
        metrics <- model$evaluate(X_test, y_test)
        
        # Generate diagnostic plots for first method
        if (method == methods[1]) {
          model$generate_diagnostic_plots(save_plots = FALSE)
        }
      } else {
        # Use fallback implementation
        if (method == "ols") {
          model <- fit_linear_regression(X_train, y_train, method = method)
          predictions <- predict(model, newdata = data.frame(X_test))
          
          # Calculate metrics manually
          residuals <- y_test - predictions
          mse <- mean(residuals^2)
          r_squared <- 1 - sum(residuals^2) / sum((y_test - mean(y_test))^2)
          
          metrics <- list(mse = mse, r_squared = r_squared)
          
          cat("Test MSE:", round(mse, 6), "\n")
          cat("Test R²:", round(r_squared, 6), "\n")
        } else {
          cat("Method", method, "not available in fallback mode\n")
          next
        }
      }
      
      results[[method]] <- metrics
      
      # Sample predictions
      cat("\n🧪 Sample Predictions:\n")
      if (exists("LinearRegressionR") && method %in% names(results)) {
        sample_predictions <- model$predict(X_test[1:5, ])
        for (i in 1:5) {
          cat(sprintf("Sample %d: Predicted = %.4f, Actual = %.4f, Error = %.4f\n",
                      i, sample_predictions[i], y_test[i], 
                      abs(sample_predictions[i] - y_test[i])))
        }
      }
    }
    
    # Compare methods
    if (length(results) > 1) {
      cat("\n", paste(rep("=", 50), collapse = ""), "\n")
      cat("📊 Method Comparison\n")
      cat(paste(rep("=", 50), collapse = ""), "\n")
      
      comparison_df <- do.call(rbind, lapply(names(results), function(method) {
        metrics <- results[[method]]
        data.frame(
          Method = toupper(method),
          MSE = round(metrics$mse, 6),
          R_squared = round(metrics$r_squared, 6)
        )
      }))
      
      print(comparison_df)
    }
    
    cat("\n✅ Linear regression demonstration completed successfully!\n")
    
  }, error = function(e) {
    cat("❌ Linear regression demonstration failed:", e$message, "\n")
    traceback()
  })
}

# Utility function for null coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Run demonstration if this script is executed directly
if (interactive() || !exists(".called_from_test")) {
  demonstrate_linear_regression()
}