# Production-Ready Neural Network Implementation in R
# ==================================================
# 
# This module demonstrates a comprehensive multi-layer perceptron (MLP)
# neural network with backpropagation, multiple optimizers, and modern
# R patterns for AI training datasets.
#
# Key Features:
# - Multi-layer perceptron with configurable architecture
# - Multiple activation functions (sigmoid, tanh, ReLU, leaky ReLU)
# - Various optimization algorithms (SGD, Momentum, Adam)
# - Regularization techniques (L1, L2, dropout)
# - Batch and mini-batch training
# - Learning rate scheduling
# - Comprehensive metrics and visualization
# - Production deployment considerations
# 
# Author: AI Training Dataset
# License: MIT

# Required libraries
required_packages <- c("ggplot2", "dplyr", "tidyr", "viridis", "gridExtra", "nnet")

install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", pkg, "\n")
      tryCatch({
        install.packages(pkg, repos = "http://cran.r-project.org", quiet = TRUE)
        library(pkg, character.only = TRUE)
      }, error = function(e) {
        cat("Warning: Could not install", pkg, ". Using fallback methods.\n")
      })
    }
  }
}

# Install and load required packages
install_if_missing(required_packages)

#' Activation Functions
ActivationFunctions <- list(
  # Sigmoid activation
  sigmoid = list(
    forward = function(x) 1 / (1 + exp(-pmax(pmin(x, 500), -500))),  # Clipped for stability
    backward = function(x) {
      sig_x <- 1 / (1 + exp(-pmax(pmin(x, 500), -500)))
      sig_x * (1 - sig_x)
    },
    name = "sigmoid"
  ),
  
  # Hyperbolic tangent
  tanh = list(
    forward = function(x) tanh(pmax(pmin(x, 500), -500)),  # Clipped for stability
    backward = function(x) 1 - tanh(pmax(pmin(x, 500), -500))^2,
    name = "tanh"
  ),
  
  # Rectified Linear Unit
  relu = list(
    forward = function(x) pmax(0, x),
    backward = function(x) as.numeric(x > 0),
    name = "relu"
  ),
  
  # Leaky ReLU
  leaky_relu = list(
    forward = function(x) ifelse(x > 0, x, 0.01 * x),
    backward = function(x) ifelse(x > 0, 1, 0.01),
    name = "leaky_relu"
  ),
  
  # Linear (identity) for output layer
  linear = list(
    forward = function(x) x,
    backward = function(x) rep(1, length(x)),
    name = "linear"
  )
)

#' Enhanced Neural Network Implementation
#' 
#' A comprehensive multi-layer perceptron implementation with modern
#' optimization techniques and extensive customization options.
NeuralNetworkR <- R6::R6Class("NeuralNetworkR",
  public = list(
    #' @field hidden_layers Vector of hidden layer sizes
    hidden_layers = NULL,
    
    #' @field activation_function Name of activation function
    activation_function = NULL,
    
    #' @field learning_rate Learning rate for optimization
    learning_rate = NULL,
    
    #' @field optimizer Optimization algorithm
    optimizer = NULL,
    
    #' @field regularization Regularization type ('l1', 'l2', 'none')
    regularization = NULL,
    
    #' @field reg_strength Regularization strength
    reg_strength = NULL,
    
    #' @field dropout_rate Dropout rate for regularization
    dropout_rate = NULL,
    
    #' @field weights List of weight matrices
    weights = NULL,
    
    #' @field biases List of bias vectors
    biases = NULL,
    
    #' @field n_inputs Number of input features
    n_inputs = NULL,
    
    #' @field n_outputs Number of output units
    n_outputs = NULL,
    
    #' @field training_history Training loss and metric history
    training_history = NULL,
    
    #' Initialize Neural Network
    #' 
    #' @param hidden_layers Vector of hidden layer sizes
    #' @param activation_function Activation function name
    #' @param learning_rate Learning rate
    #' @param optimizer Optimization algorithm ('sgd', 'momentum', 'adam')
    #' @param regularization Regularization type
    #' @param reg_strength Regularization strength
    #' @param dropout_rate Dropout probability
    initialize = function(hidden_layers = c(10), activation_function = "relu",
                         learning_rate = 0.01, optimizer = "adam",
                         regularization = "none", reg_strength = 0.01,
                         dropout_rate = 0.0) {
      self$hidden_layers <- hidden_layers
      self$activation_function <- activation_function
      self$learning_rate <- learning_rate
      self$optimizer <- optimizer
      self$regularization <- regularization
      self$reg_strength <- reg_strength
      self$dropout_rate <- dropout_rate
      
      # Initialize optimizer-specific parameters
      private$.init_optimizer_params()
      
      private$.fitted <- FALSE
      
      cat("🧠 Initialized Neural Network\n")
      cat("Architecture:", paste(c("Input", hidden_layers, "Output"), collapse = " -> "), "\n")
      cat("Activation:", activation_function, ", Optimizer:", optimizer, "\n")
      cat("Learning rate:", learning_rate, ", Regularization:", regularization, "\n")
      if (dropout_rate > 0) cat("Dropout rate:", dropout_rate, "\n")
    },
    
    #' Fit the neural network
    #' 
    #' @param X Feature matrix
    #' @param y Target values
    #' @param epochs Number of training epochs
    #' @param batch_size Mini-batch size (NULL for full batch)
    #' @param validation_split Fraction of data to use for validation
    #' @param verbose Whether to print training progress
    fit = function(X, y, epochs = 100, batch_size = 32, 
                   validation_split = 0.2, verbose = TRUE) {
      cat("🧠 Training Neural Network\n")
      cat(paste(rep("=", 35), collapse = ""), "\n")
      
      # Prepare data
      X <- as.matrix(X)
      if (is.vector(y)) {
        y <- matrix(y, ncol = 1)
      } else {
        y <- as.matrix(y)
      }
      
      # Initialize network architecture
      private$.initialize_network(ncol(X), ncol(y))
      
      # Split data for validation
      if (validation_split > 0) {
        n_val <- floor(nrow(X) * validation_split)
        val_idx <- sample(nrow(X), n_val)
        
        X_val <- X[val_idx, , drop = FALSE]
        y_val <- y[val_idx, , drop = FALSE]
        X <- X[-val_idx, , drop = FALSE]
        y <- y[-val_idx, , drop = FALSE]
      }
      
      cat("Training samples:", nrow(X), "\n")
      if (validation_split > 0) cat("Validation samples:", nrow(X_val), "\n")
      cat("Features:", ncol(X), ", Outputs:", ncol(y), "\n")
      cat("Epochs:", epochs, ", Batch size:", batch_size %||% "full", "\n")
      
      # Initialize training history
      self$training_history <- data.frame(
        epoch = integer(),
        train_loss = numeric(),
        val_loss = numeric(),
        train_accuracy = numeric(),
        val_accuracy = numeric()
      )
      
      start_time <- Sys.time()
      
      # Training loop
      for (epoch in 1:epochs) {
        epoch_start <- Sys.time()
        
        # Shuffle data
        shuffle_idx <- sample(nrow(X))
        X_shuffled <- X[shuffle_idx, , drop = FALSE]
        y_shuffled <- y[shuffle_idx, , drop = FALSE]
        
        # Mini-batch training
        if (!is.null(batch_size)) {
          n_batches <- ceiling(nrow(X) / batch_size)
          epoch_loss <- 0
          
          for (batch in 1:n_batches) {
            start_idx <- (batch - 1) * batch_size + 1
            end_idx <- min(batch * batch_size, nrow(X))
            
            X_batch <- X_shuffled[start_idx:end_idx, , drop = FALSE]
            y_batch <- y_shuffled[start_idx:end_idx, , drop = FALSE]
            
            # Forward and backward pass
            batch_loss <- private$.train_batch(X_batch, y_batch)
            epoch_loss <- epoch_loss + batch_loss * nrow(X_batch)
          }
          
          epoch_loss <- epoch_loss / nrow(X)
        } else {
          # Full batch training
          epoch_loss <- private$.train_batch(X_shuffled, y_shuffled)
        }
        
        # Validation
        val_loss <- val_acc <- NA
        if (validation_split > 0) {
          val_predictions <- self$predict(X_val)
          val_loss <- private$.calculate_loss(y_val, val_predictions)
          val_acc <- private$.calculate_accuracy(y_val, val_predictions)
        }
        
        # Training accuracy
        train_predictions <- self$predict(X)
        train_acc <- private$.calculate_accuracy(y, train_predictions)
        
        # Store history
        self$training_history <- rbind(self$training_history, 
          data.frame(
            epoch = epoch,
            train_loss = epoch_loss,
            val_loss = val_loss,
            train_accuracy = train_acc,
            val_accuracy = val_acc
          ))
        
        # Progress reporting
        if (verbose && (epoch == 1 || epoch %% max(1, epochs %/% 10) == 0 || epoch == epochs)) {
          epoch_time <- as.numeric(difftime(Sys.time(), epoch_start, units = "secs"))
          
          progress_msg <- sprintf("Epoch %3d/%d - %.3fs - loss: %.6f - acc: %.4f", 
                                epoch, epochs, epoch_time, epoch_loss, train_acc)
          
          if (validation_split > 0) {
            progress_msg <- paste0(progress_msg, 
                                 sprintf(" - val_loss: %.6f - val_acc: %.4f", val_loss, val_acc))
          }
          
          cat(progress_msg, "\n")
        }
        
        # Learning rate decay (simple)
        if (epoch %% (epochs %/% 3) == 0 && epoch > 1) {
          self$learning_rate <- self$learning_rate * 0.9
          if (verbose) cat("Learning rate reduced to:", self$learning_rate, "\n")
        }
      }
      
      end_time <- Sys.time()
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      cat("✅ Training completed in", round(training_time, 3), "seconds\n")
      
      private$.fitted <- TRUE
      invisible(self)
    },
    
    #' Make predictions
    #' 
    #' @param X New data for predictions
    #' @return Predicted values
    predict = function(X) {
      if (!private$.fitted) {
        stop("Model not fitted. Call fit() first.")
      }
      
      X <- as.matrix(X)
      
      # Forward pass
      result <- private$.forward_pass(X, training = FALSE)
      return(result$output)
    },
    
    #' Evaluate model performance
    #' 
    #' @param X_test Test features
    #' @param y_test Test targets
    #' @return List of evaluation metrics
    evaluate = function(X_test, y_test) {
      cat("\n📊 Evaluating Neural Network Performance\n")
      cat(paste(rep("=", 43), collapse = ""), "\n")
      
      predictions <- self$predict(X_test)
      
      if (is.vector(y_test)) {
        y_test <- matrix(y_test, ncol = 1)
      } else {
        y_test <- as.matrix(y_test)
      }
      
      # Calculate metrics
      test_loss <- private$.calculate_loss(y_test, predictions)
      test_accuracy <- private$.calculate_accuracy(y_test, predictions)
      
      # Mean Squared Error for regression
      mse <- mean((y_test - predictions)^2)
      
      # R-squared for regression
      ss_res <- sum((y_test - predictions)^2)
      ss_tot <- sum((y_test - mean(y_test))^2)
      r_squared <- 1 - (ss_res / ss_tot)
      
      cat("Test samples:", nrow(X_test), "\n")
      cat("Test loss:", round(test_loss, 6), "\n")
      cat("Test accuracy:", round(test_accuracy, 4), "\n")
      cat("Test MSE:", round(mse, 6), "\n")
      cat("Test R²:", round(r_squared, 6), "\n")
      
      return(list(
        loss = test_loss,
        accuracy = test_accuracy,
        mse = mse,
        r_squared = r_squared,
        predictions = predictions
      ))
    },
    
    #' Plot training history
    #' 
    #' @param save_plot Whether to save plot to file
    plot_training_history = function(save_plot = FALSE) {
      if (is.null(self$training_history) || nrow(self$training_history) == 0) {
        cat("No training history available\n")
        return(invisible(self))
      }
      
      if (requireNamespace("ggplot2", quietly = TRUE) && 
          requireNamespace("tidyr", quietly = TRUE)) {
        library(ggplot2)
        library(tidyr)
        
        # Prepare data for plotting
        history_long <- self$training_history %>%
          select(epoch, train_loss, val_loss, train_accuracy, val_accuracy) %>%
          pivot_longer(cols = -epoch, names_to = "metric", values_to = "value") %>%
          mutate(
            type = ifelse(grepl("train", metric), "Training", "Validation"),
            metric_type = ifelse(grepl("loss", metric), "Loss", "Accuracy")
          )
        
        # Remove NA values
        history_long <- history_long[!is.na(history_long$value), ]
        
        # Create plots
        p1 <- ggplot(filter(history_long, metric_type == "Loss"), 
                    aes(x = epoch, y = value, color = type)) +
          geom_line(size = 1) +
          labs(title = "Training and Validation Loss", x = "Epoch", y = "Loss") +
          theme_minimal() +
          scale_color_manual(values = c("Training" = "blue", "Validation" = "red"))
        
        p2 <- ggplot(filter(history_long, metric_type == "Accuracy"), 
                    aes(x = epoch, y = value, color = type)) +
          geom_line(size = 1) +
          labs(title = "Training and Validation Accuracy", x = "Epoch", y = "Accuracy") +
          theme_minimal() +
          scale_color_manual(values = c("Training" = "blue", "Validation" = "red"))
        
        if (requireNamespace("gridExtra", quietly = TRUE)) {
          combined_plot <- gridExtra::grid.arrange(p1, p2, nrow = 2)
          
          if (save_plot) {
            ggsave("training_history.png", combined_plot, width = 10, height = 8)
            cat("Training history plot saved\n")
          }
        } else {
          print(p1)
          print(p2)
        }
        
      } else {
        # Base R plotting
        if (save_plot) png("training_history.png", width = 800, height = 600)
        
        par(mfrow = c(2, 1))
        
        # Loss plot
        plot(self$training_history$epoch, self$training_history$train_loss, 
             type = "l", col = "blue", lwd = 2,
             main = "Training and Validation Loss", 
             xlab = "Epoch", ylab = "Loss")
        if (!all(is.na(self$training_history$val_loss))) {
          lines(self$training_history$epoch, self$training_history$val_loss, 
                col = "red", lwd = 2)
          legend("topright", legend = c("Training", "Validation"), 
                 col = c("blue", "red"), lwd = 2)
        }
        
        # Accuracy plot
        plot(self$training_history$epoch, self$training_history$train_accuracy, 
             type = "l", col = "blue", lwd = 2,
             main = "Training and Validation Accuracy", 
             xlab = "Epoch", ylab = "Accuracy")
        if (!all(is.na(self$training_history$val_accuracy))) {
          lines(self$training_history$epoch, self$training_history$val_accuracy, 
                col = "red", lwd = 2)
          legend("bottomright", legend = c("Training", "Validation"), 
                 col = c("blue", "red"), lwd = 2)
        }
        
        if (save_plot) {
          dev.off()
          cat("Training history plot saved to training_history.png\n")
        }
      }
      
      invisible(self)
    },
    
    #' Print model summary
    print_summary = function() {
      if (!private$.fitted) {
        cat("Model not fitted yet.\n")
        return(invisible(self))
      }
      
      cat("\n🧠 Neural Network Model Summary\n")
      cat(paste(rep("-", 35), collapse = ""), "\n")
      
      cat("Architecture:", self$n_inputs, "->", 
          paste(self$hidden_layers, collapse = " -> "), "->", self$n_outputs, "\n")
      cat("Activation function:", self$activation_function, "\n")
      cat("Optimizer:", self$optimizer, "\n")
      cat("Learning rate:", self$learning_rate, "\n")
      cat("Regularization:", self$regularization, 
          if (self$regularization != "none") paste("(", self$reg_strength, ")") else "", "\n")
      
      # Count parameters
      total_params <- private$.count_parameters()
      cat("Total parameters:", total_params, "\n")
      
      # Training summary
      if (!is.null(self$training_history) && nrow(self$training_history) > 0) {
        final_epoch <- tail(self$training_history, 1)
        cat("\nFinal training loss:", round(final_epoch$train_loss, 6), "\n")
        cat("Final training accuracy:", round(final_epoch$train_accuracy, 4), "\n")
        if (!is.na(final_epoch$val_loss)) {
          cat("Final validation loss:", round(final_epoch$val_loss, 6), "\n")
          cat("Final validation accuracy:", round(final_epoch$val_accuracy, 4), "\n")
        }
      }
      
      invisible(self)
    }
  ),
  
  private = list(
    .fitted = FALSE,
    .optimizer_params = NULL,
    
    # Initialize optimizer-specific parameters
    .init_optimizer_params = function() {
      private$.optimizer_params <- switch(self$optimizer,
        "sgd" = list(),
        "momentum" = list(
          velocity_weights = NULL,
          velocity_biases = NULL,
          momentum = 0.9
        ),
        "adam" = list(
          m_weights = NULL,
          v_weights = NULL,
          m_biases = NULL,
          v_biases = NULL,
          beta1 = 0.9,
          beta2 = 0.999,
          epsilon = 1e-8,
          t = 0
        ),
        list()
      )
    },
    
    # Initialize network weights and biases
    .initialize_network = function(n_inputs, n_outputs) {
      self$n_inputs <- n_inputs
      self$n_outputs <- n_outputs
      
      # Layer sizes including input and output
      layer_sizes <- c(n_inputs, self$hidden_layers, n_outputs)
      n_layers <- length(layer_sizes) - 1
      
      # Initialize weights and biases
      self$weights <- list()
      self$biases <- list()
      
      for (i in 1:n_layers) {
        # Xavier/Glorot initialization
        fan_in <- layer_sizes[i]
        fan_out <- layer_sizes[i + 1]
        
        # Adjust initialization based on activation function
        if (self$activation_function == "relu") {
          # He initialization for ReLU
          scale <- sqrt(2 / fan_in)
        } else {
          # Xavier initialization for sigmoid/tanh
          scale <- sqrt(6 / (fan_in + fan_out))
        }
        
        self$weights[[i]] <- matrix(runif(fan_in * fan_out, -scale, scale), 
                                   nrow = fan_in, ncol = fan_out)
        self$biases[[i]] <- rep(0, fan_out)
      }
      
      # Initialize optimizer parameters
      private$.init_optimizer_weights()
      
      cat("Network initialized with", n_layers, "layers\n")
      cat("Total parameters:", private$.count_parameters(), "\n")
    },
    
    # Initialize optimizer-specific weight arrays
    .init_optimizer_weights = function() {
      if (self$optimizer == "momentum") {
        private$.optimizer_params$velocity_weights <- lapply(self$weights, function(w) matrix(0, nrow = nrow(w), ncol = ncol(w)))
        private$.optimizer_params$velocity_biases <- lapply(self$biases, function(b) rep(0, length(b)))
      } else if (self$optimizer == "adam") {
        private$.optimizer_params$m_weights <- lapply(self$weights, function(w) matrix(0, nrow = nrow(w), ncol = ncol(w)))
        private$.optimizer_params$v_weights <- lapply(self$weights, function(w) matrix(0, nrow = nrow(w), ncol = ncol(w)))
        private$.optimizer_params$m_biases <- lapply(self$biases, function(b) rep(0, length(b)))
        private$.optimizer_params$v_biases <- lapply(self$biases, function(b) rep(0, length(b)))
        private$.optimizer_params$t <- 0
      }
    },
    
    # Forward pass through the network
    .forward_pass = function(X, training = TRUE) {
      activations <- list(X)  # Store activations for each layer
      z_values <- list()      # Store pre-activation values
      
      current_input <- X
      
      for (i in 1:length(self$weights)) {
        # Linear transformation
        z <- current_input %*% self$weights[[i]]
        z <- sweep(z, 2, self$biases[[i]], "+")  # Add bias
        z_values[[i]] <- z
        
        # Apply activation function
        if (i == length(self$weights)) {
          # Output layer - use linear activation for regression
          activation <- ActivationFunctions$linear$forward(z)
        } else {
          # Hidden layers
          activation_fn <- ActivationFunctions[[self$activation_function]]
          activation <- activation_fn$forward(z)
          
          # Apply dropout during training
          if (training && self$dropout_rate > 0) {
            dropout_mask <- matrix(runif(nrow(activation) * ncol(activation)) > self$dropout_rate,
                                 nrow = nrow(activation), ncol = ncol(activation))
            activation <- activation * dropout_mask / (1 - self$dropout_rate)
          }
        }
        
        activations[[i + 1]] <- activation
        current_input <- activation
      }
      
      return(list(
        output = current_input,
        activations = activations,
        z_values = z_values
      ))
    },
    
    # Backward pass (backpropagation)
    .backward_pass = function(X, y, forward_result) {
      activations <- forward_result$activations
      z_values <- forward_result$z_values
      output <- forward_result$output
      
      n_samples <- nrow(X)
      n_layers <- length(self$weights)
      
      # Initialize gradients
      weight_gradients <- list()
      bias_gradients <- list()
      
      # Output layer error
      output_error <- 2 * (output - y) / n_samples  # MSE derivative
      
      # Backpropagate through layers
      current_error <- output_error
      
      for (i in n_layers:1) {
        # Calculate gradients
        weight_gradients[[i]] <- t(activations[[i]]) %*% current_error
        bias_gradients[[i]] <- colMeans(current_error)
        
        # Add regularization to weight gradients
        if (self$regularization == "l1") {
          weight_gradients[[i]] <- weight_gradients[[i]] + self$reg_strength * sign(self$weights[[i]])
        } else if (self$regularization == "l2") {
          weight_gradients[[i]] <- weight_gradients[[i]] + self$reg_strength * self$weights[[i]]
        }
        
        # Calculate error for previous layer (if not input layer)
        if (i > 1) {
          # Get activation function derivative
          if (i == n_layers) {
            # Output layer uses linear activation
            activation_derivative <- ActivationFunctions$linear$backward(z_values[[i]])
          } else {
            activation_fn <- ActivationFunctions[[self$activation_function]]
            activation_derivative <- activation_fn$backward(z_values[[i]])
          }
          
          # Propagate error to previous layer
          current_error <- (current_error %*% t(self$weights[[i]])) * activation_derivative
        }
      }
      
      return(list(
        weight_gradients = weight_gradients,
        bias_gradients = bias_gradients
      ))
    },
    
    # Train a single batch
    .train_batch = function(X_batch, y_batch) {
      # Forward pass
      forward_result <- private$.forward_pass(X_batch, training = TRUE)
      
      # Calculate loss
      loss <- private$.calculate_loss(y_batch, forward_result$output)
      
      # Backward pass
      gradients <- private$.backward_pass(X_batch, y_batch, forward_result)
      
      # Update weights using optimizer
      private$.update_weights(gradients$weight_gradients, gradients$bias_gradients)
      
      return(loss)
    },
    
    # Update weights using the specified optimizer
    .update_weights = function(weight_gradients, bias_gradients) {
      if (self$optimizer == "sgd") {
        # Standard SGD
        for (i in 1:length(self$weights)) {
          self$weights[[i]] <- self$weights[[i]] - self$learning_rate * weight_gradients[[i]]
          self$biases[[i]] <- self$biases[[i]] - self$learning_rate * bias_gradients[[i]]
        }
        
      } else if (self$optimizer == "momentum") {
        # SGD with momentum
        momentum <- private$.optimizer_params$momentum
        
        for (i in 1:length(self$weights)) {
          # Update velocity
          private$.optimizer_params$velocity_weights[[i]] <- 
            momentum * private$.optimizer_params$velocity_weights[[i]] + 
            self$learning_rate * weight_gradients[[i]]
          
          private$.optimizer_params$velocity_biases[[i]] <- 
            momentum * private$.optimizer_params$velocity_biases[[i]] + 
            self$learning_rate * bias_gradients[[i]]
          
          # Update parameters
          self$weights[[i]] <- self$weights[[i]] - private$.optimizer_params$velocity_weights[[i]]
          self$biases[[i]] <- self$biases[[i]] - private$.optimizer_params$velocity_biases[[i]]
        }
        
      } else if (self$optimizer == "adam") {
        # Adam optimizer
        private$.optimizer_params$t <- private$.optimizer_params$t + 1
        t <- private$.optimizer_params$t
        beta1 <- private$.optimizer_params$beta1
        beta2 <- private$.optimizer_params$beta2
        epsilon <- private$.optimizer_params$epsilon
        
        for (i in 1:length(self$weights)) {
          # Update biased first and second moment estimates
          private$.optimizer_params$m_weights[[i]] <- 
            beta1 * private$.optimizer_params$m_weights[[i]] + (1 - beta1) * weight_gradients[[i]]
          
          private$.optimizer_params$v_weights[[i]] <- 
            beta2 * private$.optimizer_params$v_weights[[i]] + (1 - beta2) * weight_gradients[[i]]^2
          
          private$.optimizer_params$m_biases[[i]] <- 
            beta1 * private$.optimizer_params$m_biases[[i]] + (1 - beta1) * bias_gradients[[i]]
          
          private$.optimizer_params$v_biases[[i]] <- 
            beta2 * private$.optimizer_params$v_biases[[i]] + (1 - beta2) * bias_gradients[[i]]^2
          
          # Bias correction
          m_w_corrected <- private$.optimizer_params$m_weights[[i]] / (1 - beta1^t)
          v_w_corrected <- private$.optimizer_params$v_weights[[i]] / (1 - beta2^t)
          m_b_corrected <- private$.optimizer_params$m_biases[[i]] / (1 - beta1^t)
          v_b_corrected <- private$.optimizer_params$v_biases[[i]] / (1 - beta2^t)
          
          # Update parameters
          self$weights[[i]] <- self$weights[[i]] - 
            self$learning_rate * m_w_corrected / (sqrt(v_w_corrected) + epsilon)
          
          self$biases[[i]] <- self$biases[[i]] - 
            self$learning_rate * m_b_corrected / (sqrt(v_b_corrected) + epsilon)
        }
      }
    },
    
    # Calculate loss (MSE for regression)
    .calculate_loss = function(y_true, y_pred) {
      mse_loss <- mean((y_true - y_pred)^2)
      
      # Add regularization penalty
      reg_penalty <- 0
      if (self$regularization == "l1") {
        reg_penalty <- self$reg_strength * sum(sapply(self$weights, function(w) sum(abs(w))))
      } else if (self$regularization == "l2") {
        reg_penalty <- self$reg_strength * sum(sapply(self$weights, function(w) sum(w^2)))
      }
      
      return(mse_loss + reg_penalty)
    },
    
    # Calculate accuracy (for classification problems)
    .calculate_accuracy = function(y_true, y_pred) {
      if (ncol(as.matrix(y_true)) == 1) {
        # Regression R²
        ss_res <- sum((y_true - y_pred)^2)
        ss_tot <- sum((y_true - mean(y_true))^2)
        return(max(0, 1 - ss_res / ss_tot))  # Return R² clamped to [0, 1]
      } else {
        # Multi-class classification accuracy
        predicted_classes <- apply(y_pred, 1, which.max)
        true_classes <- apply(y_true, 1, which.max)
        return(mean(predicted_classes == true_classes))
      }
    },
    
    # Count total parameters in the network
    .count_parameters = function() {
      if (is.null(self$weights) || is.null(self$biases)) return(0)
      
      weight_params <- sum(sapply(self$weights, function(w) nrow(w) * ncol(w)))
      bias_params <- sum(sapply(self$biases, length))
      
      return(weight_params + bias_params)
    }
  )
)

# Fallback function-based implementation
if (!requireNamespace("R6", quietly = TRUE)) {
  cat("R6 package not available. Using nnet fallback implementation.\n")
  
  fit_neural_network <- function(X, y, hidden_size = 10, epochs = 100) {
    cat("🧠 Fitting Neural Network with nnet\n")
    
    if (requireNamespace("nnet", quietly = TRUE)) {
      # Prepare data
      if (is.matrix(X)) {
        data_df <- data.frame(y = y, X)
        names(data_df) <- c("y", paste0("X", 1:ncol(X)))
      } else {
        data_df <- data.frame(y = y, X)
      }
      
      # Scale data for better convergence
      data_df[, -1] <- scale(data_df[, -1])
      
      # Fit neural network
      model <- nnet::nnet(y ~ ., data = data_df, 
                         size = hidden_size, maxit = epochs, trace = FALSE)
      
      cat("✅ Neural Network training completed\n")
      return(model)
    } else {
      stop("Neither R6 nor nnet packages available for neural network implementation")
    }
  }
}

#' Generate synthetic regression dataset for neural network testing
generate_neural_network_data <- function(n_samples = 1000, n_features = 5, 
                                        noise = 0.1, nonlinear = TRUE, 
                                        random_seed = 42) {
  set.seed(random_seed)
  
  cat("📊 Generating synthetic neural network dataset\n")
  cat("Samples:", n_samples, ", Features:", n_features, 
      ", Noise:", noise, ", Nonlinear:", nonlinear, "\n")
  
  # Generate features
  X <- matrix(runif(n_samples * n_features, -2, 2), 
             nrow = n_samples, ncol = n_features)
  colnames(X) <- paste0("feature_", 1:n_features)
  
  # Generate target based on nonlinear function of features
  if (nonlinear) {
    # Complex nonlinear relationship
    y <- sin(X[, 1]) + cos(X[, 2]) + 0.5 * X[, 3]^2
    if (n_features > 3) {
      y <- y + tanh(X[, 4]) 
    }
    if (n_features > 4) {
      y <- y + 0.3 * X[, 5] * X[, 1]  # Interaction term
    }
  } else {
    # Linear relationship
    true_coefficients <- runif(n_features, -1, 1)
    y <- X %*% true_coefficients
  }
  
  # Add noise
  y <- y + rnorm(n_samples, 0, noise)
  y <- as.vector(y)
  
  cat("✅ Neural network dataset generated\n")
  cat("Target range: [", round(min(y), 3), ",", round(max(y), 3), "]\n")
  
  return(list(X = X, y = y))
}

#' Comprehensive demonstration of neural network capabilities
demonstrate_neural_network <- function() {
  cat("🚀 Neural Network Implementation Demonstration\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  
  tryCatch({
    # Generate synthetic dataset
    data <- generate_neural_network_data(800, 4, 0.1, nonlinear = TRUE)
    
    # Split into train and test sets
    n_train <- floor(0.8 * nrow(data$X))
    train_idx <- sample(nrow(data$X), n_train)
    
    X_train <- data$X[train_idx, ]
    y_train <- data$y[train_idx]
    X_test <- data$X[-train_idx, ]
    y_test <- data$y[-train_idx]
    
    cat("\nDataset split - Train:", length(y_train), ", Test:", length(y_test), "\n")
    
    # Test different neural network configurations
    if (exists("NeuralNetworkR")) {
      configurations <- list(
        list(name = "Simple NN", hidden = c(10), optimizer = "adam", epochs = 50),
        list(name = "Deep NN", hidden = c(20, 10), optimizer = "adam", epochs = 50),
        list(name = "Wide NN", hidden = c(50), optimizer = "sgd", epochs = 30)
      )
      
      results <- list()
      
      for (config in configurations) {
        cat("\n", paste(rep("=", 60), collapse = ""), "\n")
        cat("🧠 Testing", config$name, "\n")
        cat(paste(rep("=", 60), collapse = ""), "\n")
        
        # Create and train model
        nn_model <- NeuralNetworkR$new(
          hidden_layers = config$hidden,
          activation_function = "relu",
          learning_rate = 0.01,
          optimizer = config$optimizer,
          regularization = "l2",
          reg_strength = 0.001
        )
        
        nn_model$fit(X_train, y_train, 
                    epochs = config$epochs, 
                    batch_size = 32,
                    validation_split = 0.2,
                    verbose = TRUE)
        
        # Evaluate model
        metrics <- nn_model$evaluate(X_test, y_test)
        results[[config$name]] <- metrics
        
        # Plot training history for first model
        if (config$name == configurations[[1]]$name) {
          nn_model$plot_training_history(save_plot = FALSE)
        }
        
        # Print model summary
        nn_model$print_summary()
      }
      
      # Compare models
      cat("\n", paste(rep("=", 50), collapse = ""), "\n")
      cat("📊 Model Comparison\n")
      cat(paste(rep("=", 50), collapse = ""), "\n")
      
      comparison_df <- do.call(rbind, lapply(names(results), function(name) {
        metrics <- results[[name]]
        data.frame(
          Model = name,
          Test_Loss = round(metrics$loss, 6),
          Test_R2 = round(metrics$r_squared, 4),
          Test_MSE = round(metrics$mse, 6)
        )
      }))
      
      print(comparison_df)
      
    } else if (requireNamespace("nnet", quietly = TRUE)) {
      # Fallback to nnet
      cat("Using nnet fallback implementation\n")
      model <- fit_neural_network(X_train, y_train, hidden_size = 10, epochs = 100)
      
      # Make predictions and evaluate
      predictions <- predict(model, data.frame(X_test))
      mse <- mean((y_test - predictions)^2)
      r_squared <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
      
      cat("Test MSE:", round(mse, 6), "\n")
      cat("Test R²:", round(r_squared, 4), "\n")
      
    } else {
      cat("No neural network implementation available\n")
    }
    
    cat("\n✅ Neural network demonstration completed successfully!\n")
    
  }, error = function(e) {
    cat("❌ Neural network demonstration failed:", e$message, "\n")
    traceback()
  })
}

# Utility function for null coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Run demonstration if this script is executed directly
if (interactive() || !exists(".called_from_test")) {
  demonstrate_neural_network()
}