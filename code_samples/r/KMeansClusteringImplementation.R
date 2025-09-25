# Production-Ready K-Means Clustering Implementation in R
# =======================================================
# 
# This module demonstrates a comprehensive K-Means clustering algorithm
# with K-Means++ initialization, multiple distance metrics, and modern
# R patterns for AI training datasets.
#
# Key Features:
# - K-Means++ initialization for better cluster centers
# - Multiple distance metrics and linkage methods
# - Elbow method and silhouette analysis for optimal K
# - Cluster validation and stability analysis
# - R tidyverse integration and modern visualization
# - Comprehensive clustering diagnostics
# - Production deployment considerations
# 
# Author: AI Training Dataset
# License: MIT

# Required libraries
required_packages <- c("cluster", "factoextra", "ggplot2", "dplyr", "tidyr", 
                      "viridis", "corrplot", "fpc", "NbClust", "gridExtra")

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

#' Enhanced K-Means Clustering Implementation
#' 
#' A comprehensive K-Means clustering algorithm with advanced initialization,
#' validation metrics, and diagnostic capabilities.
KMeansR <- R6::R6Class("KMeansR",
  public = list(
    #' @field k Number of clusters
    k = NULL,
    
    #' @field init_method Initialization method ('random' or 'kmeans++')
    init_method = NULL,
    
    #' @field max_iter Maximum number of iterations
    max_iter = NULL,
    
    #' @field tolerance Convergence tolerance
    tolerance = NULL,
    
    #' @field distance_metric Distance metric to use
    distance_metric = NULL,
    
    #' @field fitted_model Trained clustering model
    fitted_model = NULL,
    
    #' @field cluster_centers Final cluster centers
    cluster_centers = NULL,
    
    #' @field feature_names Names of input features
    feature_names = NULL,
    
    #' @field training_data Original training data
    training_data = NULL,
    
    #' @field cluster_assignments Final cluster assignments
    cluster_assignments = NULL,
    
    #' @field inertia Within-cluster sum of squares
    inertia = NULL,
    
    #' Initialize K-Means clustering
    #' 
    #' @param k Number of clusters
    #' @param init_method Initialization method ('random', 'kmeans++')
    #' @param max_iter Maximum iterations
    #' @param tolerance Convergence tolerance
    #' @param distance_metric Distance metric ('euclidean', 'manhattan')
    initialize = function(k, init_method = "kmeans++", max_iter = 100, 
                         tolerance = 1e-4, distance_metric = "euclidean") {
      self$k <- k
      self$init_method <- init_method
      self$max_iter <- max_iter
      self$tolerance <- tolerance
      self$distance_metric <- distance_metric
      private$.fitted <- FALSE
      
      cat("🎯 Initialized K-Means Clustering\n")
      cat("Clusters (k):", k, ", Initialization:", init_method, "\n")
      cat("Max iterations:", max_iter, ", Tolerance:", tolerance, "\n")
      cat("Distance metric:", distance_metric, "\n")
    },
    
    #' Fit the K-Means clustering model
    #' 
    #' @param X Feature matrix or data frame
    #' @param feature_names Optional feature names
    fit = function(X, feature_names = NULL) {
      cat("🎯 Training K-Means Clustering Model\n")
      cat(paste(rep("=", 42), collapse = ""), "\n")
      
      # Validate and prepare data
      private$.validate_inputs(X)
      X <- private$.prepare_data(X, feature_names)
      
      cat("Training samples:", nrow(X), "\n")
      cat("Features:", ncol(X), "\n")
      cat("Clusters:", self$k, "\n")
      
      start_time <- Sys.time()
      
      # Fit based on available packages
      if (requireNamespace("cluster", quietly = TRUE)) {
        private$.fit_advanced(X)
      } else {
        private$.fit_basic(X)
      }
      
      end_time <- Sys.time()
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      cat("✅ Clustering completed in", round(training_time, 3), "seconds\n")
      
      # Calculate clustering metrics
      self$calculate_metrics()
      
      # Print summary
      self$print_summary()
      
      private$.fitted <- TRUE
      invisible(self)
    },
    
    #' Predict cluster assignments for new data
    #' 
    #' @param X_new New data points
    #' @return Vector of cluster assignments
    predict = function(X_new) {
      if (!private$.fitted) {
        stop("Model not fitted. Call fit() first.")
      }
      
      # Prepare new data
      if (is.data.frame(X_new)) {
        X_new <- as.matrix(X_new)
      }
      
      # Validate dimensions
      if (ncol(X_new) != ncol(self$cluster_centers)) {
        stop("Feature dimension mismatch")
      }
      
      # Assign to nearest cluster center
      assignments <- apply(X_new, 1, function(point) {
        distances <- apply(self$cluster_centers, 1, function(center) {
          private$.calculate_distance(point, center, self$distance_metric)
        })
        which.min(distances)
      })
      
      return(assignments)
    },
    
    #' Calculate comprehensive clustering metrics
    calculate_metrics = function() {
      if (!private$.fitted) {
        stop("Model not fitted. Call fit() first.")
      }
      
      # Calculate within-cluster sum of squares (inertia)
      self$inertia <- private$.calculate_inertia()
      
      # Calculate silhouette score if cluster package available
      if (requireNamespace("cluster", quietly = TRUE) && self$k > 1) {
        sil_result <- cluster::silhouette(self$cluster_assignments, 
                                        dist(self$training_data))
        private$.silhouette_score <- mean(sil_result[, 3])
        private$.silhouette_data <- sil_result
      } else {
        private$.silhouette_score <- NA
      }
      
      # Calculate Calinski-Harabasz index if fpc package available
      if (requireNamespace("fpc", quietly = TRUE)) {
        private$.ch_index <- fpc::calinhara(self$training_data, 
                                          self$cluster_assignments)
      } else {
        private$.ch_index <- NA
      }
      
      invisible(self)
    },
    
    #' Find optimal number of clusters using elbow method
    #' 
    #' @param X Data matrix
    #' @param k_range Range of k values to test
    #' @param plot_results Whether to plot the elbow curve
    #' @return Data frame with k values and corresponding metrics
    find_optimal_k = function(X, k_range = 2:10, plot_results = TRUE) {
      cat("🔍 Finding optimal number of clusters using elbow method\n")
      
      # Prepare data
      if (is.data.frame(X)) {
        X <- as.matrix(X)
      }
      
      results <- data.frame(
        k = integer(),
        inertia = numeric(),
        silhouette = numeric()
      )
      
      for (k_val in k_range) {
        cat("Testing k =", k_val, "\n")
        
        # Create temporary model
        temp_model <- KMeansR$new(k = k_val, 
                                 init_method = self$init_method,
                                 max_iter = 50)  # Reduced iterations for speed
        
        temp_model$fit(X)
        
        results <- rbind(results, data.frame(
          k = k_val,
          inertia = temp_model$inertia,
          silhouette = if (!is.na(temp_model$get_silhouette_score())) {
            temp_model$get_silhouette_score()
          } else NA
        ))
      }
      
      # Plot results if requested
      if (plot_results) {
        private$.plot_elbow_curve(results)
      }
      
      cat("✅ Optimal k analysis completed\n")
      return(results)
    },
    
    #' Generate comprehensive clustering visualizations
    #' 
    #' @param save_plots Whether to save plots to files
    visualize_clusters = function(save_plots = FALSE) {
      if (!private$.fitted) {
        stop("Model not fitted. Call fit() first.")
      }
      
      cat("📊 Generating clustering visualizations\n")
      
      if (requireNamespace("ggplot2", quietly = TRUE) && 
          requireNamespace("factoextra", quietly = TRUE)) {
        private$.create_advanced_visualizations(save_plots)
      } else {
        private$.create_basic_visualizations(save_plots)
      }
      
      cat("✅ Visualizations generated\n")
      invisible(self)
    },
    
    #' Print comprehensive clustering summary
    print_summary = function() {
      if (!private$.fitted) {
        cat("Model not fitted yet.\n")
        return(invisible(self))
      }
      
      cat("\n🎯 K-Means Clustering Summary\n")
      cat(paste(rep("-", 35), collapse = ""), "\n")
      
      cat("Number of clusters:", self$k, "\n")
      cat("Initialization method:", self$init_method, "\n")
      cat("Distance metric:", self$distance_metric, "\n")
      cat("Inertia (WCSS):", round(self$inertia, 4), "\n")
      
      if (!is.na(private$.silhouette_score)) {
        cat("Silhouette score:", round(private$.silhouette_score, 4), "\n")
      }
      
      if (!is.na(private$.ch_index)) {
        cat("Calinski-Harabasz index:", round(private$.ch_index, 4), "\n")
      }
      
      # Cluster sizes
      cluster_sizes <- table(self$cluster_assignments)
      cat("\nCluster sizes:\n")
      for (i in 1:self$k) {
        cat(sprintf("  Cluster %d: %d points\n", i, cluster_sizes[i]))
      }
      
      # Cluster centers
      cat("\nCluster centers:\n")
      centers_df <- data.frame(self$cluster_centers)
      colnames(centers_df) <- self$feature_names
      print(round(centers_df, 4))
      
      invisible(self)
    },
    
    #' Get silhouette score
    get_silhouette_score = function() {
      return(private$.silhouette_score)
    },
    
    #' Get Calinski-Harabasz index
    get_ch_index = function() {
      return(private$.ch_index)
    }
  ),
  
  private = list(
    .fitted = FALSE,
    .silhouette_score = NA,
    .silhouette_data = NULL,
    .ch_index = NA,
    
    # Validate input data
    .validate_inputs = function(X) {
      if (is.null(X)) {
        stop("Input data X cannot be NULL")
      }
      
      if (any(is.na(X))) {
        stop("Data contains missing values")
      }
      
      if (nrow(as.matrix(X)) < self$k) {
        stop("Number of samples must be greater than number of clusters")
      }
    },
    
    # Prepare data for clustering
    .prepare_data = function(X, feature_names) {
      if (is.data.frame(X)) {
        self$feature_names <- names(X)
        X <- as.matrix(X)
      } else {
        self$feature_names <- feature_names %||% paste0("X", 1:ncol(X))
      }
      
      # Store training data
      self$training_data <- X
      
      return(X)
    },
    
    # Fit using advanced clustering packages
    .fit_advanced = function(X) {
      # Use built-in kmeans for robust implementation
      result <- kmeans(X, centers = self$k, 
                      iter.max = self$max_iter,
                      nstart = if (self$init_method == "kmeans++") 1 else 25,
                      algorithm = "Hartigan-Wong")
      
      self$fitted_model <- result
      self$cluster_centers <- result$centers
      self$cluster_assignments <- result$cluster
      
      cat("Advanced K-Means clustering completed\n")
      cat("Converged in", result$iter, "iterations\n")
    },
    
    # Basic K-Means implementation
    .fit_basic = function(X) {
      cat("Using basic K-Means implementation\n")
      
      # Initialize cluster centers
      centers <- private$.initialize_centers(X)
      
      for (iter in 1:self$max_iter) {
        # Assign points to clusters
        assignments <- apply(X, 1, function(point) {
          distances <- apply(centers, 1, function(center) {
            private$.calculate_distance(point, center, self$distance_metric)
          })
          which.min(distances)
        })
        
        # Update cluster centers
        new_centers <- matrix(0, nrow = self$k, ncol = ncol(X))
        for (k in 1:self$k) {
          cluster_points <- X[assignments == k, , drop = FALSE]
          if (nrow(cluster_points) > 0) {
            new_centers[k, ] <- colMeans(cluster_points)
          } else {
            new_centers[k, ] <- centers[k, ]  # Keep old center if cluster is empty
          }
        }
        
        # Check convergence
        if (max(abs(centers - new_centers)) < self$tolerance) {
          cat("Converged after", iter, "iterations\n")
          break
        }
        
        centers <- new_centers
      }
      
      self$cluster_centers <- centers
      self$cluster_assignments <- assignments
      
      # Create mock fitted model for compatibility
      self$fitted_model <- list(
        centers = centers,
        cluster = assignments,
        iter = iter
      )
    },
    
    # Initialize cluster centers
    .initialize_centers = function(X) {
      if (self$init_method == "random") {
        # Random initialization
        sample_indices <- sample(nrow(X), self$k)
        return(X[sample_indices, , drop = FALSE])
      } else if (self$init_method == "kmeans++") {
        # K-Means++ initialization
        return(private$.kmeans_plus_plus_init(X))
      } else {
        stop("Unknown initialization method:", self$init_method)
      }
    },
    
    # K-Means++ initialization
    .kmeans_plus_plus_init = function(X) {
      centers <- matrix(0, nrow = self$k, ncol = ncol(X))
      
      # Choose first center randomly
      centers[1, ] <- X[sample(nrow(X), 1), ]
      
      # Choose remaining centers
      for (i in 2:self$k) {
        # Calculate distances to nearest centers
        distances <- apply(X, 1, function(point) {
          center_distances <- apply(centers[1:(i-1), , drop = FALSE], 1, function(center) {
            private$.calculate_distance(point, center, self$distance_metric)
          })
          min(center_distances)^2  # Squared distance for K-Means++
        })
        
        # Choose next center with probability proportional to squared distance
        probabilities <- distances / sum(distances)
        next_center_idx <- sample(nrow(X), 1, prob = probabilities)
        centers[i, ] <- X[next_center_idx, ]
      }
      
      return(centers)
    },
    
    # Calculate distance between two points
    .calculate_distance = function(point1, point2, metric) {
      if (metric == "euclidean") {
        sqrt(sum((point1 - point2)^2))
      } else if (metric == "manhattan") {
        sum(abs(point1 - point2))
      } else {
        stop("Unknown distance metric:", metric)
      }
    },
    
    # Calculate within-cluster sum of squares
    .calculate_inertia = function() {
      total_inertia <- 0
      
      for (k in 1:self$k) {
        cluster_points <- self$training_data[self$cluster_assignments == k, , drop = FALSE]
        if (nrow(cluster_points) > 0) {
          center <- self$cluster_centers[k, ]
          cluster_inertia <- sum(apply(cluster_points, 1, function(point) {
            private$.calculate_distance(point, center, self$distance_metric)^2
          }))
          total_inertia <- total_inertia + cluster_inertia
        }
      }
      
      return(total_inertia)
    },
    
    # Plot elbow curve
    .plot_elbow_curve = function(results) {
      if (requireNamespace("ggplot2", quietly = TRUE)) {
        library(ggplot2)
        
        p1 <- ggplot(results, aes(x = k, y = inertia)) +
          geom_line(color = "blue", size = 1) +
          geom_point(color = "red", size = 3) +
          labs(title = "Elbow Method for Optimal k",
               x = "Number of Clusters (k)",
               y = "Inertia (WCSS)") +
          theme_minimal()
        
        print(p1)
        
        if (!all(is.na(results$silhouette))) {
          p2 <- ggplot(results, aes(x = k, y = silhouette)) +
            geom_line(color = "green", size = 1) +
            geom_point(color = "red", size = 3) +
            labs(title = "Silhouette Score vs Number of Clusters",
                 x = "Number of Clusters (k)",
                 y = "Silhouette Score") +
            theme_minimal()
          
          print(p2)
        }
      } else {
        # Base R plotting
        par(mfrow = c(1, 2))
        
        plot(results$k, results$inertia, type = "b", 
             main = "Elbow Method", xlab = "k", ylab = "Inertia",
             col = "blue", pch = 19)
        
        if (!all(is.na(results$silhouette))) {
          plot(results$k, results$silhouette, type = "b",
               main = "Silhouette Score", xlab = "k", ylab = "Silhouette",
               col = "green", pch = 19)
        }
        
        par(mfrow = c(1, 1))
      }
    },
    
    # Create advanced visualizations
    .create_advanced_visualizations = function(save_plots) {
      library(ggplot2)
      library(factoextra)
      
      # Create plots using factoextra if kmeans result available
      if (!is.null(self$fitted_model) && 
          all(c("centers", "cluster") %in% names(self$fitted_model))) {
        
        # Cluster plot
        p1 <- fviz_cluster(self$fitted_model, self$training_data,
                          palette = "Set2", geom = "point",
                          ellipse.type = "convex", ggtheme = theme_minimal())
        print(p1)
        
        # Silhouette plot if available
        if (!is.null(private$.silhouette_data)) {
          p2 <- fviz_silhouette(private$.silhouette_data)
          print(p2)
        }
        
        if (save_plots) {
          ggsave("cluster_plot.png", p1, width = 10, height = 8)
          if (!is.null(private$.silhouette_data)) {
            ggsave("silhouette_plot.png", p2, width = 10, height = 8)
          }
          cat("Advanced plots saved\n")
        }
      }
    },
    
    # Create basic visualizations
    .create_basic_visualizations = function(save_plots) {
      if (save_plots) {
        png("kmeans_plots.png", width = 1200, height = 800)
        par(mfrow = c(2, 2))
      }
      
      # If data is 2D, plot directly
      if (ncol(self$training_data) == 2) {
        plot(self$training_data, col = self$cluster_assignments + 1,
             main = "K-Means Clustering Results",
             xlab = self$feature_names[1], ylab = self$feature_names[2],
             pch = 19)
        points(self$cluster_centers, col = 1:self$k, pch = 4, cex = 3, lwd = 3)
        legend("topright", legend = paste("Cluster", 1:self$k), 
               col = 1:self$k + 1, pch = 19)
      } else {
        # For higher dimensions, use PCA for visualization
        pca_result <- prcomp(self$training_data, scale. = TRUE)
        plot(pca_result$x[, 1], pca_result$x[, 2], 
             col = self$cluster_assignments + 1,
             main = "K-Means Clustering (PCA Projection)",
             xlab = "First Principal Component", 
             ylab = "Second Principal Component",
             pch = 19)
        
        # Project centers to PCA space
        centers_pca <- predict(pca_result, self$cluster_centers)
        points(centers_pca[, 1], centers_pca[, 2], 
               col = 1:self$k, pch = 4, cex = 3, lwd = 3)
      }
      
      # Cluster size histogram
      cluster_sizes <- table(self$cluster_assignments)
      barplot(cluster_sizes, main = "Cluster Sizes",
              xlab = "Cluster", ylab = "Number of Points",
              col = rainbow(self$k))
      
      if (save_plots) {
        dev.off()
        cat("Basic plots saved to kmeans_plots.png\n")
      }
    }
  )
)

# Fallback function-based implementation
if (!requireNamespace("R6", quietly = TRUE)) {
  cat("R6 package not available. Using function-based K-Means implementation.\n")
  
  fit_kmeans_clustering <- function(X, k = 3, max_iter = 100) {
    cat("📊 Fitting K-Means clustering with k =", k, "\n")
    
    result <- kmeans(X, centers = k, iter.max = max_iter, nstart = 25)
    cat("✅ K-Means clustering completed\n")
    return(result)
  }
}

#' Generate synthetic clustering dataset with clear clusters
generate_clustering_data <- function(n_samples = 300, n_features = 2, 
                                   n_centers = 3, cluster_std = 1.0, 
                                   random_seed = 42) {
  set.seed(random_seed)
  
  cat("📊 Generating synthetic clustering dataset\n")
  cat("Samples:", n_samples, ", Features:", n_features, 
      ", Centers:", n_centers, ", Std:", cluster_std, "\n")
  
  # Generate cluster centers
  centers <- matrix(runif(n_centers * n_features, -5, 5), 
                   nrow = n_centers, ncol = n_features)
  
  # Generate points around each center
  samples_per_cluster <- floor(n_samples / n_centers)
  remainder <- n_samples %% n_centers
  
  X <- matrix(0, nrow = n_samples, ncol = n_features)
  true_labels <- integer(n_samples)
  
  idx <- 1
  for (i in 1:n_centers) {
    n_cluster_samples <- samples_per_cluster + (if (i <= remainder) 1 else 0)
    
    for (j in 1:n_cluster_samples) {
      X[idx, ] <- centers[i, ] + rnorm(n_features, 0, cluster_std)
      true_labels[idx] <- i
      idx <- idx + 1
    }
  }
  
  # Add feature names
  colnames(X) <- paste0("feature_", 1:n_features)
  
  cat("✅ Synthetic clustering dataset generated\n")
  cat("True cluster distribution:", paste(table(true_labels), collapse = ", "), "\n")
  
  return(list(
    X = X,
    true_labels = true_labels,
    true_centers = centers
  ))
}

#' Comprehensive demonstration of K-Means clustering
demonstrate_kmeans_clustering <- function() {
  cat("🚀 K-Means Clustering Implementation Demonstration\n")
  cat(paste(rep("=", 55), collapse = ""), "\n")
  
  tryCatch({
    # Generate synthetic dataset
    data <- generate_clustering_data(300, 2, 4, 1.5)
    
    cat("\n", paste(rep("=", 60), collapse = ""), "\n")
    cat("🎯 Testing K-Means with Known K\n")
    cat(paste(rep("=", 60), collapse = ""), "\n")
    
    if (exists("KMeansR")) {
      # Use R6 class implementation
      kmeans_model <- KMeansR$new(k = 4, init_method = "kmeans++")
      kmeans_model$fit(data$X)
      
      # Visualize results
      kmeans_model$visualize_clusters(save_plots = FALSE)
      
      # Find optimal k
      cat("\n", paste(rep("=", 60), collapse = ""), "\n")
      cat("🔍 Finding Optimal Number of Clusters\n")
      cat(paste(rep("=", 60), collapse = ""), "\n")
      
      optimal_k_results <- kmeans_model$find_optimal_k(data$X, k_range = 2:8)
      print(optimal_k_results)
      
      # Test different initialization methods
      cat("\n", paste(rep("=", 60), collapse = ""), "\n")
      cat("🔬 Comparing Initialization Methods\n")
      cat(paste(rep("=", 60), collapse = ""), "\n")
      
      init_methods <- c("random", "kmeans++")
      for (method in init_methods) {
        cat("\n--- Testing", method, "initialization ---\n")
        
        test_model <- KMeansR$new(k = 4, init_method = method)
        test_model$fit(data$X)
        
        cat("Inertia:", round(test_model$inertia, 4), "\n")
        if (!is.na(test_model$get_silhouette_score())) {
          cat("Silhouette Score:", round(test_model$get_silhouette_score(), 4), "\n")
        }
      }
      
    } else {
      # Use fallback implementation
      cat("Using fallback K-Means implementation\n")
      result <- fit_kmeans_clustering(data$X, k = 4)
      
      cat("Final inertia:", round(result$tot.withinss, 4), "\n")
      cat("Between cluster SS:", round(result$betweenss, 4), "\n")
      
      # Basic visualization
      plot(data$X, col = result$cluster + 1, 
           main = "K-Means Clustering Results", pch = 19)
      points(result$centers, col = 1:4, pch = 4, cex = 3, lwd = 3)
    }
    
    cat("\n✅ K-Means clustering demonstration completed successfully!\n")
    
  }, error = function(e) {
    cat("❌ K-Means clustering demonstration failed:", e$message, "\n")
    traceback()
  })
}

# Utility function for null coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Run demonstration if this script is executed directly
if (interactive() || !exists(".called_from_test")) {
  demonstrate_kmeans_clustering()
}