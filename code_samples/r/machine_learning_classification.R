# Machine Learning - Classification Algorithms in R
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

cat("Machine Learning Classification in R\n")
cat("====================================\n")
