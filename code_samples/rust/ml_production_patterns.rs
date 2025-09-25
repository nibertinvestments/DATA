/*
Production-Ready Machine Learning Patterns in Rust
==================================================

This module demonstrates industry-standard ML patterns in Rust with proper
error handling, memory safety, performance optimization, and production deployment
considerations for AI training datasets.

Key Features:
- Memory safety with zero-cost abstractions
- Comprehensive error handling with Result types
- Performance-focused implementations
- Thread-safe operations with Arc and Mutex
- Efficient data processing with iterators
- Extensive documentation for AI learning
- Production-ready patterns with proper resource management

Author: AI Training Dataset
License: MIT
*/

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Custom error types for comprehensive error handling
#[derive(Debug, Clone)]
pub struct DataValidationError {
    message: String,
    field: Option<String>,
}

impl fmt::Display for DataValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.field {
            Some(field) => write!(f, "Data validation error in field '{}': {}", field, self.message),
            None => write!(f, "Data validation error: {}", self.message),
        }
    }
}

impl Error for DataValidationError {}

impl DataValidationError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            field: None,
        }
    }
    
    pub fn with_field(message: &str, field: &str) -> Self {
        Self {
            message: message.to_string(),
            field: Some(field.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelTrainingError {
    message: String,
    cause: Option<String>,
}

impl fmt::Display for ModelTrainingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.cause {
            Some(cause) => write!(f, "Model training error: {} (caused by: {})", self.message, cause),
            None => write!(f, "Model training error: {}", self.message),
        }
    }
}

impl Error for ModelTrainingError {}

impl ModelTrainingError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            cause: None,
        }
    }
    
    pub fn with_cause(message: &str, cause: &str) -> Self {
        Self {
            message: message.to_string(),
            cause: Some(cause.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictionError {
    message: String,
    index: Option<usize>,
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.index {
            Some(idx) => write!(f, "Prediction error at index {}: {}", idx, self.message),
            None => write!(f, "Prediction error: {}", self.message),
        }
    }
}

impl Error for PredictionError {}

impl PredictionError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            index: None,
        }
    }
    
    pub fn with_index(message: &str, index: usize) -> Self {
        Self {
            message: message.to_string(),
            index: Some(index),
        }
    }
}

// Core data structures
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    pub features: Vec<f64>,
    pub target: f64,
    pub id: String,
}

impl DataPoint {
    pub fn new(features: Vec<f64>, target: f64, id: Option<String>) -> Result<Self, DataValidationError> {
        // Validate features
        if features.is_empty() {
            return Err(DataValidationError::new("Features cannot be empty"));
        }
        
        for (i, &feature) in features.iter().enumerate() {
            if feature.is_nan() || feature.is_infinite() {
                return Err(DataValidationError::new(&format!(
                    "Invalid feature value at index {}: {}", i, feature
                )));
            }
        }
        
        // Validate target
        if target.is_nan() || target.is_infinite() {
            return Err(DataValidationError::new(&format!("Invalid target value: {}", target)));
        }
        
        let id = id.unwrap_or_else(|| format!("dp_{}", uuid::Uuid::new_v4()));
        
        Ok(Self { features, target, id })
    }
    
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }
}

#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub rmse: f64,
    pub mse: f64,
    pub r_squared: f64,
    pub training_time: Duration,
    pub prediction_time: Duration,
    pub model_size_bytes: usize,
    pub timestamp: std::time::SystemTime,
}

impl ModelMetrics {
    pub fn new(
        accuracy: f64,
        rmse: f64,
        mse: f64,
        r_squared: f64,
        training_time: Duration,
        prediction_time: Duration,
        model_size_bytes: usize,
    ) -> Self {
        Self {
            accuracy,
            rmse,
            mse,
            r_squared,
            training_time,
            prediction_time,
            model_size_bytes,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

impl fmt::Display for ModelMetrics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ModelMetrics{{R²={:.4}, RMSE={:.4}, TrainingTime={:?}, ModelSize={} bytes}}",
            self.r_squared, self.rmse, self.training_time, self.model_size_bytes
        )
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub row_count: usize,
    pub feature_count: usize,
    pub missing_values: HashMap<String, usize>,
}

impl ValidationResult {
    pub fn new(row_count: usize, feature_count: usize) -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            row_count,
            feature_count,
            missing_values: HashMap::new(),
        }
    }
    
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

// Data validator with comprehensive validation capabilities
pub struct DataValidator {
    min_rows: usize,
    max_missing_ratio: f64,
    required_columns: Vec<String>,
    security_checks: bool,
}

impl DataValidator {
    pub fn new() -> Self {
        Self {
            min_rows: 10,
            max_missing_ratio: 0.3,
            required_columns: Vec::new(),
            security_checks: true,
        }
    }
    
    pub fn with_min_rows(mut self, min_rows: usize) -> Self {
        self.min_rows = min_rows;
        self
    }
    
    pub fn with_max_missing_ratio(mut self, ratio: f64) -> Self {
        self.max_missing_ratio = ratio;
        self
    }
    
    pub fn with_security_checks(mut self, enabled: bool) -> Self {
        self.security_checks = enabled;
        self
    }
    
    pub fn validate_data_points(&self, data: &[DataPoint]) -> Result<ValidationResult, DataValidationError> {
        if data.is_empty() {
            let mut result = ValidationResult::new(0, 0);
            result.add_error("Dataset is empty".to_string());
            return Ok(result);
        }
        
        let mut result = ValidationResult::new(data.len(), data[0].feature_count());
        
        // Check minimum rows
        if data.len() < self.min_rows {
            result.add_error(format!("Insufficient data: {} rows < {}", data.len(), self.min_rows));
        }
        
        // Get expected feature count from first row
        let expected_features = data[0].feature_count();
        
        // Validate each data point
        let mut duplicate_check = HashMap::new();
        
        for (i, point) in data.iter().enumerate() {
            // Check feature count consistency
            if point.feature_count() != expected_features {
                result.add_error(format!(
                    "Inconsistent feature count at row {}: expected {}, got {}",
                    i, expected_features, point.feature_count()
                ));
            }
            
            // Check for invalid values (already validated in DataPoint::new, but double-check)
            for (j, &feature) in point.features.iter().enumerate() {
                if feature.is_nan() {
                    result.add_error(format!("NaN value found at row {}, feature {}", i, j));
                }
                if feature.is_infinite() {
                    result.add_error(format!("Infinite value found at row {}, feature {}", i, j));
                }
            }
            
            // Check target value
            if point.target.is_nan() {
                result.add_error(format!("NaN target at row {}", i));
            }
            if point.target.is_infinite() {
                result.add_error(format!("Infinite target at row {}", i));
            }
            
            // Check for duplicates
            let point_key = self.generate_point_key(point);
            *duplicate_check.entry(point_key).or_insert(0) += 1;
        }
        
        // Report duplicates
        let duplicate_count: usize = duplicate_check.values().filter(|&&count| count > 1)
            .map(|&count| count - 1).sum();
        
        if duplicate_count > 0 {
            let ratio = duplicate_count as f64 / data.len() as f64 * 100.0;
            result.add_warning(format!("Found {} duplicate rows ({:.2}%)", duplicate_count, ratio));
        }
        
        // Security validation if enabled
        if self.security_checks {
            self.validate_security(data, &mut result);
        }
        
        Ok(result)
    }
    
    pub fn validate_features_target(&self, x: &[Vec<f64>], y: &[f64]) -> Result<ValidationResult, DataValidationError> {
        if x.is_empty() || y.is_empty() {
            let mut result = ValidationResult::new(0, 0);
            result.add_error("Empty feature matrix or target array".to_string());
            return Ok(result);
        }
        
        let mut result = ValidationResult::new(x.len(), x[0].len());
        
        // Check shape consistency
        if x.len() != y.len() {
            result.add_error(format!("Feature and target length mismatch: {} != {}", x.len(), y.len()));
        }
        
        // Check minimum rows
        if x.len() < self.min_rows {
            result.add_error(format!("Insufficient data: {} rows < {}", x.len(), self.min_rows));
        }
        
        let expected_features = x[0].len();
        
        // Validate feature matrix
        for (i, row) in x.iter().enumerate() {
            if row.len() != expected_features {
                result.add_error(format!(
                    "Inconsistent feature count at row {}: expected {}, got {}",
                    i, expected_features, row.len()
                ));
                continue;
            }
            
            for (j, &feature) in row.iter().enumerate() {
                if feature.is_nan() {
                    result.add_error(format!("NaN value in features at [{}][{}]", i, j));
                }
                if feature.is_infinite() {
                    result.add_error(format!("Infinite value in features at [{}][{}]", i, j));
                }
            }
        }
        
        // Validate target array
        for (i, &target) in y.iter().enumerate() {
            if target.is_nan() {
                result.add_error(format!("NaN value in target at index {}", i));
            }
            if target.is_infinite() {
                result.add_error(format!("Infinite value in target at index {}", i));
            }
        }
        
        Ok(result)
    }
    
    fn generate_point_key(&self, point: &DataPoint) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &feature in &point.features {
            let rounded = (feature * 1000000.0).round() as i64;
            rounded.hash(&mut hasher); // Hash the integer representation
        }
        let target_rounded = (point.target * 1000000.0).round() as i64;
        target_rounded.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }
    
    fn validate_security(&self, data: &[DataPoint], result: &mut ValidationResult) {
        let suspicious_patterns = vec![
            "<script", "javascript:", "onload=", "onerror=",
            "'; DROP", "UNION SELECT", "../", "$(", "`",
        ];
        
        for (i, point) in data.iter().enumerate() {
            let id_lower = point.id.to_lowercase();
            for pattern in &suspicious_patterns {
                if id_lower.contains(&pattern.to_lowercase()) {
                    result.add_error(format!(
                        "Suspicious pattern '{}' detected in ID at row {}", pattern, i
                    ));
                }
            }
        }
    }
}

impl Default for DataValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ML Model trait for consistent interface
pub trait MLModel: Send + Sync {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), ModelTrainingError>;
    fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, PredictionError>;
    fn evaluate(&self, x: &[Vec<f64>], y: &[f64]) -> Result<ModelMetrics, Box<dyn Error>>;
    fn is_trained(&self) -> bool;
    fn save_model(&self) -> Result<Vec<u8>, ModelTrainingError>;
    fn load_model(&mut self, data: &[u8]) -> Result<(), ModelTrainingError>;
}

// Thread-safe linear regression implementation
pub struct SimpleLinearRegression {
    weights: Arc<Mutex<Vec<f64>>>,
    bias: Arc<Mutex<f64>>,
    is_trained: Arc<Mutex<bool>>,
    training_time: Arc<Mutex<Duration>>,
    
    // Hyperparameters
    learning_rate: f64,
    max_epochs: usize,
    tolerance: f64,
}

impl SimpleLinearRegression {
    pub fn new() -> Self {
        Self {
            weights: Arc::new(Mutex::new(Vec::new())),
            bias: Arc::new(Mutex::new(0.0)),
            is_trained: Arc::new(Mutex::new(false)),
            training_time: Arc::new(Mutex::new(Duration::ZERO)),
            learning_rate: 0.01,
            max_epochs: 1000,
            tolerance: 1e-6,
        }
    }
    
    pub fn with_hyperparameters(mut self, learning_rate: f64, max_epochs: usize, tolerance: f64) -> Self {
        self.learning_rate = learning_rate;
        self.max_epochs = max_epochs;
        self.tolerance = tolerance;
        self
    }
    
    fn calculate_predictions(&self, x: &[Vec<f64>], weights: &[f64], bias: f64) -> Vec<f64> {
        x.iter()
            .map(|row| {
                bias + row.iter().zip(weights).map(|(&feature, &weight)| feature * weight).sum::<f64>()
            })
            .collect()
    }
    
    fn calculate_cost(&self, predictions: &[f64], y: &[f64]) -> f64 {
        predictions
            .iter()
            .zip(y)
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum::<f64>()
            / (2.0 * predictions.len() as f64)
    }
}

impl MLModel for SimpleLinearRegression {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), ModelTrainingError> {
        let start_time = Instant::now();
        
        // Validate input
        if x.len() != y.len() {
            return Err(ModelTrainingError::new(&format!(
                "Feature and target length mismatch: {} != {}", x.len(), y.len()
            )));
        }
        
        if x.is_empty() {
            return Err(ModelTrainingError::new("Empty training data"));
        }
        
        let num_samples = x.len();
        let num_features = x[0].len();
        
        // Validate feature dimensions
        for (i, row) in x.iter().enumerate() {
            if row.len() != num_features {
                return Err(ModelTrainingError::new(&format!(
                    "Inconsistent feature count at row {}: expected {}, got {}",
                    i, num_features, row.len()
                )));
            }
        }
        
        // Initialize weights and bias
        let mut weights = vec![0.0; num_features];
        for weight in &mut weights {
            *weight = fastrand::f64() * 0.02 - 0.01; // Random initialization
        }
        let mut bias = 0.0;
        
        // Gradient descent with early stopping
        let mut previous_cost = f64::INFINITY;
        
        for epoch in 0..self.max_epochs {
            // Forward pass - calculate predictions and cost
            let predictions = self.calculate_predictions(x, &weights, bias);
            let cost = self.calculate_cost(&predictions, y);
            
            // Check for convergence
            if (previous_cost - cost).abs() < self.tolerance {
                println!("Model converged at epoch {} with cost {:.6}", epoch, cost);
                break;
            }
            previous_cost = cost;
            
            // Backward pass - calculate gradients
            let mut weight_gradients = vec![0.0; num_features];
            let mut bias_gradient = 0.0;
            
            for i in 0..num_samples {
                let error = predictions[i] - y[i];
                bias_gradient += error;
                
                for j in 0..num_features {
                    weight_gradients[j] += error * x[i][j];
                }
            }
            
            // Update parameters
            for j in 0..num_features {
                weights[j] -= (self.learning_rate * weight_gradients[j]) / num_samples as f64;
            }
            bias -= (self.learning_rate * bias_gradient) / num_samples as f64;
        }
        
        let training_duration = start_time.elapsed();
        
        // Store results
        *self.weights.lock().unwrap() = weights;
        *self.bias.lock().unwrap() = bias;
        *self.is_trained.lock().unwrap() = true;
        *self.training_time.lock().unwrap() = training_duration;
        
        println!("Model training completed in {:?}", training_duration);
        Ok(())
    }
    
    fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, PredictionError> {
        if !*self.is_trained.lock().unwrap() {
            return Err(PredictionError::new("Model must be trained before making predictions"));
        }
        
        if x.is_empty() {
            return Ok(Vec::new());
        }
        
        let weights = self.weights.lock().unwrap();
        let bias = *self.bias.lock().unwrap();
        
        // Validate input dimensions
        let expected_features = weights.len();
        for (i, row) in x.iter().enumerate() {
            if row.len() != expected_features {
                return Err(PredictionError::with_index(
                    &format!("Feature count mismatch: expected {}, got {}", expected_features, row.len()),
                    i,
                ));
            }
            
            // Check for invalid values
            for (j, &feature) in row.iter().enumerate() {
                if feature.is_nan() || feature.is_infinite() {
                    return Err(PredictionError::with_index(
                        &format!("Invalid feature value at position {}: {}", j, feature),
                        i,
                    ));
                }
            }
        }
        
        // Calculate predictions
        let predictions = self.calculate_predictions(x, &weights, bias);
        Ok(predictions)
    }
    
    fn evaluate(&self, x: &[Vec<f64>], y: &[f64]) -> Result<ModelMetrics, Box<dyn Error>> {
        if !*self.is_trained.lock().unwrap() {
            return Err(Box::new(ModelTrainingError::new("Model must be trained before evaluation")));
        }
        
        let start_time = Instant::now();
        let predictions = self.predict(x)?;
        let prediction_time = start_time.elapsed();
        
        // Calculate metrics
        let mut mse = 0.0;
        let mut sum_squared_total = 0.0;
        
        // Calculate mean of y
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        
        // Calculate MSE and total sum of squares
        for i in 0..y.len() {
            let error = predictions[i] - y[i];
            mse += error * error;
            sum_squared_total += (y[i] - y_mean).powi(2);
        }
        
        mse /= y.len() as f64;
        let rmse = mse.sqrt();
        
        // Calculate R-squared
        let r_squared = if sum_squared_total != 0.0 {
            1.0 - (mse * y.len() as f64) / sum_squared_total
        } else {
            0.0
        };
        
        // Estimate model size
        let weights = self.weights.lock().unwrap();
        let model_size = (weights.len() + 1) * std::mem::size_of::<f64>() + 64; // weights + bias + metadata
        
        let training_time = *self.training_time.lock().unwrap();
        
        let metrics = ModelMetrics::new(
            r_squared,
            rmse,
            mse,
            r_squared,
            training_time,
            prediction_time,
            model_size,
        );
        
        println!("Model evaluation completed. R²: {:.4}, RMSE: {:.4}", r_squared, rmse);
        Ok(metrics)
    }
    
    fn is_trained(&self) -> bool {
        *self.is_trained.lock().unwrap()
    }
    
    fn save_model(&self) -> Result<Vec<u8>, ModelTrainingError> {
        if !*self.is_trained.lock().unwrap() {
            return Err(ModelTrainingError::new("Cannot save untrained model"));
        }
        
        let weights = self.weights.lock().unwrap();
        let bias = *self.bias.lock().unwrap();
        let training_time = *self.training_time.lock().unwrap();
        
        let model_data = serde_json::json!({
            "type": "SimpleLinearRegression",
            "weights": *weights,
            "bias": bias,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "training_time_nanos": training_time.as_nanos(),
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
        
        serde_json::to_vec(&model_data).map_err(|e| {
            ModelTrainingError::with_cause("Failed to serialize model", &e.to_string())
        })
    }
    
    fn load_model(&mut self, data: &[u8]) -> Result<(), ModelTrainingError> {
        let model_data: serde_json::Value = serde_json::from_slice(data).map_err(|e| {
            ModelTrainingError::with_cause("Failed to deserialize model data", &e.to_string())
        })?;
        
        // Extract weights
        let weights_value = model_data.get("weights").ok_or_else(|| {
            ModelTrainingError::new("Missing weights in model data")
        })?;
        
        let weights: Vec<f64> = serde_json::from_value(weights_value.clone()).map_err(|e| {
            ModelTrainingError::with_cause("Invalid weights format", &e.to_string())
        })?;
        
        // Extract bias
        let bias = model_data.get("bias")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ModelTrainingError::new("Invalid bias format"))?;
        
        // Extract hyperparameters
        if let Some(lr) = model_data.get("learning_rate").and_then(|v| v.as_f64()) {
            self.learning_rate = lr;
        }
        if let Some(epochs) = model_data.get("max_epochs").and_then(|v| v.as_u64()) {
            self.max_epochs = epochs as usize;
        }
        if let Some(tol) = model_data.get("tolerance").and_then(|v| v.as_f64()) {
            self.tolerance = tol;
        }
        
        // Store loaded values
        *self.weights.lock().unwrap() = weights;
        *self.bias.lock().unwrap() = bias;
        *self.is_trained.lock().unwrap() = true;
        
        println!("Model loaded successfully");
        Ok(())
    }
}

impl Default for SimpleLinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

// Feature engineering utilities
pub struct FeatureEngineer {
    scaler_means: Arc<Mutex<HashMap<usize, f64>>>,
    scaler_stds: Arc<Mutex<HashMap<usize, f64>>>,
    is_fitted: Arc<Mutex<bool>>,
}

impl FeatureEngineer {
    pub fn new() -> Self {
        Self {
            scaler_means: Arc::new(Mutex::new(HashMap::new())),
            scaler_stds: Arc::new(Mutex::new(HashMap::new())),
            is_fitted: Arc::new(Mutex::new(false)),
        }
    }
    
    pub fn create_polynomial_features(&self, x: &[Vec<f64>], degree: usize) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        if degree < 2 || degree > 5 {
            return Err(Box::new(DataValidationError::new("Polynomial degree must be between 2 and 5")));
        }
        
        let start_time = Instant::now();
        println!("Creating polynomial features of degree {}...", degree);
        
        if x.is_empty() {
            return Ok(Vec::new());
        }
        
        let num_original_features = x[0].len();
        let mut num_poly_features = num_original_features; // Original features
        num_poly_features += num_original_features; // Squared terms
        
        if degree >= 2 {
            // Interaction terms: C(n,2) = n*(n-1)/2
            num_poly_features += (num_original_features * (num_original_features - 1)) / 2;
        }
        
        let result: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                let mut poly_features = Vec::with_capacity(num_poly_features);
                
                // Original features
                poly_features.extend_from_slice(row);
                
                // Squared terms
                for &feature in row {
                    poly_features.push(feature * feature);
                }
                
                // Interaction terms
                if degree >= 2 {
                    for i in 0..row.len() {
                        for j in (i + 1)..row.len() {
                            poly_features.push(row[i] * row[j]);
                        }
                    }
                }
                
                poly_features
            })
            .collect();
        
        let duration = start_time.elapsed();
        println!(
            "Polynomial features created in {:?}: {} -> {} features",
            duration, num_original_features, num_poly_features
        );
        
        Ok(result)
    }
    
    pub fn standard_scaler(&self, x: &[Vec<f64>], fit: bool) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        if x.is_empty() || x[0].is_empty() {
            return Err(Box::new(DataValidationError::new("Cannot scale empty data")));
        }
        
        let start_time = Instant::now();
        let num_samples = x.len();
        let num_features = x[0].len();
        
        if fit {
            let mut scaler_means = self.scaler_means.lock().unwrap();
            let mut scaler_stds = self.scaler_stds.lock().unwrap();
            
            scaler_means.clear();
            scaler_stds.clear();
            
            // Calculate means and standard deviations
            for j in 0..num_features {
                // Calculate mean
                let sum: f64 = x.iter().map(|row| row[j]).sum();
                let mean = sum / num_samples as f64;
                scaler_means.insert(j, mean);
                
                // Calculate standard deviation
                let sum_squared_diffs: f64 = x
                    .iter()
                    .map(|row| (row[j] - mean).powi(2))
                    .sum();
                let std = (sum_squared_diffs / num_samples as f64).sqrt();
                scaler_stds.insert(j, std);
            }
            
            *self.is_fitted.lock().unwrap() = true;
        }
        
        if !*self.is_fitted.lock().unwrap() {
            return Err(Box::new(DataValidationError::new("Scaler must be fitted before transform")));
        }
        
        // Apply scaling
        let scaler_means = self.scaler_means.lock().unwrap();
        let scaler_stds = self.scaler_stds.lock().unwrap();
        
        let result: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &feature)| {
                        let mean = scaler_means[&j];
                        let std = scaler_stds[&j];
                        
                        if std == 0.0 {
                            0.0 // Constant feature
                        } else {
                            (feature - mean) / std
                        }
                    })
                    .collect()
            })
            .collect();
        
        let duration = start_time.elapsed();
        println!("Standard scaling applied in {:?} to {} features", duration, num_features);
        
        Ok(result)
    }
    
    pub fn detect_outliers(&self, x: &[Vec<f64>], method: &str, threshold: f64) -> Result<Vec<usize>, Box<dyn Error>> {
        let start_time = Instant::now();
        println!("Detecting outliers using {} method...", method);
        
        if x.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut outlier_indices = std::collections::HashSet::new();
        let num_features = x[0].len();
        
        for j in 0..num_features {
            // Extract column
            let column: Vec<f64> = x.iter().map(|row| row[j]).collect();
            
            let feature_outliers = match method {
                "iqr" => self.detect_outliers_iqr(&column, threshold),
                "zscore" => self.detect_outliers_zscore(&column, threshold),
                _ => return Err(Box::new(DataValidationError::new(&format!("Unsupported outlier detection method: {}", method)))),
            };
            
            // Add to global outlier set
            for &idx in &feature_outliers {
                outlier_indices.insert(idx);
            }
        }
        
        let mut result: Vec<usize> = outlier_indices.into_iter().collect();
        result.sort_unstable();
        
        let duration = start_time.elapsed();
        println!("Outlier detection completed in {:?}. Found {} outlier rows", duration, result.len());
        
        Ok(result)
    }
    
    fn detect_outliers_iqr(&self, column: &[f64], threshold: f64) -> Vec<usize> {
        // Sort column for quartile calculation
        let mut sorted_column = column.to_vec();
        sorted_column.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_column.len();
        let q1 = sorted_column[n / 4];
        let q3 = sorted_column[3 * n / 4];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - threshold * iqr;
        let upper_bound = q3 + threshold * iqr;
        
        column
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if value < lower_bound || value > upper_bound {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
    
    fn detect_outliers_zscore(&self, column: &[f64], threshold: f64) -> Vec<usize> {
        // Calculate mean
        let mean: f64 = column.iter().sum::<f64>() / column.len() as f64;
        
        // Calculate standard deviation
        let sum_squared_diffs: f64 = column.iter().map(|&value| (value - mean).powi(2)).sum();
        let std = (sum_squared_diffs / column.len() as f64).sqrt();
        
        column
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let zscore = ((value - mean) / std).abs();
                if zscore > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

// ML utilities
pub struct MLUtilities;

impl MLUtilities {
    pub fn generate_synthetic_data(
        num_samples: usize,
        num_features: usize,
        noise_level: f64,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
        println!("Generating synthetic dataset: {} samples, {} features", num_samples, num_features);
        
        if num_samples == 0 || num_features == 0 {
            return Err(Box::new(DataValidationError::new("Number of samples and features must be positive")));
        }
        
        fastrand::seed(42); // Fixed seed for reproducibility
        
        // Generate true weights for linear relationship
        let true_weights: Vec<f64> = (0..num_features).map(|_| fastrand::f64() * 2.0 - 1.0).collect();
        let true_bias = fastrand::f64() * 2.0 - 1.0;
        
        // Generate data
        let mut x = Vec::with_capacity(num_samples);
        let mut y = Vec::with_capacity(num_samples);
        
        for _ in 0..num_samples {
            // Generate features
            let features: Vec<f64> = (0..num_features).map(|_| fastrand::f64() * 4.0 - 2.0).collect();
            
            // Generate target with linear relationship + noise
            let mut target = true_bias;
            for (j, &feature) in features.iter().enumerate() {
                target += feature * true_weights[j];
            }
            target += (fastrand::f64() * 2.0 - 1.0) * noise_level; // Add noise
            
            x.push(features);
            y.push(target);
        }
        
        Ok((x, y))
    }
    
    pub fn train_test_split(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        test_size: f64,
        random_seed: u64,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>), Box<dyn Error>> {
        if x.len() != y.len() {
            return Err(Box::new(DataValidationError::new("X and y must have the same length")));
        }
        
        if !(0.0..1.0).contains(&test_size) {
            return Err(Box::new(DataValidationError::new("test_size must be between 0 and 1")));
        }
        
        fastrand::seed(random_seed);
        
        // Create indices and shuffle
        let mut indices: Vec<usize> = (0..x.len()).collect();
        
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = fastrand::usize(0..=i);
            indices.swap(i, j);
        }
        
        let test_length = (x.len() as f64 * test_size).round() as usize;
        let train_length = x.len() - test_length;
        
        // Split data
        let mut x_train = Vec::with_capacity(train_length);
        let mut y_train = Vec::with_capacity(train_length);
        let mut x_test = Vec::with_capacity(test_length);
        let mut y_test = Vec::with_capacity(test_length);
        
        for i in 0..train_length {
            let idx = indices[i];
            x_train.push(x[idx].clone());
            y_train.push(y[idx]);
        }
        
        for i in train_length..x.len() {
            let idx = indices[i];
            x_test.push(x[idx].clone());
            y_test.push(y[idx]);
        }
        
        Ok((x_train, x_test, y_train, y_test))
    }
}

// Comprehensive demonstration function
pub fn demonstrate_rust_ml_patterns() -> Result<(), Box<dyn Error>> {
    println!("🚀 Rust ML Pipeline Demonstration");
    println!("{}", "=".repeat(40));
    
    // Generate synthetic data
    let (x, y) = MLUtilities::generate_synthetic_data(1000, 5, 0.1)?;
    println!("✅ Generated dataset: {} samples, {} features", x.len(), x[0].len());
    
    // Split data
    let (x_train, x_test, y_train, y_test) = MLUtilities::train_test_split(x, y, 0.2, 42)?;
    println!("✅ Data split: {} training, {} test samples", x_train.len(), x_test.len());
    
    // Validate data
    println!("\n🔄 Validating data...");
    let validator = DataValidator::new().with_min_rows(100).with_max_missing_ratio(0.1);
    let validation = validator.validate_features_target(&x_train, &y_train)?;
    
    if !validation.is_valid {
        println!("❌ Data validation failed: {:?}", validation.errors);
        return Ok(());
    }
    println!("✅ Data validation passed");
    
    // Train model
    println!("\n🔄 Training model...");
    let mut model = SimpleLinearRegression::new().with_hyperparameters(0.01, 1000, 1e-6);
    model.fit(&x_train, &y_train)?;
    println!("✅ Model training completed");
    
    // Make predictions
    println!("\n🔮 Making predictions...");
    let x_sample = &x_test[..x_test.len().min(10)];
    let predictions = model.predict(x_sample)?;
    
    println!(
        "Sample predictions: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        predictions.get(0).unwrap_or(&0.0),
        predictions.get(1).unwrap_or(&0.0),
        predictions.get(2).unwrap_or(&0.0),
        predictions.get(3).unwrap_or(&0.0),
        predictions.get(4).unwrap_or(&0.0)
    );
    
    // Evaluate model
    println!("\n📊 Evaluating model...");
    let metrics = model.evaluate(&x_test, &y_test)?;
    println!("R²: {:.4}", metrics.r_squared);
    println!("RMSE: {:.4}", metrics.rmse);
    println!("Training time: {:?}", metrics.training_time);
    println!("Model size: {} bytes", metrics.model_size_bytes);
    
    // Feature engineering demonstration
    println!("\n🔧 Feature Engineering...");
    let engineer = FeatureEngineer::new();
    
    // Create polynomial features
    let x_train_sample = &x_train[..x_train.len().min(100)];
    let poly_features = engineer.create_polynomial_features(x_train_sample, 2)?;
    println!(
        "Features after polynomial engineering: {} -> {}",
        x_train_sample[0].len(),
        poly_features[0].len()
    );
    
    // Standard scaling
    let scaled_features = engineer.standard_scaler(x_train_sample, true)?;
    println!("Standard scaling applied to {} features", scaled_features[0].len());
    
    // Outlier detection
    let outlier_indices = engineer.detect_outliers(&x_train, "iqr", 1.5)?;
    println!("Outlier detection: {} outlier rows found", outlier_indices.len());
    
    // Model persistence
    println!("\n💾 Testing model persistence...");
    let model_data = model.save_model()?;
    let mut new_model = SimpleLinearRegression::new();
    new_model.load_model(&model_data)?;
    println!("✅ Model save/load completed");
    
    println!("\n✅ Rust ML Pipeline demonstration completed successfully!");
    
    Ok(())
}

// External dependencies (would be in Cargo.toml)
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> String {
            format!("uuid-{}", fastrand::u64(..))
        }
    }
}

// Entry point
fn main() {
    if let Err(e) = demonstrate_rust_ml_patterns() {
        eprintln!("❌ Demonstration failed: {}", e);
        std::process::exit(1);
    }
}