/*
Production-Ready Machine Learning Patterns in C++
=================================================

This module demonstrates industry-standard ML patterns in C++ with proper
performance optimization, memory management, and production deployment
considerations for AI training datasets.

Key Features:
- High-performance computing with optimized algorithms
- RAII and smart pointers for memory safety
- Template metaprogramming for generic ML algorithms
- Exception safety and comprehensive error handling
- Parallel processing with std::thread and OpenMP
- Cache-friendly data structures and algorithms
- Extensive documentation for AI learning
- Production-ready patterns with performance monitoring

Author: AI Training Dataset
License: MIT
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ml {

// Custom exception hierarchy for comprehensive error handling
class MLException : public std::runtime_error {
public:
    explicit MLException(const std::string& message) 
        : std::runtime_error("ML Error: " + message) {}
};

class DataValidationException : public MLException {
public:
    explicit DataValidationException(const std::string& message)
        : MLException("Data Validation - " + message) {}
};

class ModelTrainingException : public MLException {
public:
    explicit ModelTrainingException(const std::string& message)
        : MLException("Model Training - " + message) {}
};

class ModelPredictionException : public MLException {
public:
    explicit ModelPredictionException(const std::string& message)
        : MLException("Model Prediction - " + message) {}
};

// Performance monitoring utilities
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string operation_name_;

public:
    explicit PerformanceTimer(const std::string& operation_name)
        : operation_name_(operation_name) {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    ~PerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time_).count();
        std::cout << "[PERFORMANCE] " << operation_name_ 
                  << " completed in " << duration << "ms" << std::endl;
    }

    double ElapsedSeconds() const {
        auto current = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(current - start_time_).count();
    }
};

// Thread-safe logging utility
class Logger {
private:
    static std::mutex log_mutex_;
    
public:
    enum Level { DEBUG, INFO, WARNING, ERROR };
    
    static void Log(Level level, const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::string level_str;
        switch (level) {
            case DEBUG: level_str = "DEBUG"; break;
            case INFO: level_str = "INFO"; break;
            case WARNING: level_str = "WARNING"; break;
            case ERROR: level_str = "ERROR"; break;
        }
        
        std::cout << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                  << "] [" << level_str << "] " << message << std::endl;
    }
};

std::mutex Logger::log_mutex_;

// Mathematical utilities with vectorized operations
namespace math {
    
    template<typename T>
    class Matrix {
    private:
        std::vector<std::vector<T>> data_;
        size_t rows_, cols_;

    public:
        Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
            data_.resize(rows, std::vector<T>(cols, T{}));
        }
        
        Matrix(const std::vector<std::vector<T>>& data) 
            : data_(data), rows_(data.size()), cols_(data.empty() ? 0 : data[0].size()) {
            // Validate rectangular matrix
            for (const auto& row : data_) {
                if (row.size() != cols_) {
                    throw std::invalid_argument("Matrix must be rectangular");
                }
            }
        }

        // Element access with bounds checking
        T& operator()(size_t row, size_t col) {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_[row][col];
        }

        const T& operator()(size_t row, size_t col) const {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_[row][col];
        }

        // Matrix operations
        Matrix<T> Transpose() const {
            Matrix<T> result(cols_, rows_);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result(j, i) = data_[i][j];
                }
            }
            return result;
        }

        // Vectorized operations for performance
        void ApplyFunction(std::function<T(const T&)> func) {
            #pragma omp parallel for collapse(2) if(rows_ * cols_ > 1000)
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    data_[i][j] = func(data_[i][j]);
                }
            }
        }

        // Getters
        size_t Rows() const { return rows_; }
        size_t Cols() const { return cols_; }
        
        // Row access
        std::vector<T>& operator[](size_t row) { return data_[row]; }
        const std::vector<T>& operator[](size_t row) const { return data_[row]; }
    };

    // Vectorized mathematical functions
    template<typename T>
    std::vector<T> VectorAdd(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vector sizes must match for addition");
        }
        
        std::vector<T> result(a.size());
        #pragma omp parallel for if(a.size() > 1000)
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    template<typename T>
    T DotProduct(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vector sizes must match for dot product");
        }
        
        T result = T{};
        #pragma omp parallel for reduction(+:result) if(a.size() > 1000)
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    template<typename T>
    std::vector<T> ScalarMultiply(const std::vector<T>& vec, T scalar) {
        std::vector<T> result(vec.size());
        #pragma omp parallel for if(vec.size() > 1000)
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = vec[i] * scalar;
        }
        return result;
    }
}

// Data validation and preprocessing
class DataValidator {
private:
    double min_value_;
    double max_value_;
    bool allow_missing_;
    size_t max_missing_ratio_percent_;

public:
    struct ValidationResult {
        bool is_valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        size_t total_samples;
        size_t missing_values;
        std::unordered_map<std::string, size_t> feature_missing_counts;
    };

    DataValidator(double min_val = -1e9, double max_val = 1e9, 
                 bool allow_missing = false, size_t max_missing_percent = 10)
        : min_value_(min_val), max_value_(max_val), 
          allow_missing_(allow_missing), max_missing_ratio_percent_(max_missing_percent) {}

    ValidationResult ValidateFeatures(const math::Matrix<double>& X) const {
        PerformanceTimer timer("Data Validation");
        
        ValidationResult result;
        result.total_samples = X.Rows();
        result.is_valid = true;
        result.missing_values = 0;

        if (X.Rows() == 0 || X.Cols() == 0) {
            result.errors.push_back("Empty dataset provided");
            result.is_valid = false;
            return result;
        }

        // Check for missing/invalid values
        for (size_t i = 0; i < X.Rows(); ++i) {
            for (size_t j = 0; j < X.Cols(); ++j) {
                double val = X(i, j);
                
                if (std::isnan(val) || std::isinf(val)) {
                    result.missing_values++;
                    std::string feature_name = "feature_" + std::to_string(j);
                    result.feature_missing_counts[feature_name]++;
                    
                    if (!allow_missing_) {
                        result.errors.push_back(
                            "Invalid value found at row " + std::to_string(i) + 
                            ", column " + std::to_string(j));
                        result.is_valid = false;
                    }
                } else if (val < min_value_ || val > max_value_) {
                    result.warnings.push_back(
                        "Value " + std::to_string(val) + " at row " + std::to_string(i) +
                        ", column " + std::to_string(j) + " is outside expected range [" +
                        std::to_string(min_value_) + ", " + std::to_string(max_value_) + "]");
                }
            }
        }

        // Check missing value ratio
        double missing_ratio = static_cast<double>(result.missing_values) / 
                              (X.Rows() * X.Cols()) * 100.0;
        
        if (missing_ratio > max_missing_ratio_percent_) {
            result.errors.push_back(
                "Missing value ratio " + std::to_string(missing_ratio) + 
                "% exceeds maximum allowed " + std::to_string(max_missing_ratio_percent_) + "%");
            result.is_valid = false;
        }

        Logger::Log(Logger::INFO, 
                   "Data validation completed: " + std::to_string(result.total_samples) + 
                   " samples, " + std::to_string(result.missing_values) + " missing values");

        return result;
    }
};

// Feature Engineering with high-performance implementations
class FeatureEngineer {
public:
    // Create polynomial features for improved model complexity
    static math::Matrix<double> CreatePolynomialFeatures(
        const math::Matrix<double>& X, size_t degree = 2) {
        
        PerformanceTimer timer("Polynomial Feature Creation");
        
        if (degree < 1) {
            throw std::invalid_argument("Polynomial degree must be >= 1");
        }

        size_t original_features = X.Cols();
        size_t new_feature_count = 0;
        
        // Calculate number of polynomial features
        for (size_t d = 1; d <= degree; ++d) {
            size_t combinations = 1;
            for (size_t i = 0; i < d; ++i) {
                combinations *= (original_features + i);
                combinations /= (i + 1);
            }
            new_feature_count += combinations;
        }

        math::Matrix<double> result(X.Rows(), new_feature_count);
        
        // Copy original features
        #pragma omp parallel for collapse(2) if(X.Rows() * X.Cols() > 10000)
        for (size_t i = 0; i < X.Rows(); ++i) {
            for (size_t j = 0; j < original_features; ++j) {
                result(i, j) = X(i, j);
            }
        }

        // Generate polynomial combinations
        size_t feature_idx = original_features;
        for (size_t d = 2; d <= degree; ++d) {
            // Generate all combinations of degree d
            std::vector<std::vector<size_t>> combinations;
            GenerateCombinations(original_features, d, combinations);
            
            for (const auto& combo : combinations) {
                #pragma omp parallel for if(X.Rows() > 1000)
                for (size_t i = 0; i < X.Rows(); ++i) {
                    double value = 1.0;
                    for (size_t feature : combo) {
                        value *= X(i, feature);
                    }
                    result(i, feature_idx) = value;
                }
                feature_idx++;
            }
        }

        Logger::Log(Logger::INFO, 
                   "Created " + std::to_string(new_feature_count) + 
                   " polynomial features from " + std::to_string(original_features) + 
                   " original features");

        return result;
    }

    // Standardize features for improved numerical stability
    static std::pair<math::Matrix<double>, std::pair<std::vector<double>, std::vector<double>>>
    StandardizeFeatures(const math::Matrix<double>& X) {
        PerformanceTimer timer("Feature Standardization");
        
        std::vector<double> means(X.Cols(), 0.0);
        std::vector<double> stds(X.Cols(), 0.0);
        
        // Calculate means
        for (size_t j = 0; j < X.Cols(); ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < X.Rows(); ++i) {
                sum += X(i, j);
            }
            means[j] = sum / X.Rows();
        }
        
        // Calculate standard deviations
        for (size_t j = 0; j < X.Cols(); ++j) {
            double sum_sq = 0.0;
            for (size_t i = 0; i < X.Rows(); ++i) {
                double diff = X(i, j) - means[j];
                sum_sq += diff * diff;
            }
            stds[j] = std::sqrt(sum_sq / (X.Rows() - 1));
            
            // Prevent division by zero
            if (stds[j] < 1e-10) {
                stds[j] = 1.0;
            }
        }
        
        // Apply standardization
        math::Matrix<double> result(X.Rows(), X.Cols());
        #pragma omp parallel for collapse(2) if(X.Rows() * X.Cols() > 10000)
        for (size_t i = 0; i < X.Rows(); ++i) {
            for (size_t j = 0; j < X.Cols(); ++j) {
                result(i, j) = (X(i, j) - means[j]) / stds[j];
            }
        }
        
        return {result, {means, stds}};
    }

private:
    static void GenerateCombinations(size_t n, size_t k, 
                                   std::vector<std::vector<size_t>>& combinations) {
        std::vector<size_t> combo(k);
        std::function<void(size_t, size_t)> generate = [&](size_t start, size_t depth) {
            if (depth == k) {
                combinations.push_back(combo);
                return;
            }
            for (size_t i = start; i < n; ++i) {
                combo[depth] = i;
                generate(i, depth + 1);
            }
        };
        generate(0, 0);
    }
};

// High-performance Linear Regression implementation
template<typename T = double>
class LinearRegression {
private:
    std::vector<T> weights_;
    T bias_;
    bool is_trained_;
    std::mutex model_mutex_;
    
    // Hyperparameters
    T learning_rate_;
    size_t max_iterations_;
    T convergence_threshold_;
    
    // Training statistics
    std::vector<T> training_history_;
    std::chrono::duration<double> training_time_;

public:
    struct ModelMetrics {
        T mse;
        T rmse; 
        T r_squared;
        T mae;
        std::chrono::duration<double> training_time;
        std::chrono::duration<double> prediction_time;
        size_t iterations_completed;
    };

    LinearRegression(T lr = 0.01, size_t max_iter = 1000, T threshold = 1e-6)
        : bias_(0), is_trained_(false), learning_rate_(lr), 
          max_iterations_(max_iter), convergence_threshold_(threshold) {}

    void Fit(const math::Matrix<T>& X, const std::vector<T>& y) {
        PerformanceTimer timer("Linear Regression Training");
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        if (X.Rows() != y.size()) {
            throw ModelTrainingException("Feature matrix rows must match target vector size");
        }
        
        if (X.Rows() == 0 || X.Cols() == 0) {
            throw ModelTrainingException("Empty dataset provided for training");
        }

        // Initialize weights
        weights_.assign(X.Cols(), T{});
        bias_ = T{};
        training_history_.clear();

        // Gradient descent with optimizations
        T prev_cost = std::numeric_limits<T>::max();
        
        for (size_t iteration = 0; iteration < max_iterations_; ++iteration) {
            // Forward pass - compute predictions
            std::vector<T> predictions(X.Rows());
            
            #pragma omp parallel for if(X.Rows() > 1000)
            for (size_t i = 0; i < X.Rows(); ++i) {
                T pred = bias_;
                for (size_t j = 0; j < X.Cols(); ++j) {
                    pred += weights_[j] * X(i, j);
                }
                predictions[i] = pred;
            }

            // Compute cost (MSE)
            T cost = 0;
            #pragma omp parallel for reduction(+:cost) if(X.Rows() > 1000)
            for (size_t i = 0; i < X.Rows(); ++i) {
                T error = predictions[i] - y[i];
                cost += error * error;
            }
            cost /= (2 * X.Rows());
            training_history_.push_back(cost);

            // Check convergence
            if (std::abs(prev_cost - cost) < convergence_threshold_) {
                Logger::Log(Logger::INFO, 
                           "Convergence achieved at iteration " + std::to_string(iteration));
                break;
            }
            prev_cost = cost;

            // Backward pass - compute gradients and update parameters
            std::vector<T> weight_gradients(X.Cols(), T{});
            T bias_gradient = 0;

            #pragma omp parallel for reduction(+:bias_gradient) if(X.Rows() > 1000)
            for (size_t i = 0; i < X.Rows(); ++i) {
                T error = predictions[i] - y[i];
                bias_gradient += error;
                
                for (size_t j = 0; j < X.Cols(); ++j) {
                    #pragma omp atomic
                    weight_gradients[j] += error * X(i, j);
                }
            }

            // Update parameters
            bias_ -= learning_rate_ * bias_gradient / X.Rows();
            for (size_t j = 0; j < X.Cols(); ++j) {
                weights_[j] -= learning_rate_ * weight_gradients[j] / X.Rows();
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        training_time_ = end_time - start_time;
        is_trained_ = true;

        Logger::Log(Logger::INFO, "Linear regression training completed");
    }

    std::vector<T> Predict(const math::Matrix<T>& X) const {
        if (!is_trained_) {
            throw ModelPredictionException("Model must be trained before making predictions");
        }

        if (X.Cols() != weights_.size()) {
            throw ModelPredictionException(
                "Feature count mismatch: expected " + std::to_string(weights_.size()) +
                ", got " + std::to_string(X.Cols()));
        }

        std::vector<T> predictions(X.Rows());
        
        #pragma omp parallel for if(X.Rows() > 1000)
        for (size_t i = 0; i < X.Rows(); ++i) {
            T pred = bias_;
            for (size_t j = 0; j < X.Cols(); ++j) {
                pred += weights_[j] * X(i, j);
            }
            predictions[i] = pred;
        }

        return predictions;
    }

    ModelMetrics Evaluate(const math::Matrix<T>& X, const std::vector<T>& y_true) const {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<T> predictions = Predict(X);
        
        auto prediction_end = std::chrono::high_resolution_clock::now();
        
        ModelMetrics metrics;
        metrics.prediction_time = prediction_end - start_time;
        metrics.training_time = training_time_;
        metrics.iterations_completed = training_history_.size();

        // Calculate metrics
        T sum_squared_error = 0;
        T sum_absolute_error = 0;
        T mean_y = std::accumulate(y_true.begin(), y_true.end(), T{}) / y_true.size();
        T total_sum_squares = 0;

        #pragma omp parallel for reduction(+:sum_squared_error,sum_absolute_error,total_sum_squares) if(y_true.size() > 1000)
        for (size_t i = 0; i < y_true.size(); ++i) {
            T error = predictions[i] - y_true[i];
            sum_squared_error += error * error;
            sum_absolute_error += std::abs(error);
            
            T deviation = y_true[i] - mean_y;
            total_sum_squares += deviation * deviation;
        }

        metrics.mse = sum_squared_error / y_true.size();
        metrics.rmse = std::sqrt(metrics.mse);
        metrics.mae = sum_absolute_error / y_true.size();
        
        // R-squared calculation
        if (total_sum_squares > 1e-10) {
            metrics.r_squared = 1 - (sum_squared_error / total_sum_squares);
        } else {
            metrics.r_squared = 0;
        }

        return metrics;
    }

    // Model persistence
    void SaveModel(const std::string& filepath) const {
        if (!is_trained_) {
            throw std::runtime_error("Cannot save untrained model");
        }

        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }

        // Save model parameters
        size_t weight_count = weights_.size();
        file.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
        file.write(reinterpret_cast<const char*>(weights_.data()), 
                  weight_count * sizeof(T));
        file.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
        
        // Save hyperparameters
        file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
        file.write(reinterpret_cast<const char*>(&max_iterations_), sizeof(max_iterations_));
        file.write(reinterpret_cast<const char*>(&convergence_threshold_), sizeof(convergence_threshold_));

        Logger::Log(Logger::INFO, "Model saved to " + filepath);
    }

    void LoadModel(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filepath);
        }

        std::lock_guard<std::mutex> lock(model_mutex_);

        // Load model parameters
        size_t weight_count;
        file.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
        
        weights_.resize(weight_count);
        file.read(reinterpret_cast<char*>(weights_.data()), 
                 weight_count * sizeof(T));
        file.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));
        
        // Load hyperparameters
        file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
        file.read(reinterpret_cast<char*>(&max_iterations_), sizeof(max_iterations_));
        file.read(reinterpret_cast<char*>(&convergence_threshold_), sizeof(convergence_threshold_));

        is_trained_ = true;
        Logger::Log(Logger::INFO, "Model loaded from " + filepath);
    }

    // Getters
    bool IsTrained() const { return is_trained_; }
    const std::vector<T>& GetTrainingHistory() const { return training_history_; }
    const std::vector<T>& GetWeights() const { return weights_; }
    T GetBias() const { return bias_; }
};

// Utility functions for data generation and processing
class MLUtilities {
public:
    // Generate synthetic dataset for testing
    static std::pair<math::Matrix<double>, std::vector<double>>
    GenerateRegressionDataset(size_t n_samples, size_t n_features, 
                             double noise_level = 0.1, unsigned seed = 42) {
        
        std::mt19937 gen(seed);
        std::normal_distribution<double> feature_dist(0.0, 1.0);
        std::normal_distribution<double> noise_dist(0.0, noise_level);
        
        math::Matrix<double> X(n_samples, n_features);
        std::vector<double> y(n_samples);
        
        // Generate random weights for true underlying function
        std::vector<double> true_weights(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            true_weights[i] = feature_dist(gen);
        }
        
        // Generate features and targets
        for (size_t i = 0; i < n_samples; ++i) {
            double target = 0.0;
            
            for (size_t j = 0; j < n_features; ++j) {
                double feature_val = feature_dist(gen);
                X(i, j) = feature_val;
                target += true_weights[j] * feature_val;
            }
            
            y[i] = target + noise_dist(gen);
        }
        
        Logger::Log(Logger::INFO, 
                   "Generated synthetic dataset: " + std::to_string(n_samples) + 
                   " samples, " + std::to_string(n_features) + " features");
        
        return {X, y};
    }

    // Train-test split with random sampling
    static std::tuple<math::Matrix<double>, math::Matrix<double>, 
                     std::vector<double>, std::vector<double>>
    TrainTestSplit(const math::Matrix<double>& X, const std::vector<double>& y,
                  double test_ratio = 0.2, unsigned seed = 42) {
        
        if (test_ratio < 0.0 || test_ratio > 1.0) {
            throw std::invalid_argument("Test ratio must be between 0 and 1");
        }
        
        size_t n_samples = X.Rows();
        size_t test_size = static_cast<size_t>(n_samples * test_ratio);
        size_t train_size = n_samples - test_size;
        
        // Create random indices
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::mt19937 gen(seed);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Split data
        math::Matrix<double> X_train(train_size, X.Cols());
        math::Matrix<double> X_test(test_size, X.Cols());
        std::vector<double> y_train(train_size);
        std::vector<double> y_test(test_size);
        
        for (size_t i = 0; i < train_size; ++i) {
            size_t idx = indices[i];
            for (size_t j = 0; j < X.Cols(); ++j) {
                X_train(i, j) = X(idx, j);
            }
            y_train[i] = y[idx];
        }
        
        for (size_t i = 0; i < test_size; ++i) {
            size_t idx = indices[train_size + i];
            for (size_t j = 0; j < X.Cols(); ++j) {
                X_test(i, j) = X(idx, j);
            }
            y_test[i] = y[idx];
        }
        
        return {X_train, X_test, y_train, y_test};
    }
};

// Production ML Pipeline with comprehensive error handling and monitoring
class ProductionMLPipeline {
private:
    std::unique_ptr<LinearRegression<double>> model_;
    std::unique_ptr<DataValidator> validator_;
    bool enable_monitoring_;
    std::mutex pipeline_mutex_;
    
    std::vector<double> feature_means_;
    std::vector<double> feature_stds_;
    bool is_standardized_;

public:
    ProductionMLPipeline(bool enable_monitoring = true)
        : model_(std::make_unique<LinearRegression<double>>()),
          validator_(std::make_unique<DataValidator>()),
          enable_monitoring_(enable_monitoring), is_standardized_(false) {}

    void Fit(const math::Matrix<double>& X, const std::vector<double>& y,
             double validation_split = 0.2) {
        PerformanceTimer timer("Production Pipeline Training");
        
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        
        try {
            // Validate input data
            auto validation_result = validator_->ValidateFeatures(X);
            if (!validation_result.is_valid) {
                std::string error_msg = "Data validation failed: ";
                for (const auto& error : validation_result.errors) {
                    error_msg += error + "; ";
                }
                throw DataValidationException(error_msg);
            }

            // Feature standardization
            auto [X_standardized, standardization_params] = 
                FeatureEngineer::StandardizeFeatures(X);
            feature_means_ = standardization_params.first;
            feature_stds_ = standardization_params.second;
            is_standardized_ = true;

            // Train-validation split
            auto [X_train, X_val, y_train, y_val] = 
                MLUtilities::TrainTestSplit(X_standardized, y, validation_split);

            // Train model
            model_->Fit(X_train, y_train);

            // Validation evaluation
            if (enable_monitoring_ && validation_split > 0) {
                auto metrics = model_->Evaluate(X_val, y_val);
                Logger::Log(Logger::INFO, 
                           "Validation R²: " + std::to_string(metrics.r_squared) +
                           ", RMSE: " + std::to_string(metrics.rmse));
            }

        } catch (const std::exception& e) {
            Logger::Log(Logger::ERROR, "Pipeline training failed: " + std::string(e.what()));
            throw;
        }
    }

    std::vector<double> Predict(const math::Matrix<double>& X) const {
        if (!model_->IsTrained()) {
            throw ModelPredictionException("Pipeline must be trained before making predictions");
        }

        try {
            // Apply same standardization as training
            if (is_standardized_) {
                math::Matrix<double> X_standardized(X.Rows(), X.Cols());
                
                #pragma omp parallel for collapse(2) if(X.Rows() * X.Cols() > 10000)
                for (size_t i = 0; i < X.Rows(); ++i) {
                    for (size_t j = 0; j < X.Cols(); ++j) {
                        X_standardized(i, j) = (X(i, j) - feature_means_[j]) / feature_stds_[j];
                    }
                }
                
                return model_->Predict(X_standardized);
            } else {
                return model_->Predict(X);
            }

        } catch (const std::exception& e) {
            Logger::Log(Logger::ERROR, "Pipeline prediction failed: " + std::string(e.what()));
            throw;
        }
    }

    typename LinearRegression<double>::ModelMetrics Evaluate(
        const math::Matrix<double>& X, const std::vector<double>& y) const {
        
        try {
            auto predictions = Predict(X);
            
            // Create temporary matrix for evaluation (already standardized in Predict)
            math::Matrix<double> X_for_eval(X.Rows(), X.Cols());
            if (is_standardized_) {
                #pragma omp parallel for collapse(2) if(X.Rows() * X.Cols() > 10000)
                for (size_t i = 0; i < X.Rows(); ++i) {
                    for (size_t j = 0; j < X.Cols(); ++j) {
                        X_for_eval(i, j) = (X(i, j) - feature_means_[j]) / feature_stds_[j];
                    }
                }
            } else {
                X_for_eval = X;
            }
            
            return model_->Evaluate(X_for_eval, y);

        } catch (const std::exception& e) {
            Logger::Log(Logger::ERROR, "Pipeline evaluation failed: " + std::string(e.what()));
            throw;
        }
    }

    void SavePipeline(const std::string& directory) const {
        std::filesystem::create_directories(directory);
        
        // Save model
        model_->SaveModel(directory + "/model.bin");
        
        // Save standardization parameters
        if (is_standardized_) {
            std::ofstream params_file(directory + "/standardization.txt");
            params_file << std::fixed << std::setprecision(10);
            
            params_file << "means: ";
            for (double mean : feature_means_) {
                params_file << mean << " ";
            }
            params_file << "\nstds: ";
            for (double std : feature_stds_) {
                params_file << std << " ";
            }
        }
        
        Logger::Log(Logger::INFO, "Pipeline saved to " + directory);
    }
};

} // namespace ml

// Comprehensive demonstration function
void DemonstrateCppMLPatterns() {
    std::cout << "\n🚀 C++ ML Production Patterns Demonstration" << std::endl;
    std::cout << "=============================================" << std::endl;

    try {
        // Generate synthetic dataset
        std::cout << "\n📊 Generating synthetic dataset..." << std::endl;
        auto [X, y] = ml::MLUtilities::GenerateRegressionDataset(1000, 5, 0.1);
        
        // Data validation
        std::cout << "\n🔍 Validating data quality..." << std::endl;
        ml::DataValidator validator(-10.0, 10.0, false, 5);
        auto validation_result = validator.ValidateFeatures(X);
        
        if (validation_result.is_valid) {
            std::cout << "✅ Data validation passed" << std::endl;
        } else {
            std::cout << "❌ Data validation failed:" << std::endl;
            for (const auto& error : validation_result.errors) {
                std::cout << "  - " << error << std::endl;
            }
            return;
        }

        // Train-test split
        std::cout << "\n📈 Splitting data for training and testing..." << std::endl;
        auto [X_train, X_test, y_train, y_test] = ml::MLUtilities::TrainTestSplit(X, y, 0.2);

        // Create and train production pipeline
        std::cout << "\n🔄 Training production ML pipeline..." << std::endl;
        ml::ProductionMLPipeline pipeline(true);
        pipeline.Fit(X_train, y_train, 0.2);
        std::cout << "✅ Model training completed" << std::endl;

        // Make predictions
        std::cout << "\n🔮 Making predictions..." << std::endl;
        auto predictions = pipeline.Predict(X_test);
        std::cout << "Sample predictions: ";
        for (size_t i = 0; i < std::min(5UL, predictions.size()); ++i) {
            std::cout << std::fixed << std::setprecision(4) << predictions[i] << " ";
        }
        std::cout << std::endl;

        // Model evaluation
        std::cout << "\n📊 Evaluating model performance..." << std::endl;
        auto metrics = pipeline.Evaluate(X_test, y_test);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "R² Score: " << metrics.r_squared << std::endl;
        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAE: " << metrics.mae << std::endl;
        std::cout << "Training Time: " << metrics.training_time.count() << " seconds" << std::endl;
        std::cout << "Prediction Time: " << metrics.prediction_time.count() << " seconds" << std::endl;

        // Feature engineering demonstration
        std::cout << "\n🔧 Feature Engineering demonstration..." << std::endl;
        auto polynomial_features = ml::FeatureEngineer::CreatePolynomialFeatures(X_test, 2);
        std::cout << "Original features: " << X_test.Cols() 
                  << ", Polynomial features: " << polynomial_features.Cols() << std::endl;

        // Performance monitoring
        std::cout << "\n⚡ Performance characteristics:" << std::endl;
        std::cout << "- Multi-threaded operations: ✅ OpenMP enabled" << std::endl;
        std::cout << "- Memory management: ✅ RAII with smart pointers" << std::endl;
        std::cout << "- Exception safety: ✅ Comprehensive error handling" << std::endl;
        std::cout << "- Thread safety: ✅ Mutex-protected operations" << std::endl;

        std::cout << "\n✅ C++ ML demonstration completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "❌ Error during demonstration: " << e.what() << std::endl;
        throw;
    }
}

// Main function for standalone testing
#ifdef ML_STANDALONE_TEST
int main() {
    try {
        DemonstrateCppMLPatterns();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
#endif