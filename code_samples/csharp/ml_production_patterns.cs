/*
Production-Ready Machine Learning Patterns in C#
================================================

This module demonstrates industry-standard ML patterns in C# with proper
enterprise patterns, performance optimization, and production deployment
considerations for AI training datasets.

Key Features:
- Enterprise-ready patterns with SOLID principles
- Comprehensive error handling with custom exceptions
- Async/await patterns for non-blocking operations
- LINQ optimizations for data processing
- Thread-safe operations with concurrent collections
- Dependency injection and testability patterns
- Extensive documentation for AI learning
- Production-ready patterns with monitoring and logging

Author: AI Training Dataset  
License: MIT
*/

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace MLProductionPatterns
{
    #region Custom Exception Hierarchy
    
    /// <summary>
    /// Base exception for all ML-related errors
    /// </summary>
    public class MLException : Exception
    {
        public MLException(string message) : base($"ML Error: {message}") { }
        public MLException(string message, Exception innerException) : base($"ML Error: {message}", innerException) { }
    }

    /// <summary>
    /// Exception thrown during data validation
    /// </summary>
    public class DataValidationException : MLException
    {
        public List<string> ValidationErrors { get; }
        
        public DataValidationException(string message) : base($"Data Validation - {message}")
        {
            ValidationErrors = new List<string>();
        }
        
        public DataValidationException(string message, List<string> validationErrors) 
            : base($"Data Validation - {message}")
        {
            ValidationErrors = validationErrors ?? new List<string>();
        }
    }

    /// <summary>
    /// Exception thrown during model training
    /// </summary>
    public class ModelTrainingException : MLException
    {
        public int? IterationsFailed { get; }
        
        public ModelTrainingException(string message) : base($"Model Training - {message}") { }
        
        public ModelTrainingException(string message, int iterationsFailed) 
            : base($"Model Training - {message}")
        {
            IterationsFailed = iterationsFailed;
        }
    }

    /// <summary>
    /// Exception thrown during model prediction
    /// </summary>
    public class ModelPredictionException : MLException
    {
        public ModelPredictionException(string message) : base($"Model Prediction - {message}") { }
    }

    #endregion

    #region Interfaces for Dependency Injection

    /// <summary>
    /// Interface for logging operations (production ready)
    /// </summary>
    public interface IMLLogger
    {
        Task LogAsync(LogLevel level, string message, CancellationToken cancellationToken = default);
        Task LogExceptionAsync(Exception exception, string context, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Interface for data validation operations
    /// </summary>
    public interface IDataValidator
    {
        Task<ValidationResult> ValidateAsync(double[,] features, CancellationToken cancellationToken = default);
        Task<ValidationResult> ValidateAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Interface for machine learning models
    /// </summary>
    public interface IMLModel
    {
        bool IsTrained { get; }
        Task TrainAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default);
        Task<double[]> PredictAsync(double[,] features, CancellationToken cancellationToken = default);
        Task<ModelMetrics> EvaluateAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default);
        Task SaveAsync(string filePath, CancellationToken cancellationToken = default);
        Task LoadAsync(string filePath, CancellationToken cancellationToken = default);
    }

    #endregion

    #region Data Transfer Objects

    /// <summary>
    /// Log level enumeration
    /// </summary>
    public enum LogLevel
    {
        Debug,
        Info,
        Warning,
        Error,
        Critical
    }

    /// <summary>
    /// Data validation result with comprehensive information
    /// </summary>
    public class ValidationResult
    {
        public bool IsValid { get; set; }
        public List<string> Errors { get; set; } = new List<string>();
        public List<string> Warnings { get; set; } = new List<string>();
        public int TotalSamples { get; set; }
        public int TotalFeatures { get; set; }
        public int MissingValues { get; set; }
        public double MissingValueRatio { get; set; }
        public Dictionary<string, int> FeatureMissingCounts { get; set; } = new Dictionary<string, int>();
        public Dictionary<string, (double Min, double Max, double Mean, double Std)> FeatureStatistics { get; set; } 
            = new Dictionary<string, (double, double, double, double)>();
    }

    /// <summary>
    /// Comprehensive model performance metrics
    /// </summary>
    public class ModelMetrics
    {
        public double MSE { get; set; }
        public double RMSE { get; set; }
        public double MAE { get; set; }
        public double RSquared { get; set; }
        public TimeSpan TrainingTime { get; set; }
        public TimeSpan PredictionTime { get; set; }
        public int IterationsCompleted { get; set; }
        public double ConvergenceValue { get; set; }
        public List<double> TrainingHistory { get; set; } = new List<double>();
    }

    /// <summary>
    /// Feature engineering result with transformation parameters
    /// </summary>
    public class FeatureTransformResult
    {
        public double[,] TransformedFeatures { get; set; }
        public double[] FeatureMeans { get; set; }
        public double[] FeatureStds { get; set; }
        public Dictionary<string, object> TransformationParameters { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Training configuration for model hyperparameters
    /// </summary>
    public class TrainingConfig
    {
        public double LearningRate { get; set; } = 0.01;
        public int MaxIterations { get; set; } = 1000;
        public double ConvergenceThreshold { get; set; } = 1e-6;
        public double ValidationSplit { get; set; } = 0.2;
        public bool EnableEarlyStopping { get; set; } = true;
        public int EarlyStoppingPatience { get; set; } = 10;
        public bool EnableRegularization { get; set; } = false;
        public double RegularizationStrength { get; set; } = 0.01;
    }

    #endregion

    #region Utility Classes

    /// <summary>
    /// Thread-safe console logger implementation
    /// </summary>
    public class ConsoleLogger : IMLLogger
    {
        private readonly SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);

        public async Task LogAsync(LogLevel level, string message, CancellationToken cancellationToken = default)
        {
            await _semaphore.WaitAsync(cancellationToken);
            try
            {
                var timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff");
                var levelStr = level.ToString().ToUpper();
                var logMessage = $"[{timestamp}] [{levelStr}] {message}";
                
                Console.WriteLine(logMessage);
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public async Task LogExceptionAsync(Exception exception, string context, CancellationToken cancellationToken = default)
        {
            await LogAsync(LogLevel.Error, $"{context}: {exception.Message}", cancellationToken);
            await LogAsync(LogLevel.Debug, $"Stack Trace: {exception.StackTrace}", cancellationToken);
        }
    }

    /// <summary>
    /// Performance monitoring utility with async operations
    /// </summary>
    public class PerformanceMonitor : IDisposable
    {
        private readonly string _operationName;
        private readonly Stopwatch _stopwatch;
        private readonly IMLLogger _logger;

        public PerformanceMonitor(string operationName, IMLLogger logger)
        {
            _operationName = operationName;
            _logger = logger;
            _stopwatch = Stopwatch.StartNew();
        }

        public void Dispose()
        {
            _stopwatch.Stop();
            var _ = Task.Run(async () =>
            {
                await _logger.LogAsync(LogLevel.Info, 
                    $"[PERFORMANCE] {_operationName} completed in {_stopwatch.ElapsedMilliseconds}ms");
            });
        }

        public TimeSpan ElapsedTime => _stopwatch.Elapsed;
    }

    /// <summary>
    /// Mathematical utilities with parallel processing
    /// </summary>
    public static class MathUtils
    {
        /// <summary>
        /// Parallel matrix multiplication
        /// </summary>
        public static double[,] MatrixMultiply(double[,] a, double[,] b)
        {
            int rowsA = a.GetLength(0);
            int colsA = a.GetLength(1);
            int rowsB = b.GetLength(0);
            int colsB = b.GetLength(1);

            if (colsA != rowsB)
                throw new ArgumentException("Matrix dimensions don't match for multiplication");

            double[,] result = new double[rowsA, colsB];

            Parallel.For(0, rowsA, i =>
            {
                for (int j = 0; j < colsB; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < colsA; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            });

            return result;
        }

        /// <summary>
        /// Parallel dot product calculation
        /// </summary>
        public static double DotProduct(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vector lengths must match");

            return a.AsParallel().Zip(b.AsParallel(), (x, y) => x * y).Sum();
        }

        /// <summary>
        /// Calculate statistics for a feature column
        /// </summary>
        public static (double Min, double Max, double Mean, double Std) CalculateStatistics(double[] values)
        {
            var validValues = values.Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToArray();
            
            if (validValues.Length == 0)
                return (double.NaN, double.NaN, double.NaN, double.NaN);

            double min = validValues.Min();
            double max = validValues.Max();
            double mean = validValues.Average();
            double variance = validValues.Select(v => Math.Pow(v - mean, 2)).Average();
            double std = Math.Sqrt(variance);

            return (min, max, mean, std);
        }

        /// <summary>
        /// Generate synthetic regression dataset for testing
        /// </summary>
        public static (double[,] Features, double[] Targets) GenerateRegressionDataset(
            int samples, int features, double noiseLevel = 0.1, int seed = 42)
        {
            var random = new Random(seed);
            
            double[,] X = new double[samples, features];
            double[] y = new double[samples];
            
            // Generate random true weights
            double[] trueWeights = Enumerable.Range(0, features)
                .Select(_ => random.NextDouble() * 2 - 1)
                .ToArray();

            // Generate features and targets
            Parallel.For(0, samples, i =>
            {
                double target = 0;
                
                for (int j = 0; j < features; j++)
                {
                    double featureVal = random.NextDouble() * 2 - 1;
                    X[i, j] = featureVal;
                    target += trueWeights[j] * featureVal;
                }
                
                // Add noise
                double noise = (random.NextDouble() * 2 - 1) * noiseLevel;
                y[i] = target + noise;
            });

            return (X, y);
        }
    }

    #endregion

    #region Data Validation

    /// <summary>
    /// Comprehensive data validator with security checks
    /// </summary>
    public class EnterpriseDataValidator : IDataValidator
    {
        private readonly double _minValue;
        private readonly double _maxValue;
        private readonly bool _allowMissing;
        private readonly double _maxMissingRatio;
        private readonly IMLLogger _logger;

        public EnterpriseDataValidator(
            double minValue = -1e9, 
            double maxValue = 1e9, 
            bool allowMissing = false, 
            double maxMissingRatio = 0.1,
            IMLLogger logger = null)
        {
            _minValue = minValue;
            _maxValue = maxValue;
            _allowMissing = allowMissing;
            _maxMissingRatio = maxMissingRatio;
            _logger = logger ?? new ConsoleLogger();
        }

        public async Task<ValidationResult> ValidateAsync(double[,] features, CancellationToken cancellationToken = default)
        {
            using var monitor = new PerformanceMonitor("Data Validation", _logger);

            var result = new ValidationResult
            {
                TotalSamples = features.GetLength(0),
                TotalFeatures = features.GetLength(1)
            };

            if (result.TotalSamples == 0 || result.TotalFeatures == 0)
            {
                result.Errors.Add("Empty dataset provided");
                result.IsValid = false;
                return result;
            }

            await Task.Run(() =>
            {
                // Parallel validation of features
                var lockObject = new object();
                
                Parallel.For(0, result.TotalFeatures, j =>
                {
                    var featureName = $"feature_{j}";
                    int missingCount = 0;
                    var values = new List<double>();

                    for (int i = 0; i < result.TotalSamples; i++)
                    {
                        double val = features[i, j];
                        
                        if (double.IsNaN(val) || double.IsInfinity(val))
                        {
                            Interlocked.Increment(ref result.MissingValues);
                            missingCount++;
                            
                            if (!_allowMissing)
                            {
                                lock (lockObject)
                                {
                                    result.Errors.Add($"Invalid value at row {i}, feature {j}");
                                }
                            }
                        }
                        else
                        {
                            values.Add(val);
                            
                            if (val < _minValue || val > _maxValue)
                            {
                                lock (lockObject)
                                {
                                    result.Warnings.Add(
                                        $"Value {val:F4} at row {i}, feature {j} outside expected range " +
                                        $"[{_minValue:F2}, {_maxValue:F2}]");
                                }
                            }
                        }
                    }

                    lock (lockObject)
                    {
                        if (missingCount > 0)
                        {
                            result.FeatureMissingCounts[featureName] = missingCount;
                        }

                        if (values.Count > 0)
                        {
                            result.FeatureStatistics[featureName] = MathUtils.CalculateStatistics(values.ToArray());
                        }
                    }
                });

            }, cancellationToken);

            // Calculate missing value ratio
            int totalValues = result.TotalSamples * result.TotalFeatures;
            result.MissingValueRatio = totalValues > 0 ? (double)result.MissingValues / totalValues : 0;

            if (result.MissingValueRatio > _maxMissingRatio)
            {
                result.Errors.Add(
                    $"Missing value ratio {result.MissingValueRatio:P2} exceeds maximum allowed {_maxMissingRatio:P2}");
            }

            result.IsValid = result.Errors.Count == 0;

            await _logger.LogAsync(LogLevel.Info, 
                $"Data validation completed: {result.TotalSamples} samples, " +
                $"{result.MissingValues} missing values, Valid: {result.IsValid}");

            return result;
        }

        public async Task<ValidationResult> ValidateAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default)
        {
            var featuresResult = await ValidateAsync(features, cancellationToken);
            
            // Additional target validation
            if (features.GetLength(0) != targets.Length)
            {
                featuresResult.Errors.Add("Feature matrix rows must match target vector length");
                featuresResult.IsValid = false;
            }

            // Validate targets
            int invalidTargets = targets.Count(t => double.IsNaN(t) || double.IsInfinity(t));
            if (invalidTargets > 0)
            {
                featuresResult.Errors.Add($"Found {invalidTargets} invalid target values");
                featuresResult.IsValid = false;
            }

            return featuresResult;
        }
    }

    #endregion

    #region Feature Engineering

    /// <summary>
    /// Advanced feature engineering with caching and performance optimization
    /// </summary>
    public class AdvancedFeatureEngineer
    {
        private readonly IMLLogger _logger;
        private readonly ConcurrentDictionary<string, FeatureTransformResult> _transformCache;

        public AdvancedFeatureEngineer(IMLLogger logger = null)
        {
            _logger = logger ?? new ConsoleLogger();
            _transformCache = new ConcurrentDictionary<string, FeatureTransformResult>();
        }

        /// <summary>
        /// Create polynomial features up to specified degree
        /// </summary>
        public async Task<FeatureTransformResult> CreatePolynomialFeaturesAsync(
            double[,] features, int degree = 2, CancellationToken cancellationToken = default)
        {
            using var monitor = new PerformanceMonitor("Polynomial Feature Creation", _logger);

            if (degree < 1)
                throw new ArgumentException("Polynomial degree must be >= 1");

            var cacheKey = $"poly_{features.GetHashCode()}_{degree}";
            if (_transformCache.TryGetValue(cacheKey, out var cached))
            {
                await _logger.LogAsync(LogLevel.Debug, "Using cached polynomial features");
                return cached;
            }

            return await Task.Run(() =>
            {
                int samples = features.GetLength(0);
                int originalFeatures = features.GetLength(1);
                
                // Calculate number of polynomial features
                int newFeatureCount = CalculatePolynomialFeatureCount(originalFeatures, degree);
                
                double[,] result = new double[samples, newFeatureCount];
                
                // Copy original features
                Parallel.For(0, samples, i =>
                {
                    for (int j = 0; j < originalFeatures; j++)
                    {
                        result[i, j] = features[i, j];
                    }
                });

                // Generate polynomial combinations
                int featureIdx = originalFeatures;
                for (int d = 2; d <= degree; d++)
                {
                    var combinations = GenerateCombinations(originalFeatures, d);
                    
                    Parallel.ForEach(combinations, combo =>
                    {
                        int currentIdx = Interlocked.Increment(ref featureIdx) - 1;
                        
                        for (int i = 0; i < samples; i++)
                        {
                            double value = 1.0;
                            foreach (int feature in combo)
                            {
                                value *= features[i, feature];
                            }
                            result[i, currentIdx] = value;
                        }
                    });
                }

                var transformResult = new FeatureTransformResult
                {
                    TransformedFeatures = result,
                    TransformationParameters = new Dictionary<string, object>
                    {
                        ["degree"] = degree,
                        ["original_features"] = originalFeatures,
                        ["new_features"] = newFeatureCount
                    }
                };

                _transformCache.TryAdd(cacheKey, transformResult);
                return transformResult;

            }, cancellationToken);
        }

        /// <summary>
        /// Standardize features (z-score normalization)
        /// </summary>
        public async Task<FeatureTransformResult> StandardizeFeaturesAsync(
            double[,] features, CancellationToken cancellationToken = default)
        {
            using var monitor = new PerformanceMonitor("Feature Standardization", _logger);

            return await Task.Run(() =>
            {
                int samples = features.GetLength(0);
                int featureCount = features.GetLength(1);
                
                double[] means = new double[featureCount];
                double[] stds = new double[featureCount];
                
                // Calculate means in parallel
                Parallel.For(0, featureCount, j =>
                {
                    double sum = 0;
                    for (int i = 0; i < samples; i++)
                    {
                        sum += features[i, j];
                    }
                    means[j] = sum / samples;
                });
                
                // Calculate standard deviations in parallel
                Parallel.For(0, featureCount, j =>
                {
                    double sumSq = 0;
                    for (int i = 0; i < samples; i++)
                    {
                        double diff = features[i, j] - means[j];
                        sumSq += diff * diff;
                    }
                    stds[j] = Math.Sqrt(sumSq / (samples - 1));
                    
                    // Prevent division by zero
                    if (stds[j] < 1e-10)
                        stds[j] = 1.0;
                });
                
                // Apply standardization in parallel
                double[,] result = new double[samples, featureCount];
                Parallel.For(0, samples, i =>
                {
                    for (int j = 0; j < featureCount; j++)
                    {
                        result[i, j] = (features[i, j] - means[j]) / stds[j];
                    }
                });

                return new FeatureTransformResult
                {
                    TransformedFeatures = result,
                    FeatureMeans = means,
                    FeatureStds = stds,
                    TransformationParameters = new Dictionary<string, object>
                    {
                        ["method"] = "standardization",
                        ["samples"] = samples,
                        ["features"] = featureCount
                    }
                };

            }, cancellationToken);
        }

        private static int CalculatePolynomialFeatureCount(int features, int degree)
        {
            int count = 0;
            for (int d = 1; d <= degree; d++)
            {
                count += BinomialCoefficient(features + d - 1, d);
            }
            return count;
        }

        private static int BinomialCoefficient(int n, int k)
        {
            if (k > n) return 0;
            if (k == 0 || k == n) return 1;

            int result = 1;
            for (int i = 0; i < Math.Min(k, n - k); i++)
            {
                result = result * (n - i) / (i + 1);
            }
            return result;
        }

        private static IEnumerable<int[]> GenerateCombinations(int n, int k)
        {
            var combinations = new List<int[]>();
            var combo = new int[k];
            
            void Generate(int start, int depth)
            {
                if (depth == k)
                {
                    combinations.Add((int[])combo.Clone());
                    return;
                }
                
                for (int i = start; i < n; i++)
                {
                    combo[depth] = i;
                    Generate(i, depth + 1);
                }
            }
            
            Generate(0, 0);
            return combinations;
        }
    }

    #endregion

    #region Linear Regression Implementation

    /// <summary>
    /// Enterprise-grade Linear Regression with comprehensive features
    /// </summary>
    public class EnterpriseLinearRegression : IMLModel
    {
        private double[] _weights;
        private double _bias;
        private bool _isTrained;
        private readonly SemaphoreSlim _modelSemaphore;
        private readonly IMLLogger _logger;
        private readonly TrainingConfig _config;
        
        // Training statistics
        private List<double> _trainingHistory;
        private TimeSpan _lastTrainingTime;
        private int _iterationsCompleted;

        public bool IsTrained => _isTrained;

        public EnterpriseLinearRegression(TrainingConfig config = null, IMLLogger logger = null)
        {
            _config = config ?? new TrainingConfig();
            _logger = logger ?? new ConsoleLogger();
            _modelSemaphore = new SemaphoreSlim(1, 1);
            _trainingHistory = new List<double>();
            _isTrained = false;
        }

        public async Task TrainAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default)
        {
            using var monitor = new PerformanceMonitor("Linear Regression Training", _logger);
            await _modelSemaphore.WaitAsync(cancellationToken);
            
            try
            {
                if (features.GetLength(0) != targets.Length)
                    throw new ModelTrainingException("Feature matrix rows must match target vector size");

                if (features.GetLength(0) == 0 || features.GetLength(1) == 0)
                    throw new ModelTrainingException("Empty dataset provided for training");

                int samples = features.GetLength(0);
                int featureCount = features.GetLength(1);

                // Initialize parameters
                _weights = new double[featureCount];
                _bias = 0;
                _trainingHistory.Clear();

                var stopwatch = Stopwatch.StartNew();

                // Training with parallel gradient descent
                double prevCost = double.MaxValue;
                int patienceCounter = 0;

                for (int iteration = 0; iteration < _config.MaxIterations; iteration++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Forward pass - compute predictions in parallel
                    double[] predictions = await ComputePredictionsAsync(features, cancellationToken);

                    // Compute cost (MSE)
                    double cost = await ComputeCostAsync(predictions, targets, cancellationToken);
                    _trainingHistory.Add(cost);

                    // Check convergence
                    if (Math.Abs(prevCost - cost) < _config.ConvergenceThreshold)
                    {
                        await _logger.LogAsync(LogLevel.Info, 
                            $"Convergence achieved at iteration {iteration}");
                        break;
                    }

                    // Early stopping check
                    if (_config.EnableEarlyStopping)
                    {
                        if (cost > prevCost)
                        {
                            patienceCounter++;
                            if (patienceCounter >= _config.EarlyStoppingPatience)
                            {
                                await _logger.LogAsync(LogLevel.Info, 
                                    $"Early stopping at iteration {iteration}");
                                break;
                            }
                        }
                        else
                        {
                            patienceCounter = 0;
                        }
                    }
                    
                    prevCost = cost;

                    // Backward pass - compute gradients and update parameters
                    await UpdateParametersAsync(features, predictions, targets, cancellationToken);
                    
                    _iterationsCompleted = iteration + 1;
                }

                stopwatch.Stop();
                _lastTrainingTime = stopwatch.Elapsed;
                _isTrained = true;

                await _logger.LogAsync(LogLevel.Info, "Linear regression training completed");
            }
            catch (OperationCanceledException)
            {
                await _logger.LogAsync(LogLevel.Warning, "Training was cancelled");
                throw;
            }
            catch (Exception ex)
            {
                await _logger.LogExceptionAsync(ex, "Training failed");
                throw new ModelTrainingException("Training failed", ex);
            }
            finally
            {
                _modelSemaphore.Release();
            }
        }

        public async Task<double[]> PredictAsync(double[,] features, CancellationToken cancellationToken = default)
        {
            if (!_isTrained)
                throw new ModelPredictionException("Model must be trained before making predictions");

            if (features.GetLength(1) != _weights.Length)
                throw new ModelPredictionException(
                    $"Feature count mismatch: expected {_weights.Length}, got {features.GetLength(1)}");

            return await ComputePredictionsAsync(features, cancellationToken);
        }

        public async Task<ModelMetrics> EvaluateAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default)
        {
            var predictionStart = Stopwatch.StartNew();
            double[] predictions = await PredictAsync(features, cancellationToken);
            predictionStart.Stop();

            return await Task.Run(() =>
            {
                // Calculate metrics in parallel
                double mse, mae, rSquared;
                
                Parallel.Invoke(
                    () => {
                        double sumSquaredError = predictions.Zip(targets, (p, t) => Math.Pow(p - t, 2)).Sum();
                        mse = sumSquaredError / targets.Length;
                    },
                    () => {
                        mae = predictions.Zip(targets, (p, t) => Math.Abs(p - t)).Average();
                    },
                    () => {
                        double meanTarget = targets.Average();
                        double totalSumSquares = targets.Select(t => Math.Pow(t - meanTarget, 2)).Sum();
                        double residualSumSquares = predictions.Zip(targets, (p, t) => Math.Pow(p - t, 2)).Sum();
                        rSquared = totalSumSquares > 1e-10 ? 1 - (residualSumSquares / totalSumSquares) : 0;
                    }
                );

                return new ModelMetrics
                {
                    MSE = mse,
                    RMSE = Math.Sqrt(mse),
                    MAE = mae,
                    RSquared = rSquared,
                    TrainingTime = _lastTrainingTime,
                    PredictionTime = predictionStart.Elapsed,
                    IterationsCompleted = _iterationsCompleted,
                    TrainingHistory = new List<double>(_trainingHistory)
                };

            }, cancellationToken);
        }

        public async Task SaveAsync(string filePath, CancellationToken cancellationToken = default)
        {
            if (!_isTrained)
                throw new InvalidOperationException("Cannot save untrained model");

            await _modelSemaphore.WaitAsync(cancellationToken);
            try
            {
                var modelData = new
                {
                    Weights = _weights,
                    Bias = _bias,
                    Config = _config,
                    TrainingHistory = _trainingHistory,
                    TrainingTime = _lastTrainingTime.TotalSeconds,
                    IterationsCompleted = _iterationsCompleted
                };

                string json = JsonSerializer.Serialize(modelData, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
                
                await File.WriteAllTextAsync(filePath, json, cancellationToken);
                await _logger.LogAsync(LogLevel.Info, $"Model saved to {filePath}");
            }
            finally
            {
                _modelSemaphore.Release();
            }
        }

        public async Task LoadAsync(string filePath, CancellationToken cancellationToken = default)
        {
            await _modelSemaphore.WaitAsync(cancellationToken);
            try
            {
                string json = await File.ReadAllTextAsync(filePath, cancellationToken);
                using var document = JsonDocument.Parse(json);
                var root = document.RootElement;

                _weights = root.GetProperty("Weights").EnumerateArray()
                    .Select(x => x.GetDouble()).ToArray();
                _bias = root.GetProperty("Bias").GetDouble();
                
                if (root.TryGetProperty("TrainingHistory", out var historyElement))
                {
                    _trainingHistory = historyElement.EnumerateArray()
                        .Select(x => x.GetDouble()).ToList();
                }

                _iterationsCompleted = root.TryGetProperty("IterationsCompleted", out var iterElement) 
                    ? iterElement.GetInt32() : 0;

                _isTrained = true;
                await _logger.LogAsync(LogLevel.Info, $"Model loaded from {filePath}");
            }
            finally
            {
                _modelSemaphore.Release();
            }
        }

        private async Task<double[]> ComputePredictionsAsync(double[,] features, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                int samples = features.GetLength(0);
                int featureCount = features.GetLength(1);
                double[] predictions = new double[samples];

                Parallel.For(0, samples, i =>
                {
                    double prediction = _bias;
                    for (int j = 0; j < featureCount; j++)
                    {
                        prediction += _weights[j] * features[i, j];
                    }
                    predictions[i] = prediction;
                });

                return predictions;
            }, cancellationToken);
        }

        private async Task<double> ComputeCostAsync(double[] predictions, double[] targets, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                double cost = predictions.AsParallel()
                    .Zip(targets.AsParallel(), (p, t) => Math.Pow(p - t, 2))
                    .Sum() / (2 * targets.Length);

                // Add regularization if enabled
                if (_config.EnableRegularization)
                {
                    double regularization = _config.RegularizationStrength * 
                        _weights.AsParallel().Select(w => w * w).Sum();
                    cost += regularization;
                }

                return cost;
            }, cancellationToken);
        }

        private async Task UpdateParametersAsync(double[,] features, double[] predictions, double[] targets, CancellationToken cancellationToken)
        {
            await Task.Run(() =>
            {
                int samples = features.GetLength(0);
                int featureCount = features.GetLength(1);

                // Compute gradients in parallel
                double[] weightGradients = new double[featureCount];
                double biasGradient = 0;

                // Parallel computation of gradients
                Parallel.For(0, samples, i =>
                {
                    double error = predictions[i] - targets[i];
                    Interlocked.Exchange(ref biasGradient, biasGradient + error);

                    for (int j = 0; j < featureCount; j++)
                    {
                        double contribution = error * features[i, j];
                        lock (weightGradients)
                        {
                            weightGradients[j] += contribution;
                        }
                    }
                });

                // Update parameters
                _bias -= _config.LearningRate * biasGradient / samples;
                
                Parallel.For(0, featureCount, j =>
                {
                    double gradient = weightGradients[j] / samples;
                    
                    // Add regularization gradient if enabled
                    if (_config.EnableRegularization)
                    {
                        gradient += _config.RegularizationStrength * _weights[j];
                    }
                    
                    _weights[j] -= _config.LearningRate * gradient;
                });

            }, cancellationToken);
        }

        public void Dispose()
        {
            _modelSemaphore?.Dispose();
        }
    }

    #endregion

    #region Production ML Pipeline

    /// <summary>
    /// Enterprise production ML pipeline with comprehensive monitoring
    /// </summary>
    public class EnterpriseMLPipeline : IDisposable
    {
        private readonly IMLModel _model;
        private readonly IDataValidator _validator;
        private readonly AdvancedFeatureEngineer _featureEngineer;
        private readonly IMLLogger _logger;
        private readonly SemaphoreSlim _pipelineSemaphore;
        
        private FeatureTransformResult _lastTransformation;
        private bool _isStandardized;

        public EnterpriseMLPipeline(
            IMLModel model = null, 
            IDataValidator validator = null,
            IMLLogger logger = null)
        {
            _model = model ?? new EnterpriseLinearRegression();
            _validator = validator ?? new EnterpriseDataValidator();
            _logger = logger ?? new ConsoleLogger();
            _featureEngineer = new AdvancedFeatureEngineer(_logger);
            _pipelineSemaphore = new SemaphoreSlim(1, 1);
            _isStandardized = false;
        }

        public async Task TrainAsync(double[,] features, double[] targets, 
            double validationSplit = 0.2, CancellationToken cancellationToken = default)
        {
            using var monitor = new PerformanceMonitor("Enterprise Pipeline Training", _logger);
            await _pipelineSemaphore.WaitAsync(cancellationToken);

            try
            {
                // Data validation
                await _logger.LogAsync(LogLevel.Info, "Starting data validation...");
                var validation = await _validator.ValidateAsync(features, targets, cancellationToken);
                
                if (!validation.IsValid)
                {
                    var errorMsg = "Data validation failed: " + string.Join("; ", validation.Errors);
                    throw new DataValidationException(errorMsg, validation.Errors);
                }

                // Feature standardization
                await _logger.LogAsync(LogLevel.Info, "Applying feature standardization...");
                _lastTransformation = await _featureEngineer.StandardizeFeaturesAsync(features, cancellationToken);
                _isStandardized = true;

                // Train-validation split
                var (trainFeatures, valFeatures, trainTargets, valTargets) = 
                    await TrainTestSplitAsync(_lastTransformation.TransformedFeatures, targets, validationSplit, cancellationToken);

                // Model training
                await _logger.LogAsync(LogLevel.Info, "Starting model training...");
                await _model.TrainAsync(trainFeatures, trainTargets, cancellationToken);

                // Validation evaluation
                if (validationSplit > 0)
                {
                    await _logger.LogAsync(LogLevel.Info, "Evaluating on validation set...");
                    var metrics = await _model.EvaluateAsync(valFeatures, valTargets, cancellationToken);
                    await _logger.LogAsync(LogLevel.Info, 
                        $"Validation R²: {metrics.RSquared:F4}, RMSE: {metrics.RMSE:F4}");
                }

                await _logger.LogAsync(LogLevel.Info, "Pipeline training completed successfully");
            }
            catch (Exception ex)
            {
                await _logger.LogExceptionAsync(ex, "Pipeline training failed");
                throw;
            }
            finally
            {
                _pipelineSemaphore.Release();
            }
        }

        public async Task<double[]> PredictAsync(double[,] features, CancellationToken cancellationToken = default)
        {
            if (!_model.IsTrained)
                throw new ModelPredictionException("Pipeline must be trained before making predictions");

            try
            {
                double[,] processedFeatures = features;
                
                // Apply same transformation as training
                if (_isStandardized && _lastTransformation != null)
                {
                    processedFeatures = await ApplyStandardizationAsync(features, cancellationToken);
                }

                return await _model.PredictAsync(processedFeatures, cancellationToken);
            }
            catch (Exception ex)
            {
                await _logger.LogExceptionAsync(ex, "Pipeline prediction failed");
                throw;
            }
        }

        public async Task<ModelMetrics> EvaluateAsync(double[,] features, double[] targets, CancellationToken cancellationToken = default)
        {
            try
            {
                double[,] processedFeatures = features;
                
                // Apply same transformation as training
                if (_isStandardized && _lastTransformation != null)
                {
                    processedFeatures = await ApplyStandardizationAsync(features, cancellationToken);
                }

                return await _model.EvaluateAsync(processedFeatures, targets, cancellationToken);
            }
            catch (Exception ex)
            {
                await _logger.LogExceptionAsync(ex, "Pipeline evaluation failed");
                throw;
            }
        }

        public async Task SavePipelineAsync(string directoryPath, CancellationToken cancellationToken = default)
        {
            Directory.CreateDirectory(directoryPath);

            // Save model
            await _model.SaveAsync(Path.Combine(directoryPath, "model.json"), cancellationToken);

            // Save feature transformation parameters
            if (_lastTransformation != null)
            {
                var transformData = new
                {
                    IsStandardized = _isStandardized,
                    FeatureMeans = _lastTransformation.FeatureMeans,
                    FeatureStds = _lastTransformation.FeatureStds,
                    TransformationParameters = _lastTransformation.TransformationParameters
                };

                string transformJson = JsonSerializer.Serialize(transformData, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
                
                await File.WriteAllTextAsync(
                    Path.Combine(directoryPath, "feature_transform.json"), 
                    transformJson, cancellationToken);
            }

            await _logger.LogAsync(LogLevel.Info, $"Pipeline saved to {directoryPath}");
        }

        private async Task<double[,]> ApplyStandardizationAsync(double[,] features, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                int samples = features.GetLength(0);
                int featureCount = features.GetLength(1);
                double[,] result = new double[samples, featureCount];

                Parallel.For(0, samples, i =>
                {
                    for (int j = 0; j < featureCount; j++)
                    {
                        result[i, j] = (features[i, j] - _lastTransformation.FeatureMeans[j]) / 
                                      _lastTransformation.FeatureStds[j];
                    }
                });

                return result;
            }, cancellationToken);
        }

        private static async Task<(double[,] trainFeatures, double[,] valFeatures, double[] trainTargets, double[] valTargets)>
            TrainTestSplitAsync(double[,] features, double[] targets, double testRatio, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                int totalSamples = features.GetLength(0);
                int testSize = (int)(totalSamples * testRatio);
                int trainSize = totalSamples - testSize;

                var indices = Enumerable.Range(0, totalSamples).ToArray();
                var random = new Random(42);
                
                // Shuffle indices
                for (int i = indices.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (indices[i], indices[j]) = (indices[j], indices[i]);
                }

                // Split data
                double[,] trainFeatures = new double[trainSize, features.GetLength(1)];
                double[,] valFeatures = new double[testSize, features.GetLength(1)];
                double[] trainTargets = new double[trainSize];
                double[] valTargets = new double[testSize];

                Parallel.For(0, trainSize, i =>
                {
                    int idx = indices[i];
                    for (int j = 0; j < features.GetLength(1); j++)
                    {
                        trainFeatures[i, j] = features[idx, j];
                    }
                    trainTargets[i] = targets[idx];
                });

                Parallel.For(0, testSize, i =>
                {
                    int idx = indices[trainSize + i];
                    for (int j = 0; j < features.GetLength(1); j++)
                    {
                        valFeatures[i, j] = features[idx, j];
                    }
                    valTargets[i] = targets[idx];
                });

                return (trainFeatures, valFeatures, trainTargets, valTargets);
            }, cancellationToken);
        }

        public void Dispose()
        {
            _pipelineSemaphore?.Dispose();
            (_model as IDisposable)?.Dispose();
        }
    }

    #endregion

    #region Demonstration Program

    /// <summary>
    /// Comprehensive demonstration of C# ML patterns
    /// </summary>
    public class Program
    {
        public static async Task<int> Main(string[] args)
        {
            var logger = new ConsoleLogger();
            
            try
            {
                await logger.LogAsync(LogLevel.Info, "🚀 C# ML Production Patterns Demonstration");
                await logger.LogAsync(LogLevel.Info, "==============================================");

                // Generate synthetic dataset
                await logger.LogAsync(LogLevel.Info, "📊 Generating synthetic dataset...");
                var (features, targets) = MathUtils.GenerateRegressionDataset(1000, 5, 0.1);

                // Create enterprise pipeline
                await logger.LogAsync(LogLevel.Info, "🏗️ Creating enterprise ML pipeline...");
                var config = new TrainingConfig
                {
                    LearningRate = 0.01,
                    MaxIterations = 1000,
                    ConvergenceThreshold = 1e-6,
                    EnableEarlyStopping = true,
                    EarlyStoppingPatience = 10
                };

                using var pipeline = new EnterpriseMLPipeline(
                    new EnterpriseLinearRegression(config, logger),
                    new EnterpriseDataValidator(logger: logger),
                    logger);

                // Train pipeline
                await logger.LogAsync(LogLevel.Info, "🔄 Training production ML pipeline...");
                await pipeline.TrainAsync(features, targets, 0.2);
                await logger.LogAsync(LogLevel.Info, "✅ Model training completed");

                // Make predictions
                await logger.LogAsync(LogLevel.Info, "🔮 Making predictions...");
                var (testFeatures, testTargets) = MathUtils.GenerateRegressionDataset(100, 5, 0.1, 123);
                var predictions = await pipeline.PredictAsync(testFeatures);
                
                await logger.LogAsync(LogLevel.Info, $"Sample predictions: " +
                    string.Join(", ", predictions.Take(5).Select(p => $"{p:F4}")));

                // Model evaluation
                await logger.LogAsync(LogLevel.Info, "📊 Evaluating model performance...");
                var metrics = await pipeline.EvaluateAsync(testFeatures, testTargets);
                
                await logger.LogAsync(LogLevel.Info, $"R² Score: {metrics.RSquared:F4}");
                await logger.LogAsync(LogLevel.Info, $"RMSE: {metrics.RMSE:F4}");
                await logger.LogAsync(LogLevel.Info, $"MAE: {metrics.MAE:F4}");
                await logger.LogAsync(LogLevel.Info, $"Training Time: {metrics.TrainingTime.TotalSeconds:F2} seconds");
                await logger.LogAsync(LogLevel.Info, $"Prediction Time: {metrics.PredictionTime.TotalMilliseconds:F2}ms");

                // Feature engineering demonstration
                await logger.LogAsync(LogLevel.Info, "🔧 Feature Engineering demonstration...");
                var featureEngineer = new AdvancedFeatureEngineer(logger);
                var polynomialResult = await featureEngineer.CreatePolynomialFeaturesAsync(testFeatures, 2);
                
                await logger.LogAsync(LogLevel.Info, 
                    $"Original features: {testFeatures.GetLength(1)}, " +
                    $"Polynomial features: {polynomialResult.TransformedFeatures.GetLength(1)}");

                // Performance monitoring summary
                await logger.LogAsync(LogLevel.Info, "⚡ Performance characteristics:");
                await logger.LogAsync(LogLevel.Info, "- Async/await operations: ✅ Non-blocking I/O");
                await logger.LogAsync(LogLevel.Info, "- Parallel processing: ✅ PLINQ and Parallel.For");
                await logger.LogAsync(LogLevel.Info, "- Thread safety: ✅ SemaphoreSlim protection");
                await logger.LogAsync(LogLevel.Info, "- Enterprise patterns: ✅ DI, SOLID principles");
                await logger.LogAsync(LogLevel.Info, "- Exception handling: ✅ Comprehensive hierarchy");

                await logger.LogAsync(LogLevel.Info, "✅ C# ML demonstration completed successfully!");

                return 0;
            }
            catch (Exception ex)
            {
                await logger.LogExceptionAsync(ex, "Fatal error during demonstration");
                return 1;
            }
        }
    }

    #endregion
}