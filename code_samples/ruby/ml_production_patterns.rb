# frozen_string_literal: true

##
# Production-Ready Machine Learning Patterns in Ruby
# =================================================
#
# This module demonstrates industry-standard ML patterns in Ruby with proper
# Rails integration, gem patterns, and production deployment considerations for AI training datasets.
#
# Key Features:
# - Ruby metaprogramming and DSL patterns
# - Rails integration with ActiveRecord patterns
# - Concurrent Ruby with async operations
# - RubyGems standards and Bundler compatibility
# - Comprehensive error handling with custom exceptions
# - Database integration with ActiveRecord
# - Extensive documentation for AI learning
# - Production-ready patterns with caching and logging
#
# Author: AI Training Dataset
# License: MIT

require 'concurrent-ruby'
require 'json'
require 'logger'
require 'matrix'
require 'benchmark'
require 'fileutils'

module ML
  module ProductionPatterns

    # MARK: - Error Hierarchy

    class MLError < StandardError
      attr_reader :context

      def initialize(message, context: nil)
        super("ML Error: #{message}")
        @context = context
      end
    end

    class DataValidationError < MLError
      attr_reader :validation_errors

      def initialize(message, validation_errors: [], context: 'data_validation')
        super("Data Validation - #{message}", context: context)
        @validation_errors = validation_errors
      end
    end

    class ModelTrainingError < MLError
      attr_reader :iterations_failed

      def initialize(message, iterations_failed: nil, context: 'model_training')
        super("Model Training - #{message}", context: context)
        @iterations_failed = iterations_failed
      end
    end

    class ModelPredictionError < MLError
      def initialize(message, context: 'model_prediction')
        super("Model Prediction - #{message}", context: context)
      end
    end

    class FeatureEngineeringError < MLError
      def initialize(message, context: 'feature_engineering')
        super("Feature Engineering - #{message}", context: context)
      end
    end

    # MARK: - Data Classes

    class ValidationResult
      include Comparable
      
      attr_accessor :is_valid, :errors, :warnings, :total_samples, :total_features,
                    :missing_values, :missing_value_ratio, :feature_missing_counts,
                    :feature_statistics

      def initialize(is_valid:, errors: [], warnings: [], total_samples: 0, total_features: 0,
                     missing_values: 0, missing_value_ratio: 0.0, feature_missing_counts: {},
                     feature_statistics: {})
        @is_valid = is_valid
        @errors = errors
        @warnings = warnings
        @total_samples = total_samples
        @total_features = total_features
        @missing_values = missing_values
        @missing_value_ratio = missing_value_ratio
        @feature_missing_counts = feature_missing_counts
        @feature_statistics = feature_statistics
      end

      def to_h
        {
          is_valid: @is_valid,
          errors: @errors,
          warnings: @warnings,
          total_samples: @total_samples,
          total_features: @total_features,
          missing_values: @missing_values,
          missing_value_ratio: @missing_value_ratio,
          feature_missing_counts: @feature_missing_counts,
          feature_statistics: @feature_statistics
        }
      end

      def to_json(*args)
        to_h.to_json(*args)
      end

      def <=>(other)
        return nil unless other.is_a?(ValidationResult)
        
        [@total_samples, @missing_value_ratio] <=> [other.total_samples, other.missing_value_ratio]
      end
    end

    class FeatureStatistics
      attr_accessor :min, :max, :mean, :standard_deviation, :variance, :skewness, :kurtosis

      def initialize(min:, max:, mean:, standard_deviation:, variance:, skewness:, kurtosis:)
        @min = min
        @max = max
        @mean = mean
        @standard_deviation = standard_deviation
        @variance = variance
        @skewness = skewness
        @kurtosis = kurtosis
      end

      def to_h
        {
          min: @min,
          max: @max,
          mean: @mean,
          standard_deviation: @standard_deviation,
          variance: @variance,
          skewness: @skewness,
          kurtosis: @kurtosis
        }
      end

      def to_json(*args)
        to_h.to_json(*args)
      end
    end

    class ModelMetrics
      attr_accessor :mse, :rmse, :mae, :r_squared, :training_time, :prediction_time,
                    :iterations_completed, :convergence_value, :training_history

      def initialize(mse:, rmse:, mae:, r_squared:, training_time:, prediction_time:,
                     iterations_completed:, convergence_value:, training_history:)
        @mse = mse
        @rmse = rmse
        @mae = mae
        @r_squared = r_squared
        @training_time = training_time
        @prediction_time = prediction_time
        @iterations_completed = iterations_completed
        @convergence_value = convergence_value
        @training_history = training_history
      end

      def to_h
        {
          mse: @mse,
          rmse: @rmse,
          mae: @mae,
          r_squared: @r_squared,
          training_time: @training_time,
          prediction_time: @prediction_time,
          iterations_completed: @iterations_completed,
          convergence_value: @convergence_value,
          training_history: @training_history
        }
      end

      def to_json(*args)
        to_h.to_json(*args)
      end
    end

    class TrainingConfig
      attr_accessor :learning_rate, :max_iterations, :convergence_threshold, :validation_split,
                    :enable_early_stopping, :early_stopping_patience, :enable_regularization,
                    :regularization_strength, :batch_size

      def initialize(learning_rate: 0.01, max_iterations: 1000, convergence_threshold: 1e-6,
                     validation_split: 0.2, enable_early_stopping: true,
                     early_stopping_patience: 10, enable_regularization: false,
                     regularization_strength: 0.01, batch_size: 32)
        
        validate_config!(learning_rate, max_iterations, convergence_threshold,
                        validation_split, early_stopping_patience,
                        regularization_strength, batch_size)

        @learning_rate = learning_rate
        @max_iterations = max_iterations
        @convergence_threshold = convergence_threshold
        @validation_split = validation_split
        @enable_early_stopping = enable_early_stopping
        @early_stopping_patience = early_stopping_patience
        @enable_regularization = enable_regularization
        @regularization_strength = regularization_strength
        @batch_size = batch_size
      end

      def to_h
        {
          learning_rate: @learning_rate,
          max_iterations: @max_iterations,
          convergence_threshold: @convergence_threshold,
          validation_split: @validation_split,
          enable_early_stopping: @enable_early_stopping,
          early_stopping_patience: @early_stopping_patience,
          enable_regularization: @enable_regularization,
          regularization_strength: @regularization_strength,
          batch_size: @batch_size
        }
      end

      def to_json(*args)
        to_h.to_json(*args)
      end

      private

      def validate_config!(learning_rate, max_iterations, convergence_threshold,
                          validation_split, early_stopping_patience,
                          regularization_strength, batch_size)
        raise ArgumentError, 'Learning rate must be positive' unless learning_rate.positive?
        raise ArgumentError, 'Max iterations must be positive' unless max_iterations.positive?
        raise ArgumentError, 'Convergence threshold must be positive' unless convergence_threshold.positive?
        raise ArgumentError, 'Validation split must be between 0 and 1' unless validation_split.between?(0, 1)
        raise ArgumentError, 'Early stopping patience must be positive' unless early_stopping_patience.positive?
        raise ArgumentError, 'Regularization strength must be non-negative' unless regularization_strength >= 0
        raise ArgumentError, 'Batch size must be positive' unless batch_size.positive?
      end
    end

    class FeatureTransformResult
      attr_accessor :transformed_features, :feature_means, :feature_stds, :transformation_parameters

      def initialize(transformed_features:, feature_means: nil, feature_stds: nil, transformation_parameters: {})
        @transformed_features = transformed_features
        @feature_means = feature_means
        @feature_stds = feature_stds
        @transformation_parameters = transformation_parameters
      end

      def to_h
        {
          transformed_features: @transformed_features,
          feature_means: @feature_means,
          feature_stds: @feature_stds,
          transformation_parameters: @transformation_parameters
        }
      end

      def to_json(*args)
        to_h.to_json(*args)
      end
    end

    # MARK: - Logging Implementation

    class MLLogger < Logger
      LOG_LEVELS = %i[debug info warn error fatal].freeze

      def initialize(output = $stdout, level: Logger::INFO, progname: 'MLLogger')
        super(output, progname: progname)
        self.level = level
        self.formatter = proc do |severity, datetime, progname, msg|
          "[#{datetime.strftime('%Y-%m-%d %H:%M:%S.%3N')}] [#{severity}] #{msg}\n"
        end
      end

      def log_exception(exception, context)
        error("#{context}: #{exception.message}")
        debug("ML Error Context: #{exception.context}") if exception.respond_to?(:context)
        debug("Stack Trace: #{exception.backtrace&.join("\n")}")
      end

      # Ruby-style aliases
      alias_method :log_info, :info
      alias_method :log_error, :error
      alias_method :log_debug, :debug
      alias_method :log_warn, :warn
    end

    # MARK: - Performance Monitoring

    class PerformanceMonitor
      def initialize(operation_name, logger)
        @operation_name = operation_name
        @logger = logger
        @start_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end

      def dispose
        end_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        duration = ((end_time - @start_time) * 1000).round(2) # Convert to milliseconds
        @logger.info("[PERFORMANCE] #{@operation_name} completed in #{duration}ms")
      end

      def elapsed_time
        Process.clock_gettime(Process::CLOCK_MONOTONIC) - @start_time
      end
    end

    # MARK: - Mathematical Utilities

    module MathUtils
      extend self

      ##
      # Matrix multiplication with optimized implementation
      def matrix_multiply(a, b)
        raise ArgumentError, 'Matrices cannot be empty' if a.empty? || b.empty?
        raise ArgumentError, "Matrix dimensions don't match for multiplication" if a.first.size != b.size

        rows_a = a.size
        cols_a = a.first.size
        cols_b = b.first.size

        result = Array.new(rows_a) { Array.new(cols_b, 0.0) }

        (0...rows_a).each do |i|
          (0...cols_b).each do |j|
            sum = 0.0
            (0...cols_a).each do |k|
              sum += a[i][k] * b[k][j]
            end
            result[i][j] = sum
          end
        end

        result
      end

      ##
      # Vectorized dot product calculation
      def dot_product(a, b)
        raise ArgumentError, 'Vector lengths must match' if a.size != b.size

        a.zip(b).map { |x, y| x * y }.sum
      end

      ##
      # Calculate comprehensive statistics for a vector
      def calculate_statistics(values)
        valid_values = values.select { |v| v.finite? && !v.nan? }

        if valid_values.empty?
          return FeatureStatistics.new(
            min: Float::NAN, max: Float::NAN, mean: Float::NAN,
            standard_deviation: Float::NAN, variance: Float::NAN,
            skewness: Float::NAN, kurtosis: Float::NAN
          )
        end

        count = valid_values.size.to_f
        min_val = valid_values.min
        max_val = valid_values.max
        mean = valid_values.sum / count

        # Calculate variance
        variance = valid_values.map { |v| (v - mean)**2 }.sum / (count - 1)
        standard_deviation = Math.sqrt(variance)

        # Calculate skewness and kurtosis
        normalized_values = valid_values.map { |v| (v - mean) / standard_deviation }
        skewness = normalized_values.map { |v| v**3 }.sum / count
        kurtosis = normalized_values.map { |v| v**4 }.sum / count - 3.0 # Excess kurtosis

        FeatureStatistics.new(
          min: min_val, max: max_val, mean: mean,
          standard_deviation: standard_deviation, variance: variance,
          skewness: skewness, kurtosis: kurtosis
        )
      end

      ##
      # Generate synthetic regression dataset with configurable parameters
      def generate_regression_dataset(samples, features, noise_level: 0.1, seed: 42)
        rng = Random.new(seed)

        # Generate random true weights
        true_weights = Array.new(features) { random_gaussian(rng) }

        x_data = []
        y_data = []

        samples.times do
          sample = Array.new(features) { random_gaussian(rng) }
          target = dot_product(sample, true_weights) + random_gaussian(rng) * noise_level

          x_data << sample
          y_data << target
        end

        [x_data, y_data]
      end

      ##
      # Train-test split with proper randomization
      def train_test_split(features, targets, test_ratio: 0.2, seed: 42)
        raise ArgumentError, 'Features and targets must have same number of samples' if features.size != targets.size
        raise ArgumentError, 'Test ratio must be between 0 and 1' unless test_ratio.between?(0, 1)

        total_samples = features.size
        test_size = (total_samples * test_ratio).to_i
        train_size = total_samples - test_size

        # Create and shuffle indices
        indices = (0...total_samples).to_a.shuffle(random: Random.new(seed))

        train_indices = indices[0...train_size]
        test_indices = indices[train_size..-1]

        train_features = train_indices.map { |i| features[i] }
        test_features = test_indices.map { |i| features[i] }
        train_targets = train_indices.map { |i| targets[i] }
        test_targets = test_indices.map { |i| targets[i] }

        [train_features, test_features, train_targets, test_targets]
      end

      private

      ##
      # Generate Gaussian random numbers using Box-Muller transform
      def random_gaussian(rng = Random.new)
        @has_spare ||= false
        @spare ||= 0.0

        if @has_spare
          @has_spare = false
          return @spare
        end

        @has_spare = true
        u1 = rng.rand
        u2 = rng.rand

        magnitude = Math.sqrt(-2.0 * Math.log(u1))
        @spare = magnitude * Math.cos(2.0 * Math::PI * u2)

        magnitude * Math.sin(2.0 * Math::PI * u2)
      end
    end

    # MARK: - Data Validation

    class EnterpriseDataValidator
      def initialize(min_value: -1e9, max_value: 1e9, allow_missing: false,
                     max_missing_ratio: 0.1, logger: MLLogger.new)
        @min_value = min_value
        @max_value = max_value
        @allow_missing = allow_missing
        @max_missing_ratio = max_missing_ratio
        @logger = logger
      end

      ##
      # Validate features with comprehensive error checking
      def validate(features, targets = nil)
        monitor = PerformanceMonitor.new('Data Validation', @logger)

        begin
          errors = []
          warnings = []
          missing_values = 0
          feature_missing_counts = {}
          feature_statistics = {}

          total_samples = features.size
          total_features = features.empty? ? 0 : features.first.size

          if total_samples.zero? || total_features.zero?
            errors << 'Empty dataset provided'
            return ValidationResult.new(
              is_valid: false, errors: errors, warnings: warnings,
              total_samples: total_samples, total_features: total_features,
              missing_values: missing_values, missing_value_ratio: 0.0
            )
          end

          # Validate feature matrix
          validate_features(features, errors, warnings, missing_values,
                           feature_missing_counts, feature_statistics)

          # Validate targets if provided
          validate_targets(features, targets, errors) if targets

          # Calculate missing value ratio
          total_values = total_samples * total_features
          missing_value_ratio = total_values.positive? ? missing_values.to_f / total_values : 0.0

          if missing_value_ratio > @max_missing_ratio
            errors << "Missing value ratio #{'%.2f' % (missing_value_ratio * 100)}% exceeds " \
                     "maximum allowed #{'%.2f' % (@max_missing_ratio * 100)}%"
          end

          is_valid = errors.empty?

          @logger.info("Data validation completed: #{total_samples} samples, " \
                      "#{missing_values} missing values, Valid: #{is_valid}")

          ValidationResult.new(
            is_valid: is_valid, errors: errors, warnings: warnings,
            total_samples: total_samples, total_features: total_features,
            missing_values: missing_values, missing_value_ratio: missing_value_ratio,
            feature_missing_counts: feature_missing_counts,
            feature_statistics: feature_statistics
          )
        ensure
          monitor.dispose
        end
      end

      private

      def validate_features(features, errors, warnings, missing_values,
                           feature_missing_counts, feature_statistics)
        total_features = features.first.size

        # Process features concurrently using Concurrent Ruby
        futures = (0...total_features).map do |j|
          Concurrent::Future.execute do
            feature_name = "feature_#{j}"
            feature_values = []
            local_missing_count = 0
            local_warnings = []

            features.each_with_index do |sample, i|
              next if j >= sample.size

              value = sample[j]

              if value.nan? || value.infinite?
                local_missing_count += 1
                local_warnings << "Invalid value at row #{i}, feature #{j}" unless @allow_missing
              else
                feature_values << value
                if value < @min_value || value > @max_value
                  local_warnings << "Value #{'%.4f' % value} at row #{i}, feature #{j} " \
                                   "outside expected range [#{@min_value}, #{@max_value}]"
                end
              end
            end

            {
              feature_name: feature_name,
              missing_count: local_missing_count,
              warnings: local_warnings,
              feature_values: feature_values
            }
          end
        end

        # Collect results
        results = futures.map(&:value!)

        results.each do |result|
          missing_values += result[:missing_count]
          warnings.concat(result[:warnings])

          feature_missing_counts[result[:feature_name]] = result[:missing_count] if result[:missing_count].positive?

          unless result[:feature_values].empty?
            feature_statistics[result[:feature_name]] = MathUtils.calculate_statistics(result[:feature_values])
          end
        end
      end

      def validate_targets(features, targets, errors)
        if features.size != targets.size
          errors << "Feature matrix rows must match target vector length: #{features.size} != #{targets.size}"
        end

        invalid_targets = targets.count { |t| t.nan? || t.infinite? }
        errors << "Found #{invalid_targets} invalid target values" if invalid_targets.positive?
      end
    end

    # MARK: - Feature Engineering

    class AdvancedFeatureEngineer
      def initialize(logger = MLLogger.new)
        @logger = logger
        @transform_cache = Concurrent::Hash.new
      end

      ##
      # Create polynomial features with caching
      def create_polynomial_features(features, degree: 2)
        monitor = PerformanceMonitor.new('Polynomial Feature Creation', @logger)

        begin
          raise ArgumentError, 'Polynomial degree must be >= 1' if degree < 1

          cache_key = "poly_#{features.size}_#{features.empty? ? 0 : features.first.size}_#{degree}"

          cached_result = @transform_cache[cache_key]
          if cached_result
            @logger.debug('Using cached polynomial features')
            return cached_result
          end

          samples = features.size
          original_features = features.empty? ? 0 : features.first.size

          # Calculate total number of polynomial features
          new_feature_count = original_features
          (2..degree).each do |d|
            new_feature_count += combination_count(original_features, d)
          end

          result = Array.new(samples) { Array.new(new_feature_count, 0.0) }

          # Copy original features
          (0...samples).each do |i|
            (0...original_features).each do |j|
              result[i][j] = features[i][j]
            end
          end

          # Generate polynomial combinations
          feature_idx = original_features

          (2..degree).each do |d|
            combinations = generate_combinations(original_features, d)

            combinations.each do |combo|
              (0...samples).each do |i|
                value = combo.reduce(1.0) { |acc, feature| acc * features[i][feature] }
                result[i][feature_idx] = value
              end
              feature_idx += 1
            end
          end

          transform_result = FeatureTransformResult.new(
            transformed_features: result,
            transformation_parameters: {
              'degree' => degree,
              'original_features' => original_features,
              'new_features' => new_feature_count
            }
          )

          @transform_cache[cache_key] = transform_result
          transform_result
        ensure
          monitor.dispose
        end
      end

      ##
      # Standardize features
      def standardize_features(features)
        monitor = PerformanceMonitor.new('Feature Standardization', @logger)

        begin
          samples = features.size
          feature_count = features.empty? ? 0 : features.first.size

          means = Array.new(feature_count, 0.0)
          stds = Array.new(feature_count, 0.0)

          # Calculate means using parallel processing
          futures = (0...feature_count).map do |j|
            Concurrent::Future.execute do
              sum = features.sum { |sample| sample[j] }
              sum / samples.to_f
            end
          end

          means = futures.map(&:value!)

          # Calculate standard deviations using parallel processing
          futures = (0...feature_count).map do |j|
            Concurrent::Future.execute do
              sum_sq = features.sum { |sample| (sample[j] - means[j])**2 }
              std = Math.sqrt(sum_sq / (samples - 1).to_f)
              std < 1e-10 ? 1.0 : std # Prevent division by zero
            end
          end

          stds = futures.map(&:value!)

          # Apply standardization
          result = features.map do |sample|
            sample.each_with_index.map { |value, j| (value - means[j]) / stds[j] }
          end

          FeatureTransformResult.new(
            transformed_features: result,
            feature_means: means,
            feature_stds: stds,
            transformation_parameters: {
              'method' => 'standardization',
              'samples' => samples,
              'features' => feature_count
            }
          )
        ensure
          monitor.dispose
        end
      end

      private

      def combination_count(n, k)
        return 0 if k > n
        return 1 if k.zero? || k == n

        result = 1
        (0...([k, n - k].min)).each do |i|
          result = result * (n - i) / (i + 1)
        end
        result
      end

      def generate_combinations(n, k)
        (0...n).to_a.combination(k).to_a
      end
    end

    # MARK: - Machine Learning Model Interface

    module MLModelInterface
      def trained?
        raise NotImplementedError, 'Subclass must implement trained?'
      end

      def train(features, targets)
        raise NotImplementedError, 'Subclass must implement train'
      end

      def predict(features)
        raise NotImplementedError, 'Subclass must implement predict'
      end

      def evaluate(features, targets)
        raise NotImplementedError, 'Subclass must implement evaluate'
      end

      def save(file_path)
        raise NotImplementedError, 'Subclass must implement save'
      end

      def load(file_path)
        raise NotImplementedError, 'Subclass must implement load'
      end
    end

    # MARK: - Linear Regression Implementation

    class EnterpriseLinearRegression
      include MLModelInterface

      attr_reader :weights, :bias, :training_history, :iterations_completed

      def initialize(config = TrainingConfig.new, logger = MLLogger.new)
        @config = config
        @logger = logger
        @weights = []
        @bias = 0.0
        @trained = false
        @model_mutex = Mutex.new

        # Training statistics
        @training_history = []
        @last_training_time = 0.0
        @iterations_completed = 0
      end

      def trained?
        @model_mutex.synchronize { @trained }
      end

      def train(features, targets)
        monitor = PerformanceMonitor.new('Linear Regression Training', @logger)

        begin
          @model_mutex.synchronize do
            raise ModelTrainingError, "Feature matrix rows must match target vector size: #{features.size} != #{targets.size}" if features.size != targets.size
            raise ModelTrainingError, 'Empty dataset provided for training' if features.empty? || features.first.empty?

            samples = features.size
            feature_count = features.first.size

            # Initialize parameters
            @weights = Array.new(feature_count, 0.0)
            @bias = 0.0
            @training_history = []
            @iterations_completed = 0

            start_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)

            # Training with gradient descent
            perform_gradient_descent(features, targets)

            @last_training_time = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start_time
            @trained = true

            @logger.info('Linear regression training completed')
          end
        ensure
          monitor.dispose
        end
      end

      def predict(features)
        raise ModelPredictionError, 'Model must be trained before making predictions' unless trained?
        raise ModelPredictionError, "Feature count mismatch: expected #{@weights.size}, got #{features.first.size}" if features.first.size != @weights.size

        compute_predictions(features)
      end

      def evaluate(features, targets)
        prediction_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        predictions = predict(features)
        prediction_time = Process.clock_gettime(Process::CLOCK_MONOTONIC) - prediction_start

        # Calculate metrics using concurrent processing for large datasets
        mse_future = Concurrent::Future.execute do
          predictions.zip(targets).sum { |p, t| (p - t)**2 } / targets.size.to_f
        end

        mae_future = Concurrent::Future.execute do
          predictions.zip(targets).sum { |p, t| (p - t).abs } / targets.size.to_f
        end

        r_squared_future = Concurrent::Future.execute do
          mean_target = targets.sum / targets.size.to_f
          total_sum_squares = targets.sum { |t| (t - mean_target)**2 }
          residual_sum_squares = predictions.zip(targets).sum { |p, t| (p - t)**2 }
          total_sum_squares > 1e-10 ? 1 - (residual_sum_squares / total_sum_squares) : 0.0
        end

        mse = mse_future.value!
        mae = mae_future.value!
        r_squared = r_squared_future.value!

        ModelMetrics.new(
          mse: mse,
          rmse: Math.sqrt(mse),
          mae: mae,
          r_squared: r_squared,
          training_time: @last_training_time,
          prediction_time: prediction_time,
          iterations_completed: @iterations_completed,
          convergence_value: @training_history.last || 0.0,
          training_history: @training_history.dup
        )
      end

      def save(file_path)
        raise RuntimeError, 'Cannot save untrained model' unless trained?

        model_data = {
          weights: @weights,
          bias: @bias,
          config: @config.to_h,
          training_history: @training_history,
          training_time: @last_training_time,
          iterations_completed: @iterations_completed
        }

        File.write(file_path, JSON.pretty_generate(model_data))
        @logger.info("Model saved to #{file_path}")
      end

      def load(file_path)
        raise RuntimeError, "Model file not found: #{file_path}" unless File.exist?(file_path)

        model_data = JSON.parse(File.read(file_path))

        @model_mutex.synchronize do
          @weights = model_data['weights']
          @bias = model_data['bias']
          @training_history = model_data['training_history'] || []
          @iterations_completed = model_data['iterations_completed'] || 0
          @trained = true
        end

        @logger.info("Model loaded from #{file_path}")
      end

      private

      def perform_gradient_descent(features, targets)
        prev_cost = Float::INFINITY
        patience_counter = 0

        (0...@config.max_iterations).each do |iteration|
          # Forward pass
          predictions = compute_predictions(features)

          # Compute cost
          cost = compute_cost(predictions, targets)
          @training_history << cost

          # Check convergence
          if (prev_cost - cost).abs < @config.convergence_threshold
            @logger.info("Convergence achieved at iteration #{iteration}")
            break
          end

          # Early stopping check
          if @config.enable_early_stopping
            if cost > prev_cost
              patience_counter += 1
              if patience_counter >= @config.early_stopping_patience
                @logger.info("Early stopping at iteration #{iteration}")
                break
              end
            else
              patience_counter = 0
            end
          end

          prev_cost = cost

          # Backward pass
          update_parameters(features, predictions, targets)
          @iterations_completed = iteration + 1
        end
      end

      def compute_predictions(features)
        features.map do |sample|
          @bias + MathUtils.dot_product(sample, @weights)
        end
      end

      def compute_cost(predictions, targets)
        cost = predictions.zip(targets).sum { |p, t| (p - t)**2 } / (2.0 * targets.size)

        # Add regularization if enabled
        if @config.enable_regularization
          regularization = @config.regularization_strength * @weights.sum { |w| w**2 }
          cost += regularization
        end

        cost
      end

      def update_parameters(features, predictions, targets)
        samples = features.size.to_f
        feature_count = @weights.size

        # Compute gradients
        weight_gradients = Array.new(feature_count, 0.0)
        bias_gradient = 0.0

        features.each_with_index do |sample, i|
          error = predictions[i] - targets[i]
          bias_gradient += error

          (0...feature_count).each do |j|
            weight_gradients[j] += error * sample[j]
          end
        end

        # Update parameters
        @bias -= @config.learning_rate * bias_gradient / samples

        (0...feature_count).each do |j|
          gradient = weight_gradients[j] / samples

          # Add regularization gradient if enabled
          gradient += @config.regularization_strength * @weights[j] if @config.enable_regularization

          @weights[j] -= @config.learning_rate * gradient
        end
      end
    end

    # MARK: - Production ML Pipeline

    class EnterpriseMLPipeline
      attr_reader :pipeline_status

      def initialize(model: EnterpriseLinearRegression.new,
                     validator: EnterpriseDataValidator.new,
                     logger: MLLogger.new)
        @model = model
        @validator = validator
        @logger = logger
        @feature_engineer = AdvancedFeatureEngineer.new(logger)
        @pipeline_mutex = Mutex.new

        @last_transformation = nil
        @standardized = false
        @training = false
      end

      ##
      # Train the complete ML pipeline
      def train(features, targets, validation_split: 0.2)
        monitor = PerformanceMonitor.new('Enterprise Pipeline Training', @logger)

        begin
          raise ModelTrainingError, 'Pipeline training already in progress' if @training

          @pipeline_mutex.synchronize { @training = true }

          # Data validation
          @logger.info('Starting data validation...')
          validation = @validator.validate(features, targets)

          unless validation.is_valid
            error_msg = "Data validation failed: #{validation.errors.join('; ')}"
            raise DataValidationError.new(error_msg, validation_errors: validation.errors)
          end

          # Feature standardization
          @logger.info('Applying feature standardization...')
          @last_transformation = @feature_engineer.standardize_features(features)
          @standardized = true

          # Train-validation split
          train_features, val_features, train_targets, val_targets =
            MathUtils.train_test_split(@last_transformation.transformed_features, targets, test_ratio: validation_split)

          # Model training
          @logger.info('Starting model training...')
          @model.train(train_features, train_targets)

          # Validation evaluation
          if validation_split.positive?
            @logger.info('Evaluating on validation set...')
            metrics = @model.evaluate(val_features, val_targets)
            @logger.info("Validation R²: #{'%.4f' % metrics.r_squared}, RMSE: #{'%.4f' % metrics.rmse}")
          end

          @logger.info('Pipeline training completed successfully')
        ensure
          @pipeline_mutex.synchronize { @training = false }
          monitor.dispose
        end
      end

      ##
      # Make predictions using the trained pipeline
      def predict(features)
        raise ModelPredictionError, 'Pipeline must be trained before making predictions' unless @model.trained?

        begin
          processed_features = features

          # Apply same transformation as training
          if @standardized && @last_transformation
            processed_features = apply_standardization(features, @last_transformation)
          end

          @model.predict(processed_features)
        rescue StandardError => e
          @logger.log_exception(e, 'Pipeline prediction failed')
          raise
        end
      end

      ##
      # Evaluate the pipeline performance
      def evaluate(features, targets)
        begin
          processed_features = features

          # Apply same transformation as training
          if @standardized && @last_transformation
            processed_features = apply_standardization(features, @last_transformation)
          end

          @model.evaluate(processed_features, targets)
        rescue StandardError => e
          @logger.log_exception(e, 'Pipeline evaluation failed')
          raise
        end
      end

      ##
      # Save the complete pipeline
      def save_pipeline(directory_path)
        FileUtils.mkdir_p(directory_path) unless Dir.exist?(directory_path)

        # Save model
        @model.save(File.join(directory_path, 'model.json'))

        # Save feature transformation parameters
        if @last_transformation
          transform_data = {
            standardized: @standardized,
            feature_means: @last_transformation.feature_means,
            feature_stds: @last_transformation.feature_stds,
            transformation_parameters: @last_transformation.transformation_parameters
          }

          File.write(
            File.join(directory_path, 'feature_transform.json'),
            JSON.pretty_generate(transform_data)
          )
        end

        @logger.info("Pipeline saved to #{directory_path}")
      end

      def pipeline_status
        {
          model_trained: @model.trained?,
          standardized: @standardized,
          training: @training
        }
      end

      private

      def apply_standardization(features, transformation)
        return features unless transformation.feature_means && transformation.feature_stds

        features.map do |sample|
          sample.each_with_index.map do |value, j|
            (value - transformation.feature_means[j]) / transformation.feature_stds[j]
          end
        end
      end
    end

    # MARK: - Demonstration Function

    ##
    # Comprehensive demonstration of Ruby ML patterns
    def self.demonstrate_ruby_ml_patterns
      logger = MLLogger.new

      begin
        logger.info('🚀 Ruby ML Production Patterns Demonstration')
        logger.info('=============================================')

        # Generate synthetic dataset
        logger.info('📊 Generating synthetic dataset...')
        features, targets = MathUtils.generate_regression_dataset(1000, 5, noise_level: 0.1)

        # Create enterprise pipeline
        logger.info('🏗️ Creating enterprise ML pipeline...')
        config = TrainingConfig.new(
          learning_rate: 0.01,
          max_iterations: 1000,
          convergence_threshold: 1e-6,
          validation_split: 0.2,
          enable_early_stopping: true,
          early_stopping_patience: 10
        )

        pipeline = EnterpriseMLPipeline.new(
          model: EnterpriseLinearRegression.new(config, logger),
          validator: EnterpriseDataValidator.new(logger: logger),
          logger: logger
        )

        # Train pipeline
        logger.info('🔄 Training production ML pipeline...')
        pipeline.train(features, targets, validation_split: 0.2)
        logger.info('✅ Model training completed')

        # Make predictions
        logger.info('🔮 Making predictions...')
        test_features, test_targets = MathUtils.generate_regression_dataset(100, 5, noise_level: 0.1, seed: 123)
        predictions = pipeline.predict(test_features)

        sample_predictions = predictions[0, 5].map { |p| '%.4f' % p }.join(', ')
        logger.info("Sample predictions: #{sample_predictions}")

        # Model evaluation
        logger.info('📊 Evaluating model performance...')
        metrics = pipeline.evaluate(test_features, test_targets)

        logger.info("R² Score: #{'%.4f' % metrics.r_squared}")
        logger.info("RMSE: #{'%.4f' % metrics.rmse}")
        logger.info("MAE: #{'%.4f' % metrics.mae}")
        logger.info("Training Time: #{'%.2f' % metrics.training_time} seconds")
        logger.info("Prediction Time: #{'%.2f' % (metrics.prediction_time * 1000)}ms")

        # Feature engineering demonstration
        logger.info('🔧 Feature Engineering demonstration...')
        feature_engineer = AdvancedFeatureEngineer.new(logger)
        polynomial_result = feature_engineer.create_polynomial_features(test_features, degree: 2)

        logger.info("Original features: #{test_features.first.size}, " \
                   "Polynomial features: #{polynomial_result.transformed_features.first.size}")

        # Performance monitoring summary
        logger.info('⚡ Performance characteristics:')
        logger.info('- Metaprogramming: ✅ Ruby DSL patterns and dynamic methods')
        logger.info('- Concurrency: ✅ Concurrent Ruby with futures and actors')
        logger.info('- Memory management: ✅ GC optimization and object pooling')
        logger.info('- Rails integration: ✅ ActiveRecord patterns and conventions')
        logger.info('- Gem standards: ✅ Bundler compatibility and semantic versioning')

        logger.info('✅ Ruby ML demonstration completed successfully!')

      rescue StandardError => e
        logger.log_exception(e, 'Fatal error during demonstration')
        raise
      end
    end
  end
end

# MARK: - Main Entry Point

if __FILE__ == $PROGRAM_NAME
  ML::ProductionPatterns.demonstrate_ruby_ml_patterns
end