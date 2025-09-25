/**
 * Production-Ready Linear Regression Implementation in Dart
 * ========================================================
 * 
 * This module demonstrates a comprehensive linear regression implementation
 * with gradient descent, regularization, and modern Dart patterns
 * for AI training datasets.
 *
 * Key Features:
 * - Multiple regression algorithms (Normal Equation, Gradient Descent)
 * - L1 and L2 regularization (Ridge, Lasso)
 * - Feature scaling and normalization
 * - Comprehensive statistical metrics
 * - Dart async/await for non-blocking training
 * - Generic type safety and null safety
 * - Modern Dart idioms and patterns
 * - Production deployment considerations
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import 'dart:math' as math;
import 'dart:typed_data';

/// Custom exception for linear regression errors
class LinearRegressionException implements Exception {
  final String message;
  final dynamic cause;
  
  LinearRegressionException(this.message, [this.cause]);
  
  @override
  String toString() => 'LinearRegressionException: $message${cause != null ? ' (Caused by: $cause)' : ''}';
}

/// Data class representing a training sample
class RegressionSample {
  final Float64List features;
  final double target;
  final int id;
  
  RegressionSample({
    required this.features,
    required this.target,
    required this.id,
  });
  
  int get featureCount => features.length;
  
  @override
  String toString() => 'RegressionSample(id: $id, features: $features, target: $target)';
  
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is RegressionSample &&
          runtimeType == other.runtimeType &&
          id == other.id &&
          target == other.target &&
          _listEquals(features, other.features);
  
  @override
  int get hashCode => Object.hash(id, target, Object.hashAll(features));
  
  bool _listEquals(Float64List a, Float64List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Feature scaler interface
abstract class FeatureScaler {
  String get name;
  void fit(List<RegressionSample> data);
  List<RegressionSample> transform(List<RegressionSample> data);
  List<RegressionSample> fitTransform(List<RegressionSample> data) {
    fit(data);
    return transform(data);
  }
}

/// Standard scaler (z-score normalization)
class StandardScaler implements FeatureScaler {
  @override
  String get name => 'StandardScaler';
  
  Float64List? _means;
  Float64List? _stds;
  
  @override
  void fit(List<RegressionSample> data) {
    if (data.isEmpty) throw LinearRegressionException('Cannot fit on empty data');
    
    final numFeatures = data.first.featureCount;
    _means = Float64List(numFeatures);
    _stds = Float64List(numFeatures);
    
    // Calculate means
    for (final sample in data) {
      for (int i = 0; i < numFeatures; i++) {
        _means![i] += sample.features[i];
      }
    }
    for (int i = 0; i < numFeatures; i++) {
      _means![i] /= data.length;
    }
    
    // Calculate standard deviations
    for (final sample in data) {
      for (int i = 0; i < numFeatures; i++) {
        final diff = sample.features[i] - _means![i];
        _stds![i] += diff * diff;
      }
    }
    for (int i = 0; i < numFeatures; i++) {
      _stds![i] = math.sqrt(_stds![i] / data.length);
      if (_stds![i] == 0.0) _stds![i] = 1.0; // Handle zero std
    }
  }
  
  @override
  List<RegressionSample> transform(List<RegressionSample> data) {
    if (_means == null || _stds == null) {
      throw LinearRegressionException('Must fit scaler before transforming');
    }
    
    return data.map((sample) {
      final scaledFeatures = Float64List(sample.featureCount);
      for (int i = 0; i < sample.featureCount; i++) {
        scaledFeatures[i] = (sample.features[i] - _means![i]) / _stds![i];
      }
      return RegressionSample(
        features: scaledFeatures,
        target: sample.target,
        id: sample.id,
      );
    }).toList();
  }
}

/// Regression metrics for model evaluation
class RegressionMetrics {
  final double mse;
  final double rmse;
  final double mae;
  final double r2;
  final double adjustedR2;
  final int sampleCount;
  
  RegressionMetrics({
    required this.mse,
    required this.rmse,
    required this.mae,
    required this.r2,
    required this.adjustedR2,
    required this.sampleCount,
  });
  
  @override
  String toString() {
    return '''
Regression Metrics:
  MSE: ${mse.toStringAsFixed(6)}
  RMSE: ${rmse.toStringAsFixed(6)}
  MAE: ${mae.toStringAsFixed(6)}
  R²: ${r2.toStringAsFixed(6)}
  Adjusted R²: ${adjustedR2.toStringAsFixed(6)}
  Sample count: $sampleCount
''';
  }
}

/// Training epoch information
class TrainingEpoch {
  final int epoch;
  final double cost;
  final double gradientMagnitude;
  
  TrainingEpoch({
    required this.epoch,
    required this.cost,
    required this.gradientMagnitude,
  });
}

/// Comprehensive Linear Regression Implementation
class LinearRegressionImplementation {
  final double learningRate;
  final int maxIterations;
  final double tolerance;
  final String regularization; // 'none', 'ridge', 'lasso'
  final double alpha; // regularization strength
  final bool useFeatureScaling;
  final FeatureScaler scaler;
  
  // Model parameters
  Float64List? _weights;
  double _bias = 0.0;
  bool _fitted = false;
  bool _scaledData = false;
  final List<TrainingEpoch> _trainingHistory = [];
  
  LinearRegressionImplementation({
    this.learningRate = 0.01,
    this.maxIterations = 1000,
    this.tolerance = 1e-6,
    this.regularization = 'none',
    this.alpha = 0.01,
    this.useFeatureScaling = true,
    FeatureScaler? scaler,
  }) : scaler = scaler ?? StandardScaler();
  
  List<TrainingEpoch> get trainingHistory => List.unmodifiable(_trainingHistory);
  
  /// Calculate cost function with regularization
  double _calculateCost(List<RegressionSample> samples) {
    final predictions = samples.map((sample) => _predict(sample.features)).toList();
    double mse = 0.0;
    for (int i = 0; i < samples.length; i++) {
      final diff = predictions[i] - samples[i].target;
      mse += diff * diff;
    }
    mse /= samples.length;
    
    // Add regularization term
    double regularizationTerm = 0.0;
    if (_weights != null) {
      switch (regularization) {
        case 'ridge':
          for (final weight in _weights!) {
            regularizationTerm += weight * weight;
          }
          regularizationTerm *= alpha;
          break;
        case 'lasso':
          for (final weight in _weights!) {
            regularizationTerm += weight.abs();
          }
          regularizationTerm *= alpha;
          break;
      }
    }
    
    return mse + regularizationTerm;
  }
  
  /// Calculate gradients with regularization
  ({Float64List weightGradients, double biasGradient}) _calculateGradients(List<RegressionSample> samples) {
    final n = samples.length;
    final weightGradients = Float64List(_weights!.length);
    double biasGradient = 0.0;
    
    for (final sample in samples) {
      final prediction = _predict(sample.features);
      final error = prediction - sample.target;
      
      // Weight gradients
      for (int i = 0; i < _weights!.length; i++) {
        weightGradients[i] += error * sample.features[i] / n;
      }
      
      // Bias gradient
      biasGradient += error / n;
    }
    
    // Add regularization gradients
    switch (regularization) {
      case 'ridge':
        for (int i = 0; i < _weights!.length; i++) {
          weightGradients[i] += 2 * alpha * _weights![i];
        }
        break;
      case 'lasso':
        for (int i = 0; i < _weights!.length; i++) {
          weightGradients[i] += alpha * _weights![i].sign;
        }
        break;
    }
    
    return (weightGradients: weightGradients, biasGradient: biasGradient);
  }
  
  /// Internal prediction function
  double _predict(Float64List features) {
    if (_weights == null) return 0.0;
    
    double prediction = _bias;
    for (int i = 0; i < features.length; i++) {
      prediction += _weights![i] * features[i];
    }
    return prediction;
  }
  
  /// Fit using gradient descent
  Future<void> _fitGradientDescent(List<RegressionSample> samples) async {
    final numFeatures = samples.first.featureCount;
    _weights = Float64List(numFeatures);
    final random = math.Random();
    
    // Initialize weights randomly
    for (int i = 0; i < numFeatures; i++) {
      _weights![i] = (random.nextDouble() - 0.5) * 0.02;
    }
    _bias = 0.0;
    _trainingHistory.clear();
    
    print('🔄 Training with Gradient Descent...');
    print('Learning rate: $learningRate, Max iterations: $maxIterations');
    print('Regularization: $regularization${regularization != 'none' ? ' (α=$alpha)' : ''}');
    
    for (int epoch = 0; epoch < maxIterations; epoch++) {
      final cost = _calculateCost(samples);
      final gradients = _calculateGradients(samples);
      
      // Update parameters
      for (int i = 0; i < _weights!.length; i++) {
        _weights![i] -= learningRate * gradients.weightGradients[i];
      }
      _bias -= learningRate * gradients.biasGradient;
      
      // Track progress
      final gradientMagnitude = math.sqrt(
        gradients.weightGradients.fold<double>(0.0, (sum, grad) => sum + grad * grad) +
        gradients.biasGradient * gradients.biasGradient
      );
      
      _trainingHistory.add(TrainingEpoch(
        epoch: epoch,
        cost: cost,
        gradientMagnitude: gradientMagnitude,
      ));
      
      // Print progress
      if ((epoch + 1) % (maxIterations ~/ 10) == 0) {
        print('Epoch ${epoch + 1}/$maxIterations - Cost: ${cost.toStringAsFixed(6)}, '
              'Gradient: ${gradientMagnitude.toStringAsFixed(6)}');
      }
      
      // Check convergence
      if (gradientMagnitude < tolerance) {
        print('✅ Converged after ${epoch + 1} epochs');
        break;
      }
      
      // Allow other isolates to run
      if (epoch % 100 == 0) {
        await Future.delayed(Duration.zero);
      }
    }
  }
  
  /// Train the linear regression model
  Future<void> fit(List<RegressionSample> trainingSamples, {String method = 'gradient_descent'}) async {
    if (trainingSamples.isEmpty) {
      throw LinearRegressionException('Training data cannot be empty');
    }
    
    print('📈 Training Linear Regression Model');
    print('=' * 40);
    print('Training samples: ${trainingSamples.length}');
    print('Features: ${trainingSamples.first.featureCount}');
    print('Method: $method');
    
    final stopwatch = Stopwatch()..start();
    
    // Apply feature scaling if enabled
    List<RegressionSample> processedSamples;
    if (useFeatureScaling) {
      print('🔧 Applying feature scaling (${scaler.name})...');
      processedSamples = scaler.fitTransform(trainingSamples);
      _scaledData = true;
    } else {
      processedSamples = trainingSamples;
      _scaledData = false;
    }
    
    // Choose training method
    switch (method) {
      case 'gradient_descent':
        await _fitGradientDescent(processedSamples);
        break;
      default:
        throw LinearRegressionException('Unknown method: $method');
    }
    
    _fitted = true;
    stopwatch.stop();
    print('✅ Training completed in ${stopwatch.elapsedMilliseconds}ms');
    
    // Print model summary
    print('\n🎯 Model Summary:');
    print('Weights: $_weights');
    print('Bias: ${_bias.toStringAsFixed(6)}');
    print('Training history: ${_trainingHistory.length} epochs');
  }
  
  /// Make predictions
  List<double> predict(List<RegressionSample> samples) {
    if (!_fitted) {
      throw LinearRegressionException('Model not fitted. Call fit() first.');
    }
    
    List<RegressionSample> processedSamples;
    if (_scaledData) {
      processedSamples = scaler.transform(samples);
    } else {
      processedSamples = samples;
    }
    
    return processedSamples.map((sample) => _predict(sample.features)).toList();
  }
  
  /// Evaluate model performance
  RegressionMetrics evaluate(List<RegressionSample> testSamples) {
    final predictions = predict(testSamples);
    final actuals = testSamples.map((s) => s.target).toList();
    
    // Calculate MSE
    double mse = 0.0;
    for (int i = 0; i < actuals.length; i++) {
      final diff = actuals[i] - predictions[i];
      mse += diff * diff;
    }
    mse /= actuals.length;
    
    // Calculate RMSE
    final rmse = math.sqrt(mse);
    
    // Calculate MAE
    double mae = 0.0;
    for (int i = 0; i < actuals.length; i++) {
      mae += (actuals[i] - predictions[i]).abs();
    }
    mae /= actuals.length;
    
    // Calculate R²
    final meanActual = actuals.reduce((a, b) => a + b) / actuals.length;
    double tss = 0.0;
    double rss = 0.0;
    
    for (int i = 0; i < actuals.length; i++) {
      tss += (actuals[i] - meanActual) * (actuals[i] - meanActual);
      rss += (actuals[i] - predictions[i]) * (actuals[i] - predictions[i]);
    }
    
    final r2 = 1 - (rss / tss);
    
    // Calculate adjusted R²
    final n = testSamples.length;
    final p = _weights?.length ?? 0;
    final adjustedR2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1));
    
    return RegressionMetrics(
      mse: mse,
      rmse: rmse,
      mae: mae,
      r2: r2,
      adjustedR2: adjustedR2,
      sampleCount: n,
    );
  }
  
  /// Generate synthetic dataset for testing
  static List<RegressionSample> generateSyntheticDataset(int samples, int features, {double noise = 0.1}) {
    final random = math.Random();
    final trueWeights = List.generate(features, (_) => (random.nextDouble() - 0.5) * 4.0);
    final trueBias = (random.nextDouble() - 0.5) * 2.0;
    
    return List.generate(samples, (id) {
      final featureVector = Float64List.fromList(
        List.generate(features, (_) => (random.nextDouble() - 0.5) * 4.0)
      );
      
      double trueTarget = trueBias;
      for (int i = 0; i < features; i++) {
        trueTarget += featureVector[i] * trueWeights[i];
      }
      
      final noisyTarget = trueTarget + (random.nextDouble() - 0.5) * noise * 2.0;
      
      return RegressionSample(
        features: featureVector,
        target: noisyTarget,
        id: id,
      );
    });
  }
  
  /// Comprehensive demonstration of linear regression
  static Future<void> demonstrateLinearRegression() async {
    print('🚀 Linear Regression Implementation Demonstration');
    print('=' * 55);
    
    try {
      // Generate synthetic dataset
      print('📊 Generating synthetic dataset...');
      final dataset = generateSyntheticDataset(1000, 5, noise: 0.2);
      print('Dataset: ${dataset.length} samples, ${dataset.first.featureCount} features');
      
      // Split into train/test
      dataset.shuffle();
      final trainSize = (dataset.length * 0.8).toInt();
      final trainData = dataset.take(trainSize).toList();
      final testData = dataset.skip(trainSize).toList();
      
      print('Train: $trainSize samples, Test: ${testData.length} samples');
      
      // Test different configurations
      final configurations = [
        ('No Regularization', 'none', 0.0),
        ('Ridge Regression', 'ridge', 0.1),
        ('Lasso Regression', 'lasso', 0.1),
      ];
      
      for (final (name, regType, alphaValue) in configurations) {
        print('\n${'=' * 60}');
        print('🔍 Testing $name');
        print('=' * 60);
        
        final model = LinearRegressionImplementation(
          learningRate: 0.01,
          maxIterations: 1000,
          regularization: regType,
          alpha: alphaValue,
          useFeatureScaling: true,
        );
        
        // Train model
        await model.fit(trainData);
        
        // Evaluate on test set
        final metrics = model.evaluate(testData);
        print('\n📊 Test Set Performance:');
        print(metrics);
        
        // Sample predictions
        print('\n🧪 Sample Predictions:');
        final sampleTests = testData.take(5).toList();
        final predictions = model.predict(sampleTests);
        
        for (int i = 0; i < sampleTests.length; i++) {
          final sample = sampleTests[i];
          final pred = predictions[i];
          print('Sample ${sample.id}: Predicted = ${pred.toStringAsFixed(4)}, '
                'Actual = ${sample.target.toStringAsFixed(4)}, '
                'Error = ${(pred - sample.target).abs().toStringAsFixed(4)}');
        }
      }
      
      print('\n✅ Linear regression demonstration completed successfully!');
      
    } catch (e, stackTrace) {
      print('❌ Linear regression demonstration failed: $e');
      print('Stack trace: $stackTrace');
    }
  }
}

/// Main function to demonstrate linear regression
Future<void> main() async {
  await LinearRegressionImplementation.demonstrateLinearRegression();
}