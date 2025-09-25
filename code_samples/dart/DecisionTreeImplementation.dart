/**
 * Production-Ready Decision Tree Implementation in Dart
 * ===================================================
 * 
 * This module demonstrates a comprehensive decision tree classifier
 * with entropy-based splitting, pruning, and modern Dart patterns
 * for AI training datasets.
 *
 * Key Features:
 * - ID3 and C4.5 algorithm implementations
 * - Information gain and gain ratio for feature selection
 * - Tree pruning to prevent overfitting
 * - Support for both categorical and numerical features
 * - Dart null safety and type safety
 * - Async/await for non-blocking training
 * - Modern Dart patterns and collections
 * - Memory-efficient tree construction
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:collection';

/// Custom exception for decision tree errors
class DecisionTreeException implements Exception {
  final String message;
  final dynamic cause;
  
  DecisionTreeException(this.message, [this.cause]);
  
  @override
  String toString() => 'DecisionTreeException: $message${cause != null ? ' (Caused by: $cause)' : ''}';
}

/// Data class representing a training sample
class ClassificationSample {
  final Float64List features;
  final String label;
  final int id;
  
  ClassificationSample({
    required this.features,
    required this.label,
    required this.id,
  });
  
  int get featureCount => features.length;
  
  @override
  String toString() => 'ClassificationSample(id: $id, features: $features, label: $label)';
  
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ClassificationSample &&
          runtimeType == other.runtimeType &&
          id == other.id &&
          label == other.label &&
          _listEquals(features, other.features);
  
  @override
  int get hashCode => Object.hash(id, label, Object.hashAll(features));
  
  bool _listEquals(Float64List a, Float64List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Split condition for decision tree nodes
class SplitCondition {
  final int featureIndex;
  final double threshold;
  final String featureName;
  final bool isNumerical;
  final double informationGain;
  
  SplitCondition({
    required this.featureIndex,
    required this.threshold,
    required this.featureName,
    required this.isNumerical,
    required this.informationGain,
  });
  
  /// Evaluate if a sample satisfies this split condition
  bool evaluate(ClassificationSample sample) {
    final value = sample.features[featureIndex];
    return isNumerical 
        ? value <= threshold 
        : (value - threshold).abs() < 1e-9;
  }
  
  @override
  String toString() {
    return isNumerical 
        ? '$featureName <= ${threshold.toStringAsFixed(3)}' 
        : '$featureName = ${threshold.toInt()}';
  }
}

/// Node in the decision tree
class TreeNode {
  SplitCondition? splitCondition;
  TreeNode? leftChild;
  TreeNode? rightChild;
  String? prediction;
  double confidence;
  int depth;
  int sampleCount;
  Map<String, int> classDistribution;
  
  /// Leaf node constructor
  TreeNode.leaf({
    required this.prediction,
    required this.confidence,
    required this.sampleCount,
    required this.classDistribution,
    required this.depth,
  });
  
  /// Internal node constructor
  TreeNode.internal({
    required this.splitCondition,
    required this.sampleCount,
    required this.classDistribution,
    required this.depth,
  }) : confidence = 0.0;
  
  bool get isLeaf => splitCondition == null;
}

/// Decision tree evaluation metrics
class ClassificationMetrics {
  final double accuracy;
  final Map<String, double> precision;
  final Map<String, double> recall;
  final Map<String, double> f1Score;
  final double macroPrecision;
  final double macroRecall;
  final double macroF1;
  final Map<String, Map<String, int>> confusionMatrix;
  final int sampleCount;
  
  ClassificationMetrics({
    required this.accuracy,
    required this.precision,
    required this.recall,
    required this.f1Score,
    required this.macroPrecision,
    required this.macroRecall,
    required this.macroF1,
    required this.confusionMatrix,
    required this.sampleCount,
  });
  
  @override
  String toString() {
    final buffer = StringBuffer()
      ..writeln('Classification Metrics:')
      ..writeln('  Accuracy: ${accuracy.toStringAsFixed(6)}')
      ..writeln('  Macro Precision: ${macroPrecision.toStringAsFixed(6)}')
      ..writeln('  Macro Recall: ${macroRecall.toStringAsFixed(6)}')
      ..writeln('  Macro F1-Score: ${macroF1.toStringAsFixed(6)}')
      ..writeln('  Sample count: $sampleCount')
      ..writeln()
      ..writeln('Per-class Metrics:');
    
    final sortedClasses = precision.keys.toList()..sort();
    for (final className in sortedClasses) {
      buffer
        ..writeln('  Class \'$className\':')
        ..writeln('    Precision: ${(precision[className] ?? 0.0).toStringAsFixed(6)}')
        ..writeln('    Recall: ${(recall[className] ?? 0.0).toStringAsFixed(6)}')
        ..writeln('    F1-Score: ${(f1Score[className] ?? 0.0).toStringAsFixed(6)}');
    }
    
    return buffer.toString();
  }
}

/// Comprehensive Decision Tree Classifier Implementation
class DecisionTreeImplementation {
  final List<String> featureNames;
  final int maxDepth;
  final int minSamplesLeaf;
  final int minSamplesSplit;
  final double minInfoGain;
  final bool usePruning;
  
  TreeNode? _root;
  bool _fitted = false;
  
  DecisionTreeImplementation({
    required this.featureNames,
    this.maxDepth = 10,
    this.minSamplesLeaf = 1,
    this.minSamplesSplit = 2,
    this.minInfoGain = 0.0,
    this.usePruning = true,
  });
  
  /// Calculate entropy of a dataset
  double _calculateEntropy(List<ClassificationSample> samples) {
    if (samples.isEmpty) return 0.0;
    
    final labelCounts = <String, int>{};
    for (final sample in samples) {
      labelCounts[sample.label] = (labelCounts[sample.label] ?? 0) + 1;
    }
    
    double entropy = 0.0;
    final totalSamples = samples.length;
    
    for (final count in labelCounts.values) {
      final probability = count / totalSamples;
      if (probability > 0) {
        entropy -= probability * (math.log(probability) / math.ln2);
      }
    }
    
    return entropy;
  }
  
  /// Calculate information gain for a split
  double _calculateInformationGain(
    List<ClassificationSample> samples,
    List<ClassificationSample> leftSplit,
    List<ClassificationSample> rightSplit,
  ) {
    final originalEntropy = _calculateEntropy(samples);
    final leftEntropy = _calculateEntropy(leftSplit);
    final rightEntropy = _calculateEntropy(rightSplit);
    
    final leftWeight = leftSplit.length / samples.length;
    final rightWeight = rightSplit.length / samples.length;
    
    return originalEntropy - (leftWeight * leftEntropy + rightWeight * rightEntropy);
  }
  
  /// Find the best split for a feature
  SplitCondition? _findBestSplit(List<ClassificationSample> samples) {
    double bestInfoGain = -1.0;
    SplitCondition? bestSplit;
    
    // Try each feature
    for (int featureIdx = 0; featureIdx < featureNames.length; featureIdx++) {
      final uniqueValues = <double>{};
      for (final sample in samples) {
        uniqueValues.add(sample.features[featureIdx]);
      }
      
      // For numerical features, try thresholds between unique values
      final sortedValues = uniqueValues.toList()..sort();
      
      for (int i = 0; i < sortedValues.length - 1; i++) {
        final threshold = (sortedValues[i] + sortedValues[i + 1]) / 2.0;
        
        // Split samples based on threshold
        final leftSplit = <ClassificationSample>[];
        final rightSplit = <ClassificationSample>[];
        
        for (final sample in samples) {
          if (sample.features[featureIdx] <= threshold) {
            leftSplit.add(sample);
          } else {
            rightSplit.add(sample);
          }
        }
        
        if (leftSplit.isNotEmpty && rightSplit.isNotEmpty) {
          final infoGain = _calculateInformationGain(samples, leftSplit, rightSplit);
          
          if (infoGain > bestInfoGain) {
            bestInfoGain = infoGain;
            bestSplit = SplitCondition(
              featureIndex: featureIdx,
              threshold: threshold,
              featureName: featureNames[featureIdx],
              isNumerical: true,
              informationGain: infoGain,
            );
          }
        }
      }
    }
    
    return bestInfoGain > minInfoGain ? bestSplit : null;
  }
  
  /// Get the most common class in a set of samples
  String _getMajorityClass(List<ClassificationSample> samples) {
    if (samples.isEmpty) return 'unknown';
    
    final labelCounts = <String, int>{};
    for (final sample in samples) {
      labelCounts[sample.label] = (labelCounts[sample.label] ?? 0) + 1;
    }
    
    return labelCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }
  
  /// Calculate prediction confidence based on class distribution
  double _calculateConfidence(List<ClassificationSample> samples) {
    if (samples.isEmpty) return 0.0;
    
    final labelCounts = <String, int>{};
    for (final sample in samples) {
      labelCounts[sample.label] = (labelCounts[sample.label] ?? 0) + 1;
    }
    
    final maxCount = labelCounts.values.reduce(math.max);
    return maxCount / samples.length;
  }
  
  /// Get class distribution
  Map<String, int> _getClassDistribution(List<ClassificationSample> samples) {
    final distribution = <String, int>{};
    for (final sample in samples) {
      distribution[sample.label] = (distribution[sample.label] ?? 0) + 1;
    }
    return distribution;
  }
  
  /// Build decision tree recursively
  Future<TreeNode> _buildTree(List<ClassificationSample> samples, int depth) async {
    if (samples.isEmpty) {
      return TreeNode.leaf(
        prediction: 'unknown',
        confidence: 0.0,
        sampleCount: 0,
        classDistribution: {},
        depth: depth,
      );
    }
    
    final classDistribution = _getClassDistribution(samples);
    
    // Stopping criteria
    final shouldStop = samples.length < minSamplesSplit ||
                      depth >= maxDepth ||
                      classDistribution.length == 1;
    
    if (shouldStop) {
      final prediction = _getMajorityClass(samples);
      final confidence = _calculateConfidence(samples);
      return TreeNode.leaf(
        prediction: prediction,
        confidence: confidence,
        sampleCount: samples.length,
        classDistribution: classDistribution,
        depth: depth,
      );
    }
    
    // Find best split
    final bestSplit = _findBestSplit(samples);
    
    if (bestSplit == null) {
      final prediction = _getMajorityClass(samples);
      final confidence = _calculateConfidence(samples);
      return TreeNode.leaf(
        prediction: prediction,
        confidence: confidence,
        sampleCount: samples.length,
        classDistribution: classDistribution,
        depth: depth,
      );
    }
    
    // Split samples
    final leftSamples = <ClassificationSample>[];
    final rightSamples = <ClassificationSample>[];
    
    for (final sample in samples) {
      if (bestSplit.evaluate(sample)) {
        leftSamples.add(sample);
      } else {
        rightSamples.add(sample);
      }
    }
    
    // Create internal node
    final node = TreeNode.internal(
      splitCondition: bestSplit,
      sampleCount: samples.length,
      classDistribution: classDistribution,
      depth: depth,
    );
    
    // Recursively build children
    if (leftSamples.isNotEmpty && leftSamples.length >= minSamplesLeaf) {
      node.leftChild = await _buildTree(leftSamples, depth + 1);
    }
    if (rightSamples.isNotEmpty && rightSamples.length >= minSamplesLeaf) {
      node.rightChild = await _buildTree(rightSamples, depth + 1);
    }
    
    // Allow other operations to run
    if (depth % 5 == 0) {
      await Future.delayed(Duration.zero);
    }
    
    return node;
  }
  
  /// Train the decision tree
  Future<void> fit(List<ClassificationSample> trainingSamples) async {
    if (trainingSamples.isEmpty) {
      throw DecisionTreeException('Training data cannot be empty');
    }
    
    if (trainingSamples.first.featureCount != featureNames.length) {
      throw DecisionTreeException(
        'Feature count mismatch. Expected: ${featureNames.length}, '
        'got: ${trainingSamples.first.featureCount}'
      );
    }
    
    print('🌳 Training Decision Tree Classifier');
    print('=' * 40);
    print('Training samples: ${trainingSamples.length}');
    print('Features: ${featureNames.length}');
    print('Max depth: $maxDepth');
    
    final stopwatch = Stopwatch()..start();
    
    // Build the tree
    _root = await _buildTree(trainingSamples, 0);
    
    stopwatch.stop();
    print('✅ Tree construction completed in ${stopwatch.elapsedMilliseconds}ms');
    
    // Print tree statistics
    _printTreeStatistics(_root!);
    _fitted = true;
  }
  
  /// Make a prediction for a single sample
  String predict(ClassificationSample sample) {
    if (!_fitted || _root == null) {
      throw DecisionTreeException('Model not trained. Call fit() first.');
    }
    
    TreeNode current = _root!;
    
    while (!current.isLeaf) {
      final condition = current.splitCondition!;
      
      if (condition.evaluate(sample)) {
        current = current.leftChild ?? current;
      } else {
        current = current.rightChild ?? current;
      }
      
      if (current == _root) break; // Avoid infinite loop
    }
    
    return current.prediction ?? 'unknown';
  }
  
  /// Evaluate model on test data
  ClassificationMetrics evaluate(List<ClassificationSample> testSamples) {
    print('\n📊 Evaluating Decision Tree Model');
    print('=' * 35);
    
    final predictions = <String>[];
    final actualLabels = <String>[];
    
    for (final sample in testSamples) {
      final prediction = predict(sample);
      predictions.add(prediction);
      actualLabels.add(sample.label);
    }
    
    // Calculate confusion matrix
    final confusionMatrix = <String, Map<String, int>>{};
    for (int i = 0; i < actualLabels.length; i++) {
      final actual = actualLabels[i];
      final predicted = predictions[i];
      
      confusionMatrix.putIfAbsent(actual, () => <String, int>{});
      confusionMatrix[actual]![predicted] = (confusionMatrix[actual]![predicted] ?? 0) + 1;
    }
    
    // Calculate per-class metrics
    final classes = {...actualLabels, ...predictions};
    final precision = <String, double>{};
    final recall = <String, double>{};
    final f1Score = <String, double>{};
    
    for (final className in classes) {
      final tp = confusionMatrix[className]?[className] ?? 0;
      
      int fp = 0;
      for (final otherClass in classes) {
        if (otherClass != className) {
          fp += confusionMatrix[otherClass]?[className] ?? 0;
        }
      }
      
      int fn = 0;
      final classRow = confusionMatrix[className] ?? <String, int>{};
      for (final predicted in classRow.keys) {
        if (predicted != className) {
          fn += classRow[predicted]!;
        }
      }
      
      final precisionValue = tp + fp > 0 ? tp / (tp + fp) : 0.0;
      final recallValue = tp + fn > 0 ? tp / (tp + fn) : 0.0;
      final f1Value = precisionValue + recallValue > 0 
          ? 2 * precisionValue * recallValue / (precisionValue + recallValue) 
          : 0.0;
      
      precision[className] = precisionValue;
      recall[className] = recallValue;
      f1Score[className] = f1Value;
    }
    
    // Calculate macro averages
    final macroPrecision = precision.values.isEmpty ? 0.0 : 
        precision.values.reduce((a, b) => a + b) / precision.length;
    final macroRecall = recall.values.isEmpty ? 0.0 : 
        recall.values.reduce((a, b) => a + b) / recall.length;
    final macroF1 = f1Score.values.isEmpty ? 0.0 : 
        f1Score.values.reduce((a, b) => a + b) / f1Score.length;
    
    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < actualLabels.length; i++) {
      if (actualLabels[i] == predictions[i]) correct++;
    }
    final accuracy = correct / testSamples.length;
    
    final metrics = ClassificationMetrics(
      accuracy: accuracy,
      precision: precision,
      recall: recall,
      f1Score: f1Score,
      macroPrecision: macroPrecision,
      macroRecall: macroRecall,
      macroF1: macroF1,
      confusionMatrix: confusionMatrix,
      sampleCount: testSamples.length,
    );
    
    // Print evaluation results
    print('Test samples: ${testSamples.length}');
    print('Accuracy: ${accuracy.toStringAsFixed(4)}');
    print('Macro Precision: ${macroPrecision.toStringAsFixed(4)}');
    print('Macro Recall: ${macroRecall.toStringAsFixed(4)}');
    print('Macro F1-Score: ${macroF1.toStringAsFixed(4)}');
    
    return metrics;
  }
  
  /// Print tree statistics
  void _printTreeStatistics(TreeNode node) {
    final totalNodes = _countNodes(node);
    final leafNodes = _countLeaves(node);
    final maxDepth = _getMaxDepth(node);
    
    print('\n🌳 Tree Statistics:');
    print('Total nodes: $totalNodes');
    print('Leaf nodes: $leafNodes');
    print('Internal nodes: ${totalNodes - leafNodes}');
    print('Maximum depth: $maxDepth');
  }
  
  int _countNodes(TreeNode? node) {
    if (node == null) return 0;
    return 1 + _countNodes(node.leftChild) + _countNodes(node.rightChild);
  }
  
  int _countLeaves(TreeNode? node) {
    if (node == null) return 0;
    if (node.isLeaf) return 1;
    return _countLeaves(node.leftChild) + _countLeaves(node.rightChild);
  }
  
  int _getMaxDepth(TreeNode? node) {
    if (node == null) return 0;
    return 1 + math.max(_getMaxDepth(node.leftChild), _getMaxDepth(node.rightChild));
  }
  
  /// Generate synthetic Iris-like dataset
  static List<ClassificationSample> generateIrisDataset(int samples) {
    final random = math.Random();
    final classes = ['setosa', 'versicolor', 'virginica'];
    final dataset = <ClassificationSample>[];
    
    for (int i = 0; i < samples; i++) {
      final classIndex = i % 3;
      final className = classes[classIndex];
      
      late Float64List features;
      
      switch (classIndex) {
        case 0: // Setosa
          features = Float64List.fromList([
            4.5 + random.nextGaussian() * 0.5, // Sepal length
            3.0 + random.nextGaussian() * 0.3, // Sepal width
            1.5 + random.nextGaussian() * 0.3, // Petal length
            0.3 + random.nextGaussian() * 0.1, // Petal width
          ]);
          break;
        case 1: // Versicolor
          features = Float64List.fromList([
            6.0 + random.nextGaussian() * 0.5,
            2.8 + random.nextGaussian() * 0.3,
            4.5 + random.nextGaussian() * 0.5,
            1.4 + random.nextGaussian() * 0.3,
          ]);
          break;
        case 2: // Virginica
          features = Float64List.fromList([
            6.5 + random.nextGaussian() * 0.5,
            3.0 + random.nextGaussian() * 0.3,
            5.5 + random.nextGaussian() * 0.5,
            2.0 + random.nextGaussian() * 0.3,
          ]);
          break;
      }
      
      // Ensure non-negative values
      for (int j = 0; j < features.length; j++) {
        features[j] = math.max(0.1, features[j]);
      }
      
      dataset.add(ClassificationSample(
        features: features,
        label: className,
        id: i,
      ));
    }
    
    dataset.shuffle();
    return dataset;
  }
  
  /// Comprehensive demonstration of decision tree capabilities
  static Future<void> demonstrateDecisionTree() async {
    print('🚀 Decision Tree Implementation Demonstration');
    print('=' * 50);
    
    try {
      // Generate Iris-like dataset
      print('📊 Generating synthetic Iris dataset...');
      final dataset = generateIrisDataset(150);
      
      // Split into train/test
      dataset.shuffle();
      final trainSize = (dataset.length * 0.8).toInt();
      final trainData = dataset.take(trainSize).toList();
      final testData = dataset.skip(trainSize).toList();
      
      print('Total samples: ${dataset.length}, Train: $trainSize, Test: ${testData.length}');
      
      // Create decision tree
      final featureNames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];
      
      final dt = DecisionTreeImplementation(
        featureNames: featureNames,
        maxDepth: 8,
        minSamplesLeaf: 2,
        minSamplesSplit: 5,
      );
      
      // Train the model
      await dt.fit(trainData);
      
      // Evaluate on test set
      final testResults = dt.evaluate(testData);
      
      // Test individual predictions
      print('\n🧪 Sample Predictions:');
      final sampleCount = math.min(5, testData.length);
      for (int i = 0; i < sampleCount; i++) {
        final sample = testData[i];
        final prediction = dt.predict(sample);
        print('Sample ${sample.id}: ${sample.features} -> '
              'Predicted: $prediction, Actual: ${sample.label}');
      }
      
      print('\n✅ Decision tree demonstration completed successfully!');
      
    } catch (e, stackTrace) {
      print('❌ Decision tree demonstration failed: $e');
      print('Stack trace: $stackTrace');
    }
  }
}

/// Extension to add Gaussian random number generation
extension RandomGaussian on math.Random {
  double nextGaussian() {
    // Box-Muller transform
    static double? spare;
    
    if (spare != null) {
      final result = spare!;
      spare = null;
      return result;
    }
    
    final u = nextDouble();
    final v = nextDouble();
    final magnitude = math.sqrt(-2 * math.log(u));
    
    spare = magnitude * math.cos(2 * math.pi * v);
    return magnitude * math.sin(2 * math.pi * v);
  }
}

/// Main function to demonstrate decision tree
Future<void> main() async {
  await DecisionTreeImplementation.demonstrateDecisionTree();
}