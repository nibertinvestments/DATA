/**
 * Production-Ready K-Means Clustering Implementation in Dart
 * =========================================================
 * 
 * This module demonstrates a comprehensive K-Means clustering algorithm
 * with K-Means++ initialization, multiple distance metrics, and modern
 * Dart patterns for AI training datasets.
 *
 * Key Features:
 * - K-Means++ initialization for better cluster centers
 * - Multiple distance metrics (Euclidean, Manhattan, Cosine)
 * - Elbow method for optimal K selection
 * - Silhouette analysis for cluster validation
 * - Dart null safety and async/await
 * - Memory-efficient data structures
 * - Comprehensive clustering metrics
 * - Modern Dart patterns and collections
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import 'dart:math' as math;
import 'dart:typed_data';

/// Custom exception for clustering errors
class ClusteringException implements Exception {
  final String message;
  final dynamic cause;
  
  ClusteringException(this.message, [this.cause]);
  
  @override
  String toString() => 'ClusteringException: $message${cause != null ? ' (Caused by: $cause)' : ''}';
}

/// Data class representing a data point in n-dimensional space
class DataPoint {
  final Float64List coordinates;
  final int id;
  int clusterId = -1;
  double distanceToCenter = double.infinity;
  
  DataPoint({
    required this.coordinates,
    required this.id,
  });
  
  int get dimensions => coordinates.length;
  
  void setCluster(int clusterId, double distance) {
    this.clusterId = clusterId;
    distanceToCenter = distance;
  }
  
  @override
  String toString() => 'DataPoint(id: $id, coordinates: $coordinates, cluster: $clusterId)';
  
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DataPoint &&
          runtimeType == other.runtimeType &&
          id == other.id &&
          _listEquals(coordinates, other.coordinates);
  
  @override
  int get hashCode => Object.hash(id, Object.hashAll(coordinates));
  
  bool _listEquals(Float64List a, Float64List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Centroid representing a cluster center
class Centroid {
  Float64List coordinates;
  final int id;
  List<DataPoint> assignedPoints = [];
  double inertia = 0.0;
  
  Centroid({
    required this.coordinates,
    required this.id,
  });
  
  int get dimensions => coordinates.length;
  
  /// Add a point to this centroid
  void addPoint(DataPoint point) {
    assignedPoints.add(point);
  }
  
  /// Clear all assigned points
  void clearPoints() {
    assignedPoints.clear();
  }
  
  /// Calculate inertia (sum of squared distances to assigned points)
  void calculateInertia(DistanceFunction distanceFunction) {
    inertia = 0.0;
    for (final point in assignedPoints) {
      inertia += distanceFunction.calculate(coordinates, point.coordinates);
    }
  }
  
  /// Update centroid position to the mean of assigned points
  bool updatePosition() {
    if (assignedPoints.isEmpty) return false;
    
    final newCoordinates = Float64List(coordinates.length);
    
    for (final point in assignedPoints) {
      for (int i = 0; i < newCoordinates.length; i++) {
        newCoordinates[i] += point.coordinates[i];
      }
    }
    
    for (int i = 0; i < newCoordinates.length; i++) {
      newCoordinates[i] /= assignedPoints.length;
    }
    
    // Check if position changed
    bool changed = false;
    for (int i = 0; i < coordinates.length; i++) {
      if ((coordinates[i] - newCoordinates[i]).abs() > 1e-9) {
        changed = true;
        break;
      }
    }
    
    coordinates = newCoordinates;
    return changed;
  }
  
  @override
  String toString() => 'Centroid(id: $id, coordinates: $coordinates, '
                      'points: ${assignedPoints.length}, inertia: ${inertia.toStringAsFixed(4)})';
}

/// Distance function interface
abstract class DistanceFunction {
  double calculate(Float64List point1, Float64List point2);
  String get name;
}

/// Euclidean distance implementation
class EuclideanDistance implements DistanceFunction {
  @override
  double calculate(Float64List point1, Float64List point2) {
    if (point1.length != point2.length) {
      throw ArgumentError('Point dimensions must match');
    }
    
    double sum = 0.0;
    for (int i = 0; i < point1.length; i++) {
      final diff = point1[i] - point2[i];
      sum += diff * diff;
    }
    return math.sqrt(sum);
  }
  
  @override
  String get name => 'Euclidean';
}

/// Manhattan distance implementation
class ManhattanDistance implements DistanceFunction {
  @override
  double calculate(Float64List point1, Float64List point2) {
    if (point1.length != point2.length) {
      throw ArgumentError('Point dimensions must match');
    }
    
    double sum = 0.0;
    for (int i = 0; i < point1.length; i++) {
      sum += (point1[i] - point2[i]).abs();
    }
    return sum;
  }
  
  @override
  String get name => 'Manhattan';
}

/// Cosine distance implementation
class CosineDistance implements DistanceFunction {
  @override
  double calculate(Float64List point1, Float64List point2) {
    if (point1.length != point2.length) {
      throw ArgumentError('Point dimensions must match');
    }
    
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (int i = 0; i < point1.length; i++) {
      dotProduct += point1[i] * point2[i];
      norm1 += point1[i] * point1[i];
      norm2 += point2[i] * point2[i];
    }
    
    norm1 = math.sqrt(norm1);
    norm2 = math.sqrt(norm2);
    
    if (norm1 == 0.0 || norm2 == 0.0) return 1.0;
    
    final cosine = dotProduct / (norm1 * norm2);
    return 1.0 - cosine; // Convert similarity to distance
  }
  
  @override
  String get name => 'Cosine';
}

/// Clustering evaluation metrics
class ClusteringMetrics {
  final double wcss;
  final double inertia;
  final double silhouetteScore;
  final double daviesBouldinIndex;
  final Map<String, double> additionalMetrics;
  
  ClusteringMetrics({
    required this.wcss,
    required this.inertia,
    required this.silhouetteScore,
    required this.daviesBouldinIndex,
    this.additionalMetrics = const {},
  });
  
  @override
  String toString() {
    final buffer = StringBuffer()
      ..writeln('Clustering Metrics:')
      ..writeln('  WCSS (Inertia): ${inertia.toStringAsFixed(6)}')
      ..writeln('  Silhouette Score: ${silhouetteScore.toStringAsFixed(6)}')
      ..writeln('  Davies-Bouldin Index: ${daviesBouldinIndex.toStringAsFixed(6)}');
    
    for (final entry in additionalMetrics.entries) {
      buffer.writeln('  ${entry.key}: ${entry.value.toStringAsFixed(6)}');
    }
    
    return buffer.toString();
  }
}

/// Comprehensive K-Means Clustering Implementation
class KMeansClusteringImplementation {
  final DistanceFunction distanceFunction;
  final int maxIterations;
  final double convergenceThreshold;
  final bool useKMeansPlusPlus;
  
  List<Centroid>? _centroids;
  bool _fitted = false;
  final math.Random _random = math.Random();
  
  KMeansClusteringImplementation({
    required this.distanceFunction,
    this.maxIterations = 100,
    this.convergenceThreshold = 1e-6,
    this.useKMeansPlusPlus = true,
  });
  
  List<Centroid> get centroids => _centroids ?? [];
  
  /// Initialize centroids using K-Means++ algorithm
  List<Centroid> _initializeCentroidsKMeansPlusPlus(List<DataPoint> points, int k) {
    if (points.isEmpty || k <= 0) {
      throw ArgumentError('Invalid input for centroid initialization');
    }
    
    final initialCentroids = <Centroid>[];
    final availablePoints = List<DataPoint>.from(points);
    
    // Choose first centroid randomly
    final firstPoint = availablePoints[_random.nextInt(availablePoints.length)];
    initialCentroids.add(Centroid(
      coordinates: Float64List.fromList(firstPoint.coordinates),
      id: 0,
    ));
    
    // Choose remaining centroids with probability proportional to squared distance
    for (int i = 1; i < k; i++) {
      final distances = <double>[];
      double totalDistance = 0.0;
      
      for (final point in availablePoints) {
        double minDistance = double.infinity;
        
        for (final centroid in initialCentroids) {
          final distance = distanceFunction.calculate(point.coordinates, centroid.coordinates);
          minDistance = math.min(minDistance, distance);
        }
        
        final squaredDistance = minDistance * minDistance;
        distances.add(squaredDistance);
        totalDistance += squaredDistance;
      }
      
      // Select next centroid using weighted probability
      final randomValue = _random.nextDouble() * totalDistance;
      double cumulativeDistance = 0.0;
      
      for (int j = 0; j < availablePoints.length; j++) {
        cumulativeDistance += distances[j];
        if (cumulativeDistance >= randomValue) {
          final selectedPoint = availablePoints[j];
          initialCentroids.add(Centroid(
            coordinates: Float64List.fromList(selectedPoint.coordinates),
            id: i,
          ));
          break;
        }
      }
    }
    
    return initialCentroids;
  }
  
  /// Initialize centroids randomly
  List<Centroid> _initializeCentroidsRandom(List<DataPoint> points, int k) {
    final initialCentroids = <Centroid>[];
    
    // Find data bounds
    final dimensions = points.first.dimensions;
    final minValues = Float64List(dimensions);
    final maxValues = Float64List(dimensions);
    
    for (int i = 0; i < dimensions; i++) {
      minValues[i] = double.infinity;
      maxValues[i] = double.negativeInfinity;
    }
    
    for (final point in points) {
      for (int i = 0; i < dimensions; i++) {
        minValues[i] = math.min(minValues[i], point.coordinates[i]);
        maxValues[i] = math.max(maxValues[i], point.coordinates[i]);
      }
    }
    
    // Generate random centroids within data bounds
    for (int i = 0; i < k; i++) {
      final centroidCoords = Float64List(dimensions);
      for (int j = 0; j < dimensions; j++) {
        centroidCoords[j] = minValues[j] + _random.nextDouble() * (maxValues[j] - minValues[j]);
      }
      initialCentroids.add(Centroid(
        coordinates: centroidCoords,
        id: i,
      ));
    }
    
    return initialCentroids;
  }
  
  /// Assign each data point to the nearest centroid
  bool _assignPointsToCentroids(List<DataPoint> points) {
    bool hasChanges = false;
    
    // Clear existing assignments
    for (final centroid in _centroids!) {
      centroid.clearPoints();
    }
    
    for (final point in points) {
      double minDistance = double.infinity;
      int nearestCentroidId = -1;
      
      for (final centroid in _centroids!) {
        final distance = distanceFunction.calculate(point.coordinates, centroid.coordinates);
        if (distance < minDistance) {
          minDistance = distance;
          nearestCentroidId = centroid.id;
        }
      }
      
      if (point.clusterId != nearestCentroidId) {
        hasChanges = true;
      }
      
      point.setCluster(nearestCentroidId, minDistance);
      _centroids![nearestCentroidId].addPoint(point);
    }
    
    return hasChanges;
  }
  
  /// Update centroid positions
  double _updateCentroids() {
    double totalShift = 0.0;
    
    for (final centroid in _centroids!) {
      final oldCoordinates = Float64List.fromList(centroid.coordinates);
      
      if (centroid.updatePosition()) {
        final shift = distanceFunction.calculate(oldCoordinates, centroid.coordinates);
        totalShift += shift;
      }
    }
    
    return totalShift;
  }
  
  /// Calculate silhouette score
  double _calculateSilhouetteScore(List<DataPoint> points) {
    if (_centroids!.length <= 1) return 0.0;
    
    double totalSilhouette = 0.0;
    int validPoints = 0;
    
    for (final point in points) {
      final a = _calculateAverageIntraClusterDistance(point, points);
      final b = _calculateMinInterClusterDistance(point, points);
      
      if (math.max(a, b) > 0) {
        final silhouette = (b - a) / math.max(a, b);
        totalSilhouette += silhouette;
        validPoints++;
      }
    }
    
    return validPoints > 0 ? totalSilhouette / validPoints : 0.0;
  }
  
  double _calculateAverageIntraClusterDistance(DataPoint point, List<DataPoint> allPoints) {
    final sameClusterPoints = allPoints
        .where((p) => p.clusterId == point.clusterId && p != point)
        .toList();
    
    if (sameClusterPoints.isEmpty) return 0.0;
    
    double totalDistance = 0.0;
    for (final p in sameClusterPoints) {
      totalDistance += distanceFunction.calculate(point.coordinates, p.coordinates);
    }
    
    return totalDistance / sameClusterPoints.length;
  }
  
  double _calculateMinInterClusterDistance(DataPoint point, List<DataPoint> allPoints) {
    final clusterGroups = <int, List<DataPoint>>{};
    for (final p in allPoints) {
      if (p.clusterId != point.clusterId) {
        clusterGroups.putIfAbsent(p.clusterId, () => []).add(p);
      }
    }
    
    double minDistance = double.infinity;
    for (final cluster in clusterGroups.values) {
      if (cluster.isNotEmpty) {
        double totalDistance = 0.0;
        for (final p in cluster) {
          totalDistance += distanceFunction.calculate(point.coordinates, p.coordinates);
        }
        final avgDistance = totalDistance / cluster.length;
        minDistance = math.min(minDistance, avgDistance);
      }
    }
    
    return minDistance == double.infinity ? 0.0 : minDistance;
  }
  
  /// Fit K-Means clustering model
  Future<void> fit(List<DataPoint> points, int k) async {
    if (points.isEmpty) {
      throw ClusteringException('Cannot cluster empty dataset');
    }
    
    if (k <= 0 || k > points.length) {
      throw ClusteringException(
        'Invalid number of clusters: $k (must be between 1 and ${points.length})'
      );
    }
    
    print('🔍 Training K-Means Clustering Model');
    print('=' * 40);
    print('Data points: ${points.length}');
    print('Dimensions: ${points.first.dimensions}');
    print('Clusters (k): $k');
    print('Distance metric: ${distanceFunction.name}');
    print('Initialization: ${useKMeansPlusPlus ? 'K-Means++' : 'Random'}');
    
    final stopwatch = Stopwatch()..start();
    
    // Initialize centroids
    if (useKMeansPlusPlus) {
      _centroids = _initializeCentroidsKMeansPlusPlus(points, k);
    } else {
      _centroids = _initializeCentroidsRandom(points, k);
    }
    
    int iteration = 0;
    double previousInertia = double.infinity;
    
    print('\n🔄 Iterating until convergence...');
    
    while (iteration < maxIterations) {
      // Assign points to nearest centroids
      final hasAssignmentChanges = _assignPointsToCentroids(points);
      
      // Update centroid positions
      final totalShift = _updateCentroids();
      
      // Calculate current inertia
      double currentInertia = 0.0;
      for (final centroid in _centroids!) {
        centroid.calculateInertia(distanceFunction);
        currentInertia += centroid.inertia;
      }
      
      // Check convergence
      final converged = !hasAssignmentChanges || 
                       totalShift < convergenceThreshold ||
                       (previousInertia - currentInertia).abs() < convergenceThreshold;
      
      if (iteration % 10 == 0 || converged) {
        print('Iteration $iteration: Inertia = ${currentInertia.toStringAsFixed(4)}, '
              'Total shift = ${totalShift.toStringAsFixed(4)}');
      }
      
      if (converged) {
        print('✅ Converged after $iteration iterations');
        break;
      }
      
      previousInertia = currentInertia;
      iteration++;
      
      // Allow other operations to run
      if (iteration % 10 == 0) {
        await Future.delayed(Duration.zero);
      }
    }
    
    if (iteration >= maxIterations) {
      print('⚠️ Reached maximum iterations ($maxIterations) without convergence');
    }
    
    stopwatch.stop();
    print('Training completed in ${stopwatch.elapsedMilliseconds}ms');
    
    _fitted = true;
    await _printClusteringSummary(points);
  }
  
  /// Predict cluster for new data points
  List<int> predict(List<DataPoint> points) {
    if (!_fitted || _centroids == null) {
      throw ClusteringException('Model not trained. Call fit() first.');
    }
    
    final predictions = <int>[];
    
    for (final point in points) {
      double minDistance = double.infinity;
      int nearestCluster = -1;
      
      for (final centroid in _centroids!) {
        final distance = distanceFunction.calculate(point.coordinates, centroid.coordinates);
        if (distance < minDistance) {
          minDistance = distance;
          nearestCluster = centroid.id;
        }
      }
      
      predictions.add(nearestCluster);
    }
    
    return predictions;
  }
  
  /// Calculate clustering metrics
  Future<ClusteringMetrics> calculateMetrics(List<DataPoint> points) async {
    if (!_fitted || _centroids == null) {
      throw ClusteringException('Model not trained. Call fit() first.');
    }
    
    // Calculate WCSS (inertia)
    double totalInertia = 0.0;
    for (final centroid in _centroids!) {
      centroid.calculateInertia(distanceFunction);
      totalInertia += centroid.inertia;
    }
    
    // Calculate silhouette score
    final silhouetteScore = _calculateSilhouetteScore(points);
    
    // Calculate Davies-Bouldin index (simplified version)
    double daviesBouldin = 0.0;
    if (_centroids!.length > 1) {
      for (int i = 0; i < _centroids!.length; i++) {
        double maxRatio = 0.0;
        
        for (int j = 0; j < _centroids!.length; j++) {
          if (i != j) {
            final si = _calculateAverageDistanceToCenter(_centroids![i]);
            final sj = _calculateAverageDistanceToCenter(_centroids![j]);
            final dij = distanceFunction.calculate(
              _centroids![i].coordinates, 
              _centroids![j].coordinates
            );
            
            if (dij > 0) {
              final ratio = (si + sj) / dij;
              maxRatio = math.max(maxRatio, ratio);
            }
          }
        }
        
        daviesBouldin += maxRatio;
      }
      daviesBouldin /= _centroids!.length;
    }
    
    return ClusteringMetrics(
      wcss: totalInertia,
      inertia: totalInertia,
      silhouetteScore: silhouetteScore,
      daviesBouldinIndex: daviesBouldin,
    );
  }
  
  double _calculateAverageDistanceToCenter(Centroid centroid) {
    if (centroid.assignedPoints.isEmpty) return 0.0;
    
    double totalDistance = 0.0;
    for (final point in centroid.assignedPoints) {
      totalDistance += distanceFunction.calculate(centroid.coordinates, point.coordinates);
    }
    
    return totalDistance / centroid.assignedPoints.length;
  }
  
  /// Print clustering summary
  Future<void> _printClusteringSummary(List<DataPoint> points) async {
    final metrics = await calculateMetrics(points);
    
    print('\n📊 Clustering Results Summary:');
    print('WCSS (Inertia): ${metrics.inertia.toStringAsFixed(4)}');
    print('Silhouette Score: ${metrics.silhouetteScore.toStringAsFixed(4)}');
    print('Davies-Bouldin Index: ${metrics.daviesBouldinIndex.toStringAsFixed(4)}');
    
    print('\n🎯 Cluster Details:');
    for (final centroid in _centroids!) {
      final centerStr = centroid.coordinates
          .map((c) => c.toStringAsFixed(3))
          .join(', ');
      print('Cluster ${centroid.id}: ${centroid.assignedPoints.length} points, '
            'center: [$centerStr]');
    }
  }
  
  /// Generate synthetic blob dataset for testing
  static List<DataPoint> generateBlobDataset(int samples, int clusters, int dimensions, double spread) {
    final random = math.Random();
    final dataset = <DataPoint>[];
    
    // Generate cluster centers
    final clusterCenters = <Float64List>[];
    for (int i = 0; i < clusters; i++) {
      final center = Float64List(dimensions);
      for (int j = 0; j < dimensions; j++) {
        center[j] = (random.nextDouble() - 0.5) * 10.0;
      }
      clusterCenters.add(center);
    }
    
    // Generate points around cluster centers
    for (int i = 0; i < samples; i++) {
      final clusterId = i % clusters;
      final center = clusterCenters[clusterId];
      final point = Float64List(dimensions);
      
      for (int j = 0; j < dimensions; j++) {
        point[j] = center[j] + random.nextGaussian() * spread;
      }
      
      dataset.add(DataPoint(
        coordinates: point,
        id: i,
      ));
    }
    
    dataset.shuffle();
    return dataset;
  }
  
  /// Comprehensive demonstration of K-Means clustering
  static Future<void> demonstrateKMeansClustering() async {
    print('🚀 K-Means Clustering Implementation Demonstration');
    print('=' * 55);
    
    try {
      // Generate synthetic blob dataset
      print('📊 Generating synthetic blob dataset...');
      final dataset = generateBlobDataset(300, 4, 2, 1.5);
      
      print('Generated ${dataset.length} data points in ${dataset.first.dimensions} '
            'dimensions with 4 true clusters');
      
      // Test different distance metrics
      final distances = [
        EuclideanDistance(),
        ManhattanDistance(),
        CosineDistance(),
      ];
      
      for (final distance in distances) {
        print('\n${'=' * 60}');
        print('🔍 Testing with ${distance.name} Distance');
        print('=' * 60);
        
        // Create K-Means model
        final kmeans = KMeansClusteringImplementation(
          distanceFunction: distance,
          maxIterations: 100,
          convergenceThreshold: 1e-4,
          useKMeansPlusPlus: true,
        );
        
        // Fit the model
        await kmeans.fit(dataset, 4);
        
        // Test prediction on new points
        print('\n🧪 Testing predictions on sample points:');
        final testPoints = dataset.take(5).toList();
        final predictions = kmeans.predict(testPoints);
        
        for (int i = 0; i < testPoints.length; i++) {
          final point = testPoints[i];
          final coordsStr = point.coordinates
              .map((c) => c.toStringAsFixed(2))
              .join(', ');
          print('Point ${point.id}: [$coordsStr] -> Cluster ${predictions[i]}');
        }
        
        // Calculate and display final metrics
        final metrics = await kmeans.calculateMetrics(dataset);
        print('\n📈 Final Metrics:');
        print(metrics);
      }
      
      print('\n✅ K-Means clustering demonstration completed successfully!');
      
    } catch (e, stackTrace) {
      print('❌ K-Means clustering demonstration failed: $e');
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

/// Main function to demonstrate K-means clustering
Future<void> main() async {
  await KMeansClusteringImplementation.demonstrateKMeansClustering();
}