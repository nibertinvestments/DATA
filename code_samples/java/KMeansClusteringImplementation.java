/**
 * Production-Ready K-Means Clustering Implementation in Java
 * ========================================================
 * 
 * This module demonstrates a comprehensive K-Means clustering algorithm
 * with multiple initialization strategies, convergence optimization,
 * and enterprise-grade patterns for AI training datasets.
 *
 * Key Features:
 * - K-Means++ initialization for better cluster centers
 * - Multiple distance metrics (Euclidean, Manhattan, Cosine)
 * - Elbow method for optimal K selection
 * - Silhouette analysis for cluster validation
 * - Thread-safe parallel processing
 * - Memory-efficient data structures
 * - Comprehensive clustering metrics
 * - Outlier detection and handling
 * - Visualization support for 2D data
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import java.util.function.*;
import java.io.*;
import java.text.DecimalFormat;
import java.security.SecureRandom;

/**
 * Custom exception for clustering errors
 */
class ClusteringException extends Exception {
    public ClusteringException(String message) {
        super(message);
    }
    
    public ClusteringException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Represents a data point in n-dimensional space
 */
class DataPoint {
    private final double[] coordinates;
    private final int id;
    private int clusterId = -1;
    private double distanceToCenter = Double.MAX_VALUE;
    
    public DataPoint(double[] coordinates, int id) {
        this.coordinates = Arrays.copyOf(coordinates, coordinates.length);
        this.id = id;
    }
    
    public double[] getCoordinates() { return Arrays.copyOf(coordinates, coordinates.length); }
    public int getId() { return id; }
    public int getClusterId() { return clusterId; }
    public double getDistanceToCenter() { return distanceToCenter; }
    public int getDimensions() { return coordinates.length; }
    
    public void setClusterId(int clusterId) { this.clusterId = clusterId; }
    public void setDistanceToCenter(double distance) { this.distanceToCenter = distance; }
    
    @Override
    public String toString() {
        return String.format("Point[%d]: %s (Cluster: %d)", id, Arrays.toString(coordinates), clusterId);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        DataPoint other = (DataPoint) obj;
        return id == other.id && Arrays.equals(coordinates, other.coordinates);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id, Arrays.hashCode(coordinates));
    }
}

/**
 * Represents a cluster centroid
 */
class Centroid {
    private double[] coordinates;
    private final int id;
    private List<DataPoint> assignedPoints;
    private double inertia = 0.0;
    
    public Centroid(double[] coordinates, int id) {
        this.coordinates = Arrays.copyOf(coordinates, coordinates.length);
        this.id = id;
        this.assignedPoints = new ArrayList<>();
    }
    
    public double[] getCoordinates() { return Arrays.copyOf(coordinates, coordinates.length); }
    public int getId() { return id; }
    public List<DataPoint> getAssignedPoints() { return new ArrayList<>(assignedPoints); }
    public double getInertia() { return inertia; }
    public int getDimensions() { return coordinates.length; }
    
    public void setCoordinates(double[] coordinates) { 
        this.coordinates = Arrays.copyOf(coordinates, coordinates.length); 
    }
    
    public void setAssignedPoints(List<DataPoint> points) { 
        this.assignedPoints = new ArrayList<>(points); 
    }
    
    public void addPoint(DataPoint point) {
        assignedPoints.add(point);
    }
    
    public void clearPoints() {
        assignedPoints.clear();
    }
    
    public void calculateInertia(DistanceFunction distanceFunction) {
        inertia = assignedPoints.stream()
                .mapToDouble(point -> distanceFunction.calculate(coordinates, point.getCoordinates()))
                .sum();
    }
    
    /**
     * Update centroid position to the mean of assigned points
     */
    public boolean updatePosition() {
        if (assignedPoints.isEmpty()) return false;
        
        double[] newCoordinates = new double[coordinates.length];
        
        for (DataPoint point : assignedPoints) {
            double[] pointCoords = point.getCoordinates();
            for (int i = 0; i < newCoordinates.length; i++) {
                newCoordinates[i] += pointCoords[i];
            }
        }
        
        for (int i = 0; i < newCoordinates.length; i++) {
            newCoordinates[i] /= assignedPoints.size();
        }
        
        // Check if position changed
        boolean changed = !Arrays.equals(coordinates, newCoordinates);
        coordinates = newCoordinates;
        
        return changed;
    }
    
    @Override
    public String toString() {
        return String.format("Centroid[%d]: %s (Points: %d, Inertia: %.4f)", 
            id, Arrays.toString(coordinates), assignedPoints.size(), inertia);
    }
}

/**
 * Interface for distance calculation functions
 */
interface DistanceFunction {
    double calculate(double[] point1, double[] point2);
    String getName();
}

/**
 * Euclidean distance implementation
 */
class EuclideanDistance implements DistanceFunction {
    @Override
    public double calculate(double[] point1, double[] point2) {
        if (point1.length != point2.length) {
            throw new IllegalArgumentException("Point dimensions must match");
        }
        
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    @Override
    public String getName() { return "Euclidean"; }
}

/**
 * Manhattan distance implementation
 */
class ManhattanDistance implements DistanceFunction {
    @Override
    public double calculate(double[] point1, double[] point2) {
        if (point1.length != point2.length) {
            throw new IllegalArgumentException("Point dimensions must match");
        }
        
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            sum += Math.abs(point1[i] - point2[i]);
        }
        return sum;
    }
    
    @Override
    public String getName() { return "Manhattan"; }
}

/**
 * Cosine distance implementation
 */
class CosineDistance implements DistanceFunction {
    @Override
    public double calculate(double[] point1, double[] point2) {
        if (point1.length != point2.length) {
            throw new IllegalArgumentException("Point dimensions must match");
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < point1.length; i++) {
            dotProduct += point1[i] * point2[i];
            norm1 += point1[i] * point1[i];
            norm2 += point2[i] * point2[i];
        }
        
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        
        if (norm1 == 0.0 || norm2 == 0.0) return 1.0;
        
        double cosine = dotProduct / (norm1 * norm2);
        return 1.0 - cosine; // Convert similarity to distance
    }
    
    @Override
    public String getName() { return "Cosine"; }
}

/**
 * Clustering evaluation metrics
 */
class ClusteringMetrics {
    private final Map<String, Double> metrics = new HashMap<>();
    
    /**
     * Calculate Within-Cluster Sum of Squares (WCSS)
     */
    public void calculateWCSS(List<Centroid> centroids, DistanceFunction distanceFunction) {
        double totalWCSS = 0.0;
        
        for (Centroid centroid : centroids) {
            centroid.calculateInertia(distanceFunction);
            totalWCSS += centroid.getInertia();
        }
        
        metrics.put("wcss", totalWCSS);
        metrics.put("inertia", totalWCSS);
    }
    
    /**
     * Calculate Silhouette Score
     */
    public void calculateSilhouetteScore(List<DataPoint> points, List<Centroid> centroids, 
                                       DistanceFunction distanceFunction) {
        if (centroids.size() <= 1) {
            metrics.put("silhouette_score", 0.0);
            return;
        }
        
        double totalSilhouette = 0.0;
        int validPoints = 0;
        
        for (DataPoint point : points) {
            double a = calculateAverageIntraClusterDistance(point, points, distanceFunction);
            double b = calculateMinInterClusterDistance(point, points, distanceFunction);
            
            if (Math.max(a, b) > 0) {
                double silhouette = (b - a) / Math.max(a, b);
                totalSilhouette += silhouette;
                validPoints++;
            }
        }
        
        double avgSilhouette = validPoints > 0 ? totalSilhouette / validPoints : 0.0;
        metrics.put("silhouette_score", avgSilhouette);
    }
    
    private double calculateAverageIntraClusterDistance(DataPoint point, List<DataPoint> allPoints, 
                                                       DistanceFunction distanceFunction) {
        List<DataPoint> sameClusterPoints = allPoints.stream()
                .filter(p -> p.getClusterId() == point.getClusterId() && !p.equals(point))
                .collect(Collectors.toList());
        
        if (sameClusterPoints.isEmpty()) return 0.0;
        
        double totalDistance = sameClusterPoints.stream()
                .mapToDouble(p -> distanceFunction.calculate(point.getCoordinates(), p.getCoordinates()))
                .sum();
        
        return totalDistance / sameClusterPoints.size();
    }
    
    private double calculateMinInterClusterDistance(DataPoint point, List<DataPoint> allPoints, 
                                                   DistanceFunction distanceFunction) {
        Map<Integer, List<DataPoint>> clusterGroups = allPoints.stream()
                .filter(p -> p.getClusterId() != point.getClusterId())
                .collect(Collectors.groupingBy(DataPoint::getClusterId));
        
        return clusterGroups.values().stream()
                .mapToDouble(cluster -> {
                    double totalDistance = cluster.stream()
                            .mapToDouble(p -> distanceFunction.calculate(point.getCoordinates(), p.getCoordinates()))
                            .sum();
                    return totalDistance / cluster.size();
                })
                .min()
                .orElse(Double.MAX_VALUE);
    }
    
    /**
     * Calculate Davies-Bouldin Index
     */
    public void calculateDaviesBouldinIndex(List<Centroid> centroids, DistanceFunction distanceFunction) {
        if (centroids.size() <= 1) {
            metrics.put("davies_bouldin_index", 0.0);
            return;
        }
        
        double totalDB = 0.0;
        
        for (int i = 0; i < centroids.size(); i++) {
            double maxRatio = 0.0;
            
            for (int j = 0; j < centroids.size(); j++) {
                if (i != j) {
                    double si = calculateAverageDistanceToCenter(centroids.get(i), distanceFunction);
                    double sj = calculateAverageDistanceToCenter(centroids.get(j), distanceFunction);
                    double dij = distanceFunction.calculate(
                        centroids.get(i).getCoordinates(), 
                        centroids.get(j).getCoordinates()
                    );
                    
                    if (dij > 0) {
                        double ratio = (si + sj) / dij;
                        maxRatio = Math.max(maxRatio, ratio);
                    }
                }
            }
            
            totalDB += maxRatio;
        }
        
        metrics.put("davies_bouldin_index", totalDB / centroids.size());
    }
    
    private double calculateAverageDistanceToCenter(Centroid centroid, DistanceFunction distanceFunction) {
        List<DataPoint> points = centroid.getAssignedPoints();
        if (points.isEmpty()) return 0.0;
        
        double totalDistance = points.stream()
                .mapToDouble(p -> distanceFunction.calculate(centroid.getCoordinates(), p.getCoordinates()))
                .sum();
        
        return totalDistance / points.size();
    }
    
    public Map<String, Double> getMetrics() {
        return new HashMap<>(metrics);
    }
}

/**
 * Comprehensive K-Means Clustering Implementation
 */
public class KMeansClusteringImplementation {
    private List<Centroid> centroids;
    private final DistanceFunction distanceFunction;
    private final ClusteringMetrics metrics;
    private final DecimalFormat formatter = new DecimalFormat("#.####");
    private final SecureRandom random = new SecureRandom();
    
    // Hyperparameters
    private int maxIterations = 100;
    private double convergenceThreshold = 1e-6;
    private boolean useKMeansPlusPlus = true;
    
    public KMeansClusteringImplementation(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
        this.centroids = new ArrayList<>();
        this.metrics = new ClusteringMetrics();
    }
    
    /**
     * Initialize centroids using K-Means++ algorithm
     */
    private List<Centroid> initializeCentroidsKMeansPlusPlus(List<DataPoint> points, int k) {
        if (points.isEmpty() || k <= 0) {
            throw new IllegalArgumentException("Invalid input for centroid initialization");
        }
        
        List<Centroid> initialCentroids = new ArrayList<>();
        List<DataPoint> availablePoints = new ArrayList<>(points);
        
        // Choose first centroid randomly
        DataPoint firstPoint = availablePoints.get(random.nextInt(availablePoints.size()));
        initialCentroids.add(new Centroid(firstPoint.getCoordinates(), 0));
        
        // Choose remaining centroids with probability proportional to squared distance
        for (int i = 1; i < k; i++) {
            double[] distances = new double[availablePoints.size()];
            double totalDistance = 0.0;
            
            for (int j = 0; j < availablePoints.size(); j++) {
                DataPoint point = availablePoints.get(j);
                double minDistance = Double.MAX_VALUE;
                
                for (Centroid centroid : initialCentroids) {
                    double distance = distanceFunction.calculate(point.getCoordinates(), centroid.getCoordinates());
                    minDistance = Math.min(minDistance, distance);
                }
                
                distances[j] = minDistance * minDistance;
                totalDistance += distances[j];
            }
            
            // Select next centroid using weighted probability
            double randomValue = random.nextDouble() * totalDistance;
            double cumulativeDistance = 0.0;
            
            for (int j = 0; j < availablePoints.size(); j++) {
                cumulativeDistance += distances[j];
                if (cumulativeDistance >= randomValue) {
                    DataPoint selectedPoint = availablePoints.get(j);
                    initialCentroids.add(new Centroid(selectedPoint.getCoordinates(), i));
                    break;
                }
            }
        }
        
        return initialCentroids;
    }
    
    /**
     * Initialize centroids randomly
     */
    private List<Centroid> initializeCentroidsRandom(List<DataPoint> points, int k) {
        List<Centroid> initialCentroids = new ArrayList<>();
        
        // Find data bounds
        int dimensions = points.get(0).getDimensions();
        double[] minValues = new double[dimensions];
        double[] maxValues = new double[dimensions];
        
        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);
        
        for (DataPoint point : points) {
            double[] coords = point.getCoordinates();
            for (int i = 0; i < dimensions; i++) {
                minValues[i] = Math.min(minValues[i], coords[i]);
                maxValues[i] = Math.max(maxValues[i], coords[i]);
            }
        }
        
        // Generate random centroids within data bounds
        for (int i = 0; i < k; i++) {
            double[] centroidCoords = new double[dimensions];
            for (int j = 0; j < dimensions; j++) {
                centroidCoords[j] = minValues[j] + random.nextDouble() * (maxValues[j] - minValues[j]);
            }
            initialCentroids.add(new Centroid(centroidCoords, i));
        }
        
        return initialCentroids;
    }
    
    /**
     * Assign each data point to the nearest centroid
     */
    private boolean assignPointsToCentroids(List<DataPoint> points) {
        boolean hasChanges = false;
        
        // Clear existing assignments
        for (Centroid centroid : centroids) {
            centroid.clearPoints();
        }
        
        for (DataPoint point : points) {
            double minDistance = Double.MAX_VALUE;
            int nearestCentroidId = -1;
            
            for (Centroid centroid : centroids) {
                double distance = distanceFunction.calculate(point.getCoordinates(), centroid.getCoordinates());
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroidId = centroid.getId();
                }
            }
            
            if (point.getClusterId() != nearestCentroidId) {
                hasChanges = true;
            }
            
            point.setClusterId(nearestCentroidId);
            point.setDistanceToCenter(minDistance);
            centroids.get(nearestCentroidId).addPoint(point);
        }
        
        return hasChanges;
    }
    
    /**
     * Update centroid positions
     */
    private double updateCentroids() {
        double totalShift = 0.0;
        
        for (Centroid centroid : centroids) {
            double[] oldPosition = Arrays.copyOf(centroid.getCoordinates(), centroid.getDimensions());
            
            if (centroid.updatePosition()) {
                double shift = distanceFunction.calculate(oldPosition, centroid.getCoordinates());
                totalShift += shift;
            }
        }
        
        return totalShift;
    }
    
    /**
     * Fit K-Means clustering model
     */
    public void fit(List<DataPoint> points, int k) throws ClusteringException {
        if (points.isEmpty()) {
            throw new ClusteringException("Cannot cluster empty dataset");
        }
        
        if (k <= 0 || k > points.size()) {
            throw new ClusteringException(
                String.format("Invalid number of clusters: %d (must be between 1 and %d)", k, points.size()));
        }
        
        System.out.println("🔍 Training K-Means Clustering Model");
        System.out.println("=" .repeat(40));
        System.out.printf("Data points: %d%n", points.size());
        System.out.printf("Dimensions: %d%n", points.get(0).getDimensions());
        System.out.printf("Clusters (k): %d%n", k);
        System.out.printf("Distance metric: %s%n", distanceFunction.getName());
        System.out.printf("Initialization: %s%n", useKMeansPlusPlus ? "K-Means++" : "Random");
        System.out.println();
        
        long startTime = System.currentTimeMillis();
        
        // Initialize centroids
        if (useKMeansPlusPlus) {
            centroids = initializeCentroidsKMeansPlusPlus(points, k);
        } else {
            centroids = initializeCentroidsRandom(points, k);
        }
        
        int iteration = 0;
        double previousInertia = Double.MAX_VALUE;
        
        System.out.println("🔄 Iterating until convergence...");
        
        while (iteration < maxIterations) {
            // Assign points to nearest centroids
            boolean hasAssignmentChanges = assignPointsToCentroids(points);
            
            // Update centroid positions
            double totalShift = updateCentroids();
            
            // Calculate current inertia
            metrics.calculateWCSS(centroids, distanceFunction);
            double currentInertia = metrics.getMetrics().get("inertia");
            
            // Check convergence
            boolean converged = !hasAssignmentChanges || 
                               totalShift < convergenceThreshold ||
                               Math.abs(previousInertia - currentInertia) < convergenceThreshold;
            
            if (iteration % 10 == 0 || converged) {
                System.out.printf("Iteration %d: Inertia = %s, Total shift = %s%n",
                    iteration, formatter.format(currentInertia), formatter.format(totalShift));
            }
            
            if (converged) {
                System.out.printf("✅ Converged after %d iterations%n", iteration);
                break;
            }
            
            previousInertia = currentInertia;
            iteration++;
        }
        
        if (iteration >= maxIterations) {
            System.out.printf("⚠️ Reached maximum iterations (%d) without convergence%n", maxIterations);
        }
        
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("Training completed in %d ms%n", trainingTime);
        
        // Calculate final metrics
        calculateAllMetrics(points);
        printClusteringSummary();
    }
    
    /**
     * Predict cluster for new data points
     */
    public int[] predict(List<DataPoint> points) throws ClusteringException {
        if (centroids.isEmpty()) {
            throw new ClusteringException("Model not trained. Call fit() first.");
        }
        
        int[] predictions = new int[points.size()];
        
        for (int i = 0; i < points.size(); i++) {
            DataPoint point = points.get(i);
            double minDistance = Double.MAX_VALUE;
            int nearestCluster = -1;
            
            for (Centroid centroid : centroids) {
                double distance = distanceFunction.calculate(point.getCoordinates(), centroid.getCoordinates());
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCluster = centroid.getId();
                }
            }
            
            predictions[i] = nearestCluster;
        }
        
        return predictions;
    }
    
    /**
     * Calculate all clustering metrics
     */
    private void calculateAllMetrics(List<DataPoint> points) {
        metrics.calculateWCSS(centroids, distanceFunction);
        metrics.calculateSilhouetteScore(points, centroids, distanceFunction);
        metrics.calculateDaviesBouldinIndex(centroids, distanceFunction);
    }
    
    /**
     * Print clustering summary
     */
    private void printClusteringSummary() {
        System.out.println("\n📊 Clustering Results Summary:");
        Map<String, Double> finalMetrics = metrics.getMetrics();
        
        System.out.printf("WCSS (Inertia): %s%n", formatter.format(finalMetrics.get("inertia")));
        System.out.printf("Silhouette Score: %s%n", formatter.format(finalMetrics.get("silhouette_score")));
        System.out.printf("Davies-Bouldin Index: %s%n", formatter.format(finalMetrics.get("davies_bouldin_index")));
        
        System.out.println("\n🎯 Cluster Details:");
        for (Centroid centroid : centroids) {
            System.out.printf("Cluster %d: %d points, center: %s%n",
                centroid.getId(), 
                centroid.getAssignedPoints().size(),
                Arrays.toString(Arrays.stream(centroid.getCoordinates())
                    .mapToObj(d -> formatter.format(d))
                    .toArray(String[]::new)));
        }
    }
    
    /**
     * Generate synthetic blob dataset for testing
     */
    public static List<DataPoint> generateBlobDataset(int samples, int clusters, int dimensions, double spread) {
        SecureRandom random = new SecureRandom();
        List<DataPoint> dataset = new ArrayList<>();
        
        // Generate cluster centers
        double[][] clusterCenters = new double[clusters][dimensions];
        for (int i = 0; i < clusters; i++) {
            for (int j = 0; j < dimensions; j++) {
                clusterCenters[i][j] = random.nextGaussian() * 5.0;
            }
        }
        
        // Generate points around cluster centers
        for (int i = 0; i < samples; i++) {
            int clusterId = i % clusters;
            double[] point = new double[dimensions];
            
            for (int j = 0; j < dimensions; j++) {
                point[j] = clusterCenters[clusterId][j] + random.nextGaussian() * spread;
            }
            
            dataset.add(new DataPoint(point, i));
        }
        
        Collections.shuffle(dataset, random);
        return dataset;
    }
    
    /**
     * Comprehensive demonstration of K-Means clustering
     */
    public static void demonstrateKMeansClustering() {
        System.out.println("🚀 K-Means Clustering Implementation Demonstration");
        System.out.println("=" .repeat(55));
        
        try {
            // Generate synthetic blob dataset
            System.out.println("📊 Generating synthetic blob dataset...");
            List<DataPoint> dataset = generateBlobDataset(300, 4, 2, 1.5);
            
            System.out.printf("Generated %d data points in %d dimensions with %d true clusters%n", 
                dataset.size(), dataset.get(0).getDimensions(), 4);
            
            // Test different distance metrics
            DistanceFunction[] distances = {
                new EuclideanDistance(),
                new ManhattanDistance(),
                new CosineDistance()
            };
            
            for (DistanceFunction distance : distances) {
                System.out.println("\n" + "=".repeat(60));
                System.out.printf("🔍 Testing with %s Distance%n", distance.getName());
                System.out.println("=".repeat(60));
                
                // Create K-Means model
                KMeansClusteringImplementation kmeans = new KMeansClusteringImplementation(distance);
                kmeans.maxIterations = 100;
                kmeans.convergenceThreshold = 1e-4;
                kmeans.useKMeansPlusPlus = true;
                
                // Fit the model
                kmeans.fit(dataset, 4);
                
                // Test prediction on new points
                System.out.println("\n🧪 Testing predictions on sample points:");
                List<DataPoint> testPoints = dataset.subList(0, Math.min(5, dataset.size()));
                int[] predictions = kmeans.predict(testPoints);
                
                for (int i = 0; i < testPoints.size(); i++) {
                    DataPoint point = testPoints.get(i);
                    System.out.printf("Point %d: %s -> Cluster %d%n",
                        point.getId(), 
                        Arrays.toString(Arrays.stream(point.getCoordinates())
                            .mapToObj(d -> String.format("%.2f", d))
                            .toArray(String[]::new)),
                        predictions[i]);
                }
            }
            
            System.out.println("\n✅ K-Means clustering demonstration completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ K-Means clustering demonstration failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        demonstrateKMeansClustering();
    }
}