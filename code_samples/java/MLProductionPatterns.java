/**
 * Production-Ready Machine Learning Patterns in Java
 * ==================================================
 * 
 * This module demonstrates industry-standard ML patterns in Java with proper
 * error handling, validation, type safety, and production deployment
 * considerations for AI training datasets.
 *
 * Key Features:
 * - Strong type safety with generics
 * - Comprehensive error handling and validation  
 * - Thread-safe operations where applicable
 * - Memory-efficient data processing
 * - Integration with Java ML ecosystem
 * - Extensive documentation for AI learning
 * - Enterprise-ready patterns
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import java.util.function.*;
import java.io.*;
import java.time.Instant;
import java.time.Duration;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.regex.Pattern;

/**
 * Custom exception for data validation errors
 */
class DataValidationException extends Exception {
    public DataValidationException(String message) {
        super(message);
    }
    
    public DataValidationException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Custom exception for model training errors
 */
class ModelTrainingException extends Exception {
    public ModelTrainingException(String message) {
        super(message);
    }
    
    public ModelTrainingException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Custom exception for prediction errors
 */
class PredictionException extends Exception {
    public PredictionException(String message) {
        super(message);
    }
    
    public PredictionException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Data class representing a training data point
 */
class DataPoint {
    private final double[] features;
    private final double target;
    private final String id;
    
    public DataPoint(double[] features, double target) {
        this(features, target, UUID.randomUUID().toString());
    }
    
    public DataPoint(double[] features, double target, String id) {
        this.features = Arrays.copyOf(features, features.length);
        this.target = target;
        this.id = id;
    }
    
    public double[] getFeatures() {
        return Arrays.copyOf(features, features.length);
    }
    
    public double getTarget() {
        return target;
    }
    
    public String getId() {
        return id;
    }
    
    public int getFeatureCount() {
        return features.length;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof DataPoint)) return false;
        DataPoint other = (DataPoint) obj;
        return Arrays.equals(features, other.features) && 
               Double.compare(target, other.target) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(features), target);
    }
    
    @Override
    public String toString() {
        return String.format("DataPoint{features=%s, target=%.4f, id='%s'}", 
                           Arrays.toString(features), target, id);
    }
}

/**
 * Data class for model performance metrics
 */
class ModelMetrics {
    private final double accuracy;
    private final double rmse;
    private final double mse;
    private final double rSquared;
    private final long trainingTimeMs;
    private final long predictionTimeMs;
    private final long modelSizeBytes;
    private final Instant timestamp;
    
    public ModelMetrics(double accuracy, double rmse, double mse, double rSquared,
                       long trainingTimeMs, long predictionTimeMs, long modelSizeBytes) {
        this.accuracy = accuracy;
        this.rmse = rmse;
        this.mse = mse;
        this.rSquared = rSquared;
        this.trainingTimeMs = trainingTimeMs;
        this.predictionTimeMs = predictionTimeMs;
        this.modelSizeBytes = modelSizeBytes;
        this.timestamp = Instant.now();
    }
    
    // Getters
    public double getAccuracy() { return accuracy; }
    public double getRmse() { return rmse; }
    public double getMse() { return mse; }
    public double getRSquared() { return rSquared; }
    public long getTrainingTimeMs() { return trainingTimeMs; }
    public long getPredictionTimeMs() { return predictionTimeMs; }
    public long getModelSizeBytes() { return modelSizeBytes; }
    public Instant getTimestamp() { return timestamp; }
    
    @Override
    public String toString() {
        return String.format("ModelMetrics{accuracy=%.4f, rmse=%.4f, rSquared=%.4f, " +
                           "trainingTime=%dms, predictionTime=%dms, modelSize=%d bytes}", 
                           accuracy, rmse, rSquared, trainingTimeMs, predictionTimeMs, modelSizeBytes);
    }
}

/**
 * Data class for validation results
 */
class ValidationResult {
    private final boolean isValid;
    private final List<String> errors;
    private final List<String> warnings;
    private final int rowCount;
    private final int columnCount;
    private final Map<String, Integer> missingValues;
    
    public ValidationResult(boolean isValid, List<String> errors, List<String> warnings,
                          int rowCount, int columnCount, Map<String, Integer> missingValues) {
        this.isValid = isValid;
        this.errors = new ArrayList<>(errors);
        this.warnings = new ArrayList<>(warnings);
        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this.missingValues = new HashMap<>(missingValues);
    }
    
    // Getters
    public boolean isValid() { return isValid; }
    public List<String> getErrors() { return new ArrayList<>(errors); }
    public List<String> getWarnings() { return new ArrayList<>(warnings); }
    public int getRowCount() { return rowCount; }
    public int getColumnCount() { return columnCount; }
    public Map<String, Integer> getMissingValues() { return new HashMap<>(missingValues); }
    
    @Override
    public String toString() {
        return String.format("ValidationResult{isValid=%b, errors=%d, warnings=%d, " +
                           "rows=%d, columns=%d}", 
                           isValid, errors.size(), warnings.size(), rowCount, columnCount);
    }
}

/**
 * Comprehensive data validator for Java ML pipelines
 */
class DataValidator {
    private final int minRows;
    private final double maxMissingRatio;
    private final Set<String> requiredColumns;
    private final boolean enableSecurityChecks;
    
    // Security patterns for validation
    private static final Map<String, Pattern> SECURITY_PATTERNS = Map.of(
        "sql_injection", Pattern.compile("(?i)('|(\\\\')|(\"|(\\\\)\");|(\\bor\\b|\\bOR\\b).+?(=|like)|\\bunion\\b|\\bUNION\\b|\\bselect\\b|\\bSELECT\\b)"),
        "xss", Pattern.compile("(?i)(<script|javascript:|onload=|onerror=|<iframe|eval\\(|alert\\()"),
        "path_traversal", Pattern.compile("\\.\\.[\\\\/]"),
        "command_injection", Pattern.compile("(?i)(;|\\||&|\\$\\(|`|\\bcat\\b|\\brm\\b|\\bls\\b)")
    );
    
    public DataValidator() {
        this(10, 0.3, Collections.emptySet(), true);
    }
    
    public DataValidator(int minRows, double maxMissingRatio, Set<String> requiredColumns, boolean enableSecurityChecks) {
        this.minRows = minRows;
        this.maxMissingRatio = maxMissingRatio;
        this.requiredColumns = new HashSet<>(requiredColumns);
        this.enableSecurityChecks = enableSecurityChecks;
    }
    
    /**
     * Validate a list of data points for ML training
     */
    public ValidationResult validateDataPoints(List<DataPoint> dataPoints) throws DataValidationException {
        if (dataPoints == null) {
            throw new IllegalArgumentException("Data points list cannot be null");
        }
        
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        Map<String, Integer> missingValues = new HashMap<>();
        
        // Check minimum rows
        if (dataPoints.size() < minRows) {
            errors.add(String.format("Insufficient data: %d rows < %d", dataPoints.size(), minRows));
        }
        
        // Check for empty dataset
        if (dataPoints.isEmpty()) {
            errors.add("Dataset is empty");
            return new ValidationResult(false, errors, warnings, 0, 0, missingValues);
        }
        
        // Check feature consistency
        int expectedFeatureCount = dataPoints.get(0).getFeatureCount();
        for (int i = 0; i < dataPoints.size(); i++) {
            DataPoint point = dataPoints.get(i);
            if (point.getFeatureCount() != expectedFeatureCount) {
                errors.add(String.format("Inconsistent feature count at row %d: expected %d, got %d",
                                        i, expectedFeatureCount, point.getFeatureCount()));
            }
            
            // Check for invalid values
            double[] features = point.getFeatures();
            for (int j = 0; j < features.length; j++) {
                if (Double.isNaN(features[j])) {
                    errors.add(String.format("NaN value found at row %d, feature %d", i, j));
                }
                if (Double.isInfinite(features[j])) {
                    errors.add(String.format("Infinite value found at row %d, feature %d", i, j));
                }
            }
            
            // Check target value
            if (Double.isNaN(point.getTarget())) {
                errors.add(String.format("NaN target value at row %d", i));
            }
            if (Double.isInfinite(point.getTarget())) {
                errors.add(String.format("Infinite target value at row %d", i));
            }
        }
        
        // Check for duplicates
        Set<DataPoint> uniquePoints = new HashSet<>(dataPoints);
        if (uniquePoints.size() != dataPoints.size()) {
            int duplicateCount = dataPoints.size() - uniquePoints.size();
            warnings.add(String.format("Found %d duplicate rows (%.2f%%)",
                                     duplicateCount, (double) duplicateCount / dataPoints.size() * 100));
        }
        
        // Security validation on string data (if any metadata contains strings)
        if (enableSecurityChecks) {
            validateSecurity(dataPoints, errors);
        }
        
        boolean isValid = errors.isEmpty();
        return new ValidationResult(isValid, errors, warnings, dataPoints.size(), expectedFeatureCount, missingValues);
    }
    
    /**
     * Validate features and target arrays for ML training
     */
    public ValidationResult validateFeaturesTarget(double[][] X, double[] y) throws DataValidationException {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        Map<String, Integer> missingValues = new HashMap<>();
        
        // Check null inputs
        if (X == null || y == null) {
            throw new IllegalArgumentException("Features and target arrays cannot be null");
        }
        
        // Check shape consistency
        if (X.length != y.length) {
            errors.add(String.format("Feature and target length mismatch: %d != %d", X.length, y.length));
        }
        
        // Check minimum rows
        if (X.length < minRows) {
            errors.add(String.format("Insufficient data: %d rows < %d", X.length, minRows));
        }
        
        if (X.length == 0) {
            errors.add("Empty feature matrix");
            return new ValidationResult(false, errors, warnings, 0, 0, missingValues);
        }
        
        int featureCount = X[0].length;
        
        // Validate feature matrix
        for (int i = 0; i < X.length; i++) {
            if (X[i] == null) {
                errors.add(String.format("Null feature row at index %d", i));
                continue;
            }
            
            if (X[i].length != featureCount) {
                errors.add(String.format("Inconsistent feature count at row %d: expected %d, got %d",
                                        i, featureCount, X[i].length));
                continue;
            }
            
            for (int j = 0; j < X[i].length; j++) {
                if (Double.isNaN(X[i][j])) {
                    errors.add(String.format("NaN value in features at [%d][%d]", i, j));
                }
                if (Double.isInfinite(X[i][j])) {
                    errors.add(String.format("Infinite value in features at [%d][%d]", i, j));
                }
            }
        }
        
        // Validate target array
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(y[i])) {
                errors.add(String.format("NaN value in target at index %d", i));
            }
            if (Double.isInfinite(y[i])) {
                errors.add(String.format("Infinite value in target at index %d", i));
            }
        }
        
        boolean isValid = errors.isEmpty();
        return new ValidationResult(isValid, errors, warnings, X.length, featureCount, missingValues);
    }
    
    private void validateSecurity(List<DataPoint> dataPoints, List<String> errors) {
        // This is a simplified security check - in practice, you'd validate string fields
        // For this example, we'll check if any ID contains suspicious patterns
        for (int i = 0; i < dataPoints.size(); i++) {
            String id = dataPoints.get(i).getId();
            if (id != null) {
                for (Map.Entry<String, Pattern> entry : SECURITY_PATTERNS.entrySet()) {
                    if (entry.getValue().matcher(id).find()) {
                        errors.add(String.format("Potential %s pattern detected in ID at row %d", 
                                                entry.getKey(), i));
                    }
                }
            }
        }
    }
}

/**
 * Simple linear regression implementation with comprehensive error handling
 */
class SimpleLinearRegression {
    private double[] weights;
    private double bias;
    private boolean isTrained = false;
    private long trainingTimeMs = 0;
    private final Object lock = new Object(); // For thread safety
    
    // Hyperparameters
    private final double learningRate;
    private final int maxEpochs;
    private final double tolerance;
    
    public SimpleLinearRegression() {
        this(0.01, 1000, 1e-6);
    }
    
    public SimpleLinearRegression(double learningRate, int maxEpochs, double tolerance) {
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        this.tolerance = tolerance;
    }
    
    /**
     * Train the linear regression model
     */
    public void fit(double[][] X, double[] y) throws ModelTrainingException, DataValidationException {
        synchronized (lock) {
            long startTime = System.currentTimeMillis();
            
            try {
                // Validate input data
                DataValidator validator = new DataValidator();
                ValidationResult validation = validator.validateFeaturesTarget(X, y);
                if (!validation.isValid()) {
                    throw new DataValidationException("Data validation failed: " + 
                                                    String.join(", ", validation.getErrors()));
                }
                
                int numSamples = X.length;
                int numFeatures = X[0].length;
                
                // Initialize weights and bias
                weights = new double[numFeatures];
                Random random = new Random(42); // Fixed seed for reproducibility
                for (int i = 0; i < numFeatures; i++) {
                    weights[i] = random.nextGaussian() * 0.01;
                }
                bias = 0.0;
                
                // Gradient descent
                double previousCost = Double.MAX_VALUE;
                
                for (int epoch = 0; epoch < maxEpochs; epoch++) {
                    // Forward pass - calculate predictions
                    double[] predictions = new double[numSamples];
                    for (int i = 0; i < numSamples; i++) {
                        predictions[i] = bias;
                        for (int j = 0; j < numFeatures; j++) {
                            predictions[i] += X[i][j] * weights[j];
                        }
                    }
                    
                    // Calculate cost (MSE)
                    double cost = 0.0;
                    for (int i = 0; i < numSamples; i++) {
                        double error = predictions[i] - y[i];
                        cost += error * error;
                    }
                    cost /= (2 * numSamples);
                    
                    // Check for convergence
                    if (Math.abs(previousCost - cost) < tolerance) {
                        System.out.printf("Converged at epoch %d with cost %.6f%n", epoch, cost);
                        break;
                    }
                    previousCost = cost;
                    
                    // Calculate gradients
                    double[] weightGradients = new double[numFeatures];
                    double biasGradient = 0.0;
                    
                    for (int i = 0; i < numSamples; i++) {
                        double error = predictions[i] - y[i];
                        biasGradient += error;
                        
                        for (int j = 0; j < numFeatures; j++) {
                            weightGradients[j] += error * X[i][j];
                        }
                    }
                    
                    // Update parameters
                    for (int j = 0; j < numFeatures; j++) {
                        weights[j] -= (learningRate * weightGradients[j]) / numSamples;
                    }
                    bias -= (learningRate * biasGradient) / numSamples;
                }
                
                trainingTimeMs = System.currentTimeMillis() - startTime;
                isTrained = true;
                
                System.out.printf("Model training completed in %d ms%n", trainingTimeMs);
                
            } catch (Exception e) {
                throw new ModelTrainingException("Training failed: " + e.getMessage(), e);
            }
        }
    }
    
    /**
     * Make predictions using the trained model
     */
    public double[] predict(double[][] X) throws PredictionException {
        if (!isTrained) {
            throw new PredictionException("Model must be trained before making predictions");
        }
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input features cannot be null or empty");
        }
        
        try {
            double[] predictions = new double[X.length];
            
            for (int i = 0; i < X.length; i++) {
                if (X[i] == null) {
                    throw new PredictionException("Feature row " + i + " is null");
                }
                
                if (X[i].length != weights.length) {
                    throw new PredictionException(String.format(
                        "Feature count mismatch at row %d: expected %d, got %d",
                        i, weights.length, X[i].length));
                }
                
                // Calculate prediction: y = X * w + b
                double prediction = bias;
                for (int j = 0; j < weights.length; j++) {
                    if (Double.isNaN(X[i][j]) || Double.isInfinite(X[i][j])) {
                        throw new PredictionException(String.format(
                            "Invalid value at [%d][%d]: %f", i, j, X[i][j]));
                    }
                    prediction += X[i][j] * weights[j];
                }
                predictions[i] = prediction;
            }
            
            return predictions;
            
        } catch (Exception e) {
            throw new PredictionException("Prediction failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Evaluate the model and return comprehensive metrics
     */
    public ModelMetrics evaluate(double[][] X, double[] y) throws PredictionException, DataValidationException {
        long startTime = System.currentTimeMillis();
        
        // Make predictions
        double[] predictions = predict(X);
        long predictionTime = System.currentTimeMillis() - startTime;
        
        // Calculate metrics
        double mse = 0.0;
        double sumSquaredTotal = 0.0;
        double yMean = Arrays.stream(y).average().orElse(0.0);
        
        for (int i = 0; i < y.length; i++) {
            double error = predictions[i] - y[i];
            mse += error * error;
            sumSquaredTotal += Math.pow(y[i] - yMean, 2);
        }
        
        mse /= y.length;
        double rmse = Math.sqrt(mse);
        double rSquared = 1.0 - (mse * y.length) / sumSquaredTotal;
        
        // Estimate model size
        long modelSize = estimateModelSize();
        
        System.out.printf("Model evaluation completed. R²: %.4f, RMSE: %.4f%n", rSquared, rmse);
        
        return new ModelMetrics(rSquared, rmse, mse, rSquared, trainingTimeMs, predictionTime, modelSize);
    }
    
    private long estimateModelSize() {
        // Rough estimation: weights + bias + metadata
        return (weights.length + 1) * Double.BYTES + 64; // 64 bytes for metadata
    }
    
    /**
     * Save model to a string representation
     */
    public String saveModel() throws ModelTrainingException {
        if (!isTrained) {
            throw new ModelTrainingException("Cannot save untrained model");
        }
        
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("SimpleLinearRegression\n");
            sb.append("weights:").append(Arrays.toString(weights)).append("\n");
            sb.append("bias:").append(bias).append("\n");
            sb.append("learningRate:").append(learningRate).append("\n");
            sb.append("maxEpochs:").append(maxEpochs).append("\n");
            sb.append("tolerance:").append(tolerance).append("\n");
            sb.append("trainingTime:").append(trainingTimeMs).append("\n");
            sb.append("timestamp:").append(Instant.now()).append("\n");
            
            return sb.toString();
            
        } catch (Exception e) {
            throw new ModelTrainingException("Failed to save model: " + e.getMessage(), e);
        }
    }
    
    /**
     * Load model from string representation
     */
    public void loadModel(String modelData) throws ModelTrainingException {
        try {
            String[] lines = modelData.split("\n");
            for (String line : lines) {
                if (line.startsWith("weights:")) {
                    String weightsStr = line.substring(8);
                    weightsStr = weightsStr.substring(1, weightsStr.length() - 1); // Remove [ ]
                    String[] weightStrs = weightsStr.split(", ");
                    weights = Arrays.stream(weightStrs).mapToDouble(Double::parseDouble).toArray();
                } else if (line.startsWith("bias:")) {
                    bias = Double.parseDouble(line.substring(5));
                } else if (line.startsWith("trainingTime:")) {
                    trainingTimeMs = Long.parseLong(line.substring(13));
                }
            }
            
            isTrained = true;
            System.out.println("Model loaded successfully");
            
        } catch (Exception e) {
            throw new ModelTrainingException("Failed to load model: " + e.getMessage(), e);
        }
    }
    
    // Getters
    public boolean isTrained() { return isTrained; }
    public double[] getWeights() { return Arrays.copyOf(weights, weights.length); }
    public double getBias() { return bias; }
    public long getTrainingTime() { return trainingTimeMs; }
}

/**
 * Feature engineering utilities for Java ML
 */
class FeatureEngineer {
    private final Map<String, Double> featureMeans = new HashMap<>();
    private final Map<String, Double> featureStds = new HashMap<>();
    private boolean isScalerFitted = false;
    
    /**
     * Create polynomial features up to a specified degree
     */
    public double[][] createPolynomialFeatures(double[][] X, int degree) {
        if (degree < 2 || degree > 5) {
            throw new IllegalArgumentException("Polynomial degree must be between 2 and 5");
        }
        
        long startTime = System.currentTimeMillis();
        System.out.printf("Creating polynomial features of degree %d...%n", degree);
        
        int numSamples = X.length;
        int numOriginalFeatures = X[0].length;
        
        // Calculate number of polynomial features
        int numPolyFeatures = numOriginalFeatures;
        
        // Add squared terms
        numPolyFeatures += numOriginalFeatures;
        
        // Add interaction terms
        if (degree >= 2) {
            numPolyFeatures += (numOriginalFeatures * (numOriginalFeatures - 1)) / 2;
        }
        
        double[][] polyFeatures = new double[numSamples][numPolyFeatures];
        
        for (int i = 0; i < numSamples; i++) {
            int featureIndex = 0;
            
            // Original features
            System.arraycopy(X[i], 0, polyFeatures[i], featureIndex, numOriginalFeatures);
            featureIndex += numOriginalFeatures;
            
            // Squared terms
            for (int j = 0; j < numOriginalFeatures; j++) {
                polyFeatures[i][featureIndex++] = X[i][j] * X[i][j];
            }
            
            // Interaction terms
            if (degree >= 2) {
                for (int j = 0; j < numOriginalFeatures; j++) {
                    for (int k = j + 1; k < numOriginalFeatures; k++) {
                        polyFeatures[i][featureIndex++] = X[i][j] * X[i][k];
                    }
                }
            }
        }
        
        long endTime = System.currentTimeMillis();
        System.out.printf("Polynomial features created in %d ms: %d -> %d features%n",
                         endTime - startTime, numOriginalFeatures, numPolyFeatures);
        
        return polyFeatures;
    }
    
    /**
     * Apply standard scaling (z-score normalization)
     */
    public double[][] standardScaler(double[][] X, boolean fit) {
        long startTime = System.currentTimeMillis();
        
        if (X.length == 0 || X[0].length == 0) {
            throw new IllegalArgumentException("Cannot scale empty data");
        }
        
        int numSamples = X.length;
        int numFeatures = X[0].length;
        
        if (fit) {
            // Calculate means and standard deviations
            for (int j = 0; j < numFeatures; j++) {
                double sum = 0.0;
                for (int i = 0; i < numSamples; i++) {
                    sum += X[i][j];
                }
                double mean = sum / numSamples;
                featureMeans.put("feature_" + j, mean);
                
                double sumSquaredDiffs = 0.0;
                for (int i = 0; i < numSamples; i++) {
                    sumSquaredDiffs += Math.pow(X[i][j] - mean, 2);
                }
                double std = Math.sqrt(sumSquaredDiffs / numSamples);
                featureStds.put("feature_" + j, std);
            }
            isScalerFitted = true;
        }
        
        if (!isScalerFitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        
        // Apply scaling
        double[][] scaledX = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                double mean = featureMeans.get("feature_" + j);
                double std = featureStds.get("feature_" + j);
                
                if (std == 0.0) {
                    scaledX[i][j] = 0.0; // Constant feature
                } else {
                    scaledX[i][j] = (X[i][j] - mean) / std;
                }
            }
        }
        
        long endTime = System.currentTimeMillis();
        System.out.printf("Standard scaling applied in %d ms to %d features%n",
                         endTime - startTime, numFeatures);
        
        return scaledX;
    }
    
    /**
     * Detect outliers using the IQR method
     */
    public Map<String, Object> detectOutliers(double[][] X, String method, double threshold) {
        long startTime = System.currentTimeMillis();
        System.out.printf("Detecting outliers using %s method...%n", method);
        
        Set<Integer> outlierIndices = new HashSet<>();
        List<Map<String, Object>> outlierInfo = new ArrayList<>();
        int numFeatures = X[0].length;
        
        for (int j = 0; j < numFeatures; j++) {
            // Extract column
            double[] column = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                column[i] = X[i][j];
            }
            
            List<Integer> featureOutliers = new ArrayList<>();
            
            if ("iqr".equals(method)) {
                // Sort column for quartile calculation
                double[] sortedColumn = Arrays.copyOf(column, column.length);
                Arrays.sort(sortedColumn);
                
                int n = sortedColumn.length;
                double q1 = sortedColumn[n / 4];
                double q3 = sortedColumn[3 * n / 4];
                double iqr = q3 - q1;
                
                double lowerBound = q1 - threshold * iqr;
                double upperBound = q3 + threshold * iqr;
                
                for (int i = 0; i < column.length; i++) {
                    if (column[i] < lowerBound || column[i] > upperBound) {
                        featureOutliers.add(i);
                        outlierIndices.add(i);
                    }
                }
                
            } else if ("zscore".equals(method)) {
                double mean = Arrays.stream(column).average().orElse(0.0);
                double sumSquaredDiffs = Arrays.stream(column)
                                               .map(x -> Math.pow(x - mean, 2))
                                               .sum();
                double std = Math.sqrt(sumSquaredDiffs / column.length);
                
                for (int i = 0; i < column.length; i++) {
                    double zscore = Math.abs((column[i] - mean) / std);
                    if (zscore > threshold) {
                        featureOutliers.add(i);
                        outlierIndices.add(i);
                    }
                }
            }
            
            if (!featureOutliers.isEmpty()) {
                Map<String, Object> info = new HashMap<>();
                info.put("feature", j);
                info.put("outlierCount", featureOutliers.size());
                info.put("outlierPercentage", (double) featureOutliers.size() / column.length * 100);
                info.put("method", method);
                info.put("threshold", threshold);
                outlierInfo.add(info);
            }
        }
        
        long endTime = System.currentTimeMillis();
        System.out.printf("Outlier detection completed in %d ms. Found %d outlier rows%n",
                         endTime - startTime, outlierIndices.size());
        
        Map<String, Object> result = new HashMap<>();
        result.put("outlierIndices", new ArrayList<>(outlierIndices));
        result.put("outlierInfo", outlierInfo);
        
        return result;
    }
}

/**
 * ML utilities for data generation and processing
 */
class MLUtilities {
    
    /**
     * Generate synthetic regression data
     */
    public static Map<String, Object> generateSyntheticData(int numSamples, int numFeatures, double noiseLevel) {
        System.out.printf("Generating synthetic dataset: %d samples, %d features%n", numSamples, numFeatures);
        
        Random random = new Random(42); // Fixed seed for reproducibility
        
        // Generate true weights for linear relationship
        double[] trueWeights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            trueWeights[i] = random.nextGaussian();
        }
        double trueBias = random.nextGaussian();
        
        // Generate data
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Generate features
            for (int j = 0; j < numFeatures; j++) {
                X[i][j] = random.nextGaussian() * 2; // Scale features
            }
            
            // Generate target with linear relationship + noise
            double target = trueBias;
            for (int j = 0; j < numFeatures; j++) {
                target += X[i][j] * trueWeights[j];
            }
            target += random.nextGaussian() * noiseLevel; // Add noise
            y[i] = target;
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("X", X);
        result.put("y", y);
        result.put("trueWeights", trueWeights);
        result.put("trueBias", trueBias);
        
        return result;
    }
    
    /**
     * Split data into train and test sets
     */
    public static Map<String, Object> trainTestSplit(double[][] X, double[] y, double testSize, int randomSeed) {
        Random random = new Random(randomSeed);
        
        // Create indices and shuffle
        List<Integer> indices = IntStream.range(0, X.length).boxed().collect(Collectors.toList());
        Collections.shuffle(indices, random);
        
        int testLength = (int) (X.length * testSize);
        int trainLength = X.length - testLength;
        
        // Split indices
        List<Integer> trainIndices = indices.subList(0, trainLength);
        List<Integer> testIndices = indices.subList(trainLength, X.length);
        
        // Create train and test arrays
        double[][] XTrain = new double[trainLength][];
        double[] yTrain = new double[trainLength];
        double[][] XTest = new double[testLength][];
        double[] yTest = new double[testLength];
        
        for (int i = 0; i < trainLength; i++) {
            int idx = trainIndices.get(i);
            XTrain[i] = Arrays.copyOf(X[idx], X[idx].length);
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < testLength; i++) {
            int idx = testIndices.get(i);
            XTest[i] = Arrays.copyOf(X[idx], X[idx].length);
            yTest[i] = y[idx];
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("XTrain", XTrain);
        result.put("yTrain", yTrain);
        result.put("XTest", XTest);
        result.put("yTest", yTest);
        
        return result;
    }
}

/**
 * Main demonstration class
 */
public class MLProductionPatterns {
    
    /**
     * Comprehensive demonstration of Java ML patterns
     */
    public static void demonstrateJavaMLPatterns() {
        System.out.println("🚀 Java ML Pipeline Demonstration");
        System.out.println("=".repeat(45));
        
        try {
            // Generate synthetic data
            Map<String, Object> data = MLUtilities.generateSyntheticData(1000, 5, 0.1);
            double[][] X = (double[][]) data.get("X");
            double[] y = (double[]) data.get("y");
            System.out.printf("✅ Generated dataset: %d samples, %d features%n", X.length, X[0].length);
            
            // Split data
            Map<String, Object> split = MLUtilities.trainTestSplit(X, y, 0.2, 42);
            double[][] XTrain = (double[][]) split.get("XTrain");
            double[] yTrain = (double[]) split.get("yTrain");
            double[][] XTest = (double[][]) split.get("XTest");
            double[] yTest = (double[]) split.get("yTest");
            System.out.printf("✅ Data split: %d training, %d test samples%n", XTrain.length, XTest.length);
            
            // Validate data
            System.out.println("\n🔄 Validating data...");
            DataValidator validator = new DataValidator();
            ValidationResult validation = validator.validateFeaturesTarget(XTrain, yTrain);
            
            if (!validation.isValid()) {
                System.err.println("❌ Data validation failed: " + validation.getErrors());
                return;
            }
            System.out.println("✅ Data validation passed");
            
            // Train model
            System.out.println("\n🔄 Training model...");
            SimpleLinearRegression model = new SimpleLinearRegression();
            model.fit(XTrain, yTrain);
            System.out.println("✅ Model training completed");
            
            // Make predictions
            System.out.println("\n🔮 Making predictions...");
            double[][] XSample = Arrays.copyOf(XTest, Math.min(10, XTest.length));
            double[] predictions = model.predict(XSample);
            System.out.printf("Sample predictions: [%.4f, %.4f, %.4f, %.4f, %.4f]%n",
                             predictions[0], predictions[1], predictions[2], predictions[3], predictions[4]);
            
            // Evaluate model
            System.out.println("\n📊 Evaluating model...");
            ModelMetrics metrics = model.evaluate(XTest, yTest);
            System.out.printf("R²: %.4f%n", metrics.getRSquared());
            System.out.printf("RMSE: %.4f%n", metrics.getRmse());
            System.out.printf("Training time: %d ms%n", metrics.getTrainingTimeMs());
            System.out.printf("Model size: %d bytes%n", metrics.getModelSizeBytes());
            
            // Feature engineering demonstration
            System.out.println("\n🔧 Feature Engineering...");
            FeatureEngineer engineer = new FeatureEngineer();
            
            // Create polynomial features
            double[][] XTrainSample = Arrays.copyOf(XTrain, Math.min(100, XTrain.length));
            double[][] polyFeatures = engineer.createPolynomialFeatures(XTrainSample, 2);
            System.out.printf("Features after polynomial engineering: %d -> %d%n", 
                             XTrainSample[0].length, polyFeatures[0].length);
            
            // Standard scaling
            double[][] scaledFeatures = engineer.standardScaler(XTrainSample, true);
            System.out.printf("Standard scaling applied to %d features%n", scaledFeatures[0].length);
            
            // Outlier detection
            @SuppressWarnings("unchecked")
            Map<String, Object> outlierResults = engineer.detectOutliers(XTrain, "iqr", 1.5);
            List<Integer> outlierIndices = (List<Integer>) outlierResults.get("outlierIndices");
            System.out.printf("Outlier detection: %d outlier rows found%n", outlierIndices.size());
            
            // Model persistence
            System.out.println("\n💾 Testing model persistence...");
            String modelData = model.saveModel();
            SimpleLinearRegression newModel = new SimpleLinearRegression();
            newModel.loadModel(modelData);
            System.out.println("✅ Model save/load completed");
            
            System.out.println("\n✅ Java ML Pipeline demonstration completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Demonstration failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        demonstrateJavaMLPatterns();
    }
}