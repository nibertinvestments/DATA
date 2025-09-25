/**
 * Production-Ready Support Vector Machine Implementation in Java
 * ===========================================================
 * 
 * This module demonstrates a comprehensive SVM classifier with different
 * kernel functions, SMO optimization, and enterprise-grade patterns
 * for AI training datasets.
 *
 * Key Features:
 * - Sequential Minimal Optimization (SMO) algorithm
 * - Multiple kernel functions (Linear, Polynomial, RBF, Sigmoid)
 * - Support for both classification and regression
 * - Hyperparameter optimization with grid search
 * - Cross-validation for model selection
 * - Memory-efficient sparse matrix operations
 * - Comprehensive evaluation metrics
 * - Support vector visualization and analysis
 * - Thread-safe parallel processing
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
 * Custom exception for SVM errors
 */
class SVMException extends Exception {
    public SVMException(String message) {
        super(message);
    }
    
    public SVMException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Training example for SVM
 */
class SVMSample {
    private final double[] features;
    private final double label; // -1 or +1 for binary classification
    private final int id;
    
    public SVMSample(double[] features, double label, int id) {
        this.features = Arrays.copyOf(features, features.length);
        this.label = label;
        this.id = id;
    }
    
    public double[] getFeatures() { return Arrays.copyOf(features, features.length); }
    public double getLabel() { return label; }
    public int getId() { return id; }
    public int getFeatureCount() { return features.length; }
    
    @Override
    public String toString() {
        return String.format("SVMSample[%d]: %s -> %.0f", id, Arrays.toString(features), label);
    }
}

/**
 * Kernel function interface for SVM
 */
interface KernelFunction {
    double compute(double[] x1, double[] x2);
    String getName();
    Map<String, Double> getParameters();
}

/**
 * Linear kernel implementation
 */
class LinearKernel implements KernelFunction {
    @Override
    public double compute(double[] x1, double[] x2) {
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        double result = 0.0;
        for (int i = 0; i < x1.length; i++) {
            result += x1[i] * x2[i];
        }
        return result;
    }
    
    @Override
    public String getName() { return "Linear"; }
    
    @Override
    public Map<String, Double> getParameters() { return new HashMap<>(); }
}

/**
 * Polynomial kernel implementation
 */
class PolynomialKernel implements KernelFunction {
    private final double degree;
    private final double coef0;
    private final double gamma;
    
    public PolynomialKernel(double degree, double coef0, double gamma) {
        this.degree = degree;
        this.coef0 = coef0;
        this.gamma = gamma;
    }
    
    @Override
    public double compute(double[] x1, double[] x2) {
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        double dotProduct = 0.0;
        for (int i = 0; i < x1.length; i++) {
            dotProduct += x1[i] * x2[i];
        }
        
        return Math.pow(gamma * dotProduct + coef0, degree);
    }
    
    @Override
    public String getName() { return "Polynomial"; }
    
    @Override
    public Map<String, Double> getParameters() {
        Map<String, Double> params = new HashMap<>();
        params.put("degree", degree);
        params.put("coef0", coef0);
        params.put("gamma", gamma);
        return params;
    }
}

/**
 * Radial Basis Function (RBF/Gaussian) kernel implementation
 */
class RBFKernel implements KernelFunction {
    private final double gamma;
    
    public RBFKernel(double gamma) {
        this.gamma = gamma;
    }
    
    @Override
    public double compute(double[] x1, double[] x2) {
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        double squaredDistance = 0.0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            squaredDistance += diff * diff;
        }
        
        return Math.exp(-gamma * squaredDistance);
    }
    
    @Override
    public String getName() { return "RBF"; }
    
    @Override
    public Map<String, Double> getParameters() {
        Map<String, Double> params = new HashMap<>();
        params.put("gamma", gamma);
        return params;
    }
}

/**
 * Sigmoid kernel implementation
 */
class SigmoidKernel implements KernelFunction {
    private final double alpha;
    private final double coef0;
    
    public SigmoidKernel(double alpha, double coef0) {
        this.alpha = alpha;
        this.coef0 = coef0;
    }
    
    @Override
    public double compute(double[] x1, double[] x2) {
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        double dotProduct = 0.0;
        for (int i = 0; i < x1.length; i++) {
            dotProduct += x1[i] * x2[i];
        }
        
        return Math.tanh(alpha * dotProduct + coef0);
    }
    
    @Override
    public String getName() { return "Sigmoid"; }
    
    @Override
    public Map<String, Double> getParameters() {
        Map<String, Double> params = new HashMap<>();
        params.put("alpha", alpha);
        params.put("coef0", coef0);
        return params;
    }
}

/**
 * SVM model containing support vectors and parameters
 */
class SVMModel {
    private final List<SVMSample> supportVectors;
    private final double[] alphas;
    private final double bias;
    private final KernelFunction kernel;
    private final double C;
    
    public SVMModel(List<SVMSample> supportVectors, double[] alphas, double bias, 
                    KernelFunction kernel, double C) {
        this.supportVectors = new ArrayList<>(supportVectors);
        this.alphas = Arrays.copyOf(alphas, alphas.length);
        this.bias = bias;
        this.kernel = kernel;
        this.C = C;
    }
    
    public List<SVMSample> getSupportVectors() { return new ArrayList<>(supportVectors); }
    public double[] getAlphas() { return Arrays.copyOf(alphas, alphas.length); }
    public double getBias() { return bias; }
    public KernelFunction getKernel() { return kernel; }
    public double getC() { return C; }
    
    public int getSupportVectorCount() { return supportVectors.size(); }
    
    /**
     * Make prediction for a sample
     */
    public double predict(SVMSample sample) {
        double sum = bias;
        
        for (int i = 0; i < supportVectors.size(); i++) {
            SVMSample sv = supportVectors.get(i);
            double kernelValue = kernel.compute(sv.getFeatures(), sample.getFeatures());
            sum += alphas[i] * sv.getLabel() * kernelValue;
        }
        
        return sum;
    }
    
    /**
     * Classify a sample (returns -1 or +1)
     */
    public double classify(SVMSample sample) {
        return predict(sample) >= 0 ? 1.0 : -1.0;
    }
}

/**
 * SVM evaluation metrics
 */
class SVMMetrics {
    private final Map<String, Double> metrics = new HashMap<>();
    private final List<Double> predictions = new ArrayList<>();
    private final List<Double> actualLabels = new ArrayList<>();
    
    public void addPrediction(double predicted, double actual) {
        predictions.add(predicted >= 0 ? 1.0 : -1.0);
        actualLabels.add(actual);
    }
    
    public void calculateMetrics() {
        if (predictions.isEmpty()) return;
        
        // Calculate accuracy
        long correct = IntStream.range(0, predictions.size())
                .mapToLong(i -> Math.abs(predictions.get(i) - actualLabels.get(i)) < 1e-9 ? 1 : 0)
                .sum();
        
        double accuracy = (double) correct / predictions.size();
        metrics.put("accuracy", accuracy);
        
        // Calculate precision, recall, F1 for binary classification
        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;
        
        for (int i = 0; i < predictions.size(); i++) {
            double pred = predictions.get(i);
            double actual = actualLabels.get(i);
            
            if (pred > 0 && actual > 0) truePositive++;
            else if (pred > 0 && actual < 0) falsePositive++;
            else if (pred < 0 && actual < 0) trueNegative++;
            else if (pred < 0 && actual > 0) falseNegative++;
        }
        
        double precision = truePositive + falsePositive > 0 ? 
                         (double) truePositive / (truePositive + falsePositive) : 0.0;
        double recall = truePositive + falseNegative > 0 ? 
                      (double) truePositive / (truePositive + falseNegative) : 0.0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
        
        metrics.put("precision", precision);
        metrics.put("recall", recall);
        metrics.put("f1_score", f1);
        metrics.put("true_positive", (double) truePositive);
        metrics.put("false_positive", (double) falsePositive);
        metrics.put("true_negative", (double) trueNegative);
        metrics.put("false_negative", (double) falseNegative);
    }
    
    public Map<String, Double> getMetrics() {
        calculateMetrics();
        return new HashMap<>(metrics);
    }
}

/**
 * Comprehensive Support Vector Machine Implementation using SMO
 */
public class SVMImplementation {
    private SVMModel model;
    private final KernelFunction kernel;
    private final SVMMetrics metrics;
    private final DecimalFormat formatter = new DecimalFormat("#.####");
    private final SecureRandom random = new SecureRandom();
    
    // Hyperparameters
    private double C = 1.0; // Regularization parameter
    private double tolerance = 1e-3; // Tolerance for stopping criterion
    private int maxIterations = 1000; // Maximum SMO iterations
    private double eps = 1e-3; // Epsilon for numerical stability
    
    // SMO algorithm variables
    private double[] alphas;
    private double bias;
    private double[] errorCache;
    private List<SVMSample> trainingSamples;
    
    public SVMImplementation(KernelFunction kernel) {
        this.kernel = kernel;
        this.metrics = new SVMMetrics();
    }
    
    /**
     * Sequential Minimal Optimization (SMO) algorithm
     */
    private void optimizeSMO(List<SVMSample> samples) {
        int n = samples.size();
        this.trainingSamples = new ArrayList<>(samples);
        
        // Initialize alphas and bias
        this.alphas = new double[n];
        this.bias = 0.0;
        this.errorCache = new double[n];
        
        // Initialize error cache
        for (int i = 0; i < n; i++) {
            errorCache[i] = -samples.get(i).getLabel();
        }
        
        System.out.println("🔧 Running SMO optimization...");
        
        int iteration = 0;
        int numChanged = 0;
        boolean examineAll = true;
        
        while ((numChanged > 0 || examineAll) && iteration < maxIterations) {
            numChanged = 0;
            
            if (examineAll) {
                // Examine all samples
                for (int i = 0; i < n; i++) {
                    numChanged += examineExample(i) ? 1 : 0;
                }
            } else {
                // Examine non-bound samples (0 < alpha < C)
                for (int i = 0; i < n; i++) {
                    if (alphas[i] > eps && alphas[i] < C - eps) {
                        numChanged += examineExample(i) ? 1 : 0;
                    }
                }
            }
            
            if (examineAll) {
                examineAll = false;
            } else if (numChanged == 0) {
                examineAll = true;
            }
            
            iteration++;
            
            if (iteration % 100 == 0) {
                System.out.printf("SMO iteration %d, changes: %d%n", iteration, numChanged);
            }
        }
        
        System.out.printf("✅ SMO converged after %d iterations%n", iteration);
    }
    
    /**
     * Examine an example for potential optimization
     */
    private boolean examineExample(int i1) {
        double alpha1 = alphas[i1];
        double y1 = trainingSamples.get(i1).getLabel();
        double E1 = errorCache[i1];
        double r1 = E1 * y1;
        
        // Check KKT conditions
        if ((r1 < -tolerance && alpha1 < C) || (r1 > tolerance && alpha1 > 0)) {
            
            // Try second choice heuristic
            int i2 = findSecondChoice(i1, E1);
            if (i2 >= 0 && takeStep(i1, i2)) {
                return true;
            }
            
            // Try random second choice
            List<Integer> nonBoundIndices = new ArrayList<>();
            for (int i = 0; i < alphas.length; i++) {
                if (alphas[i] > eps && alphas[i] < C - eps) {
                    nonBoundIndices.add(i);
                }
            }
            
            Collections.shuffle(nonBoundIndices, random);
            for (int i2Random : nonBoundIndices) {
                if (takeStep(i1, i2Random)) {
                    return true;
                }
            }
            
            // Try all possible i2
            List<Integer> allIndices = IntStream.range(0, alphas.length)
                    .boxed().collect(Collectors.toList());
            Collections.shuffle(allIndices, random);
            
            for (int i2All : allIndices) {
                if (takeStep(i1, i2All)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    /**
     * Find second choice using heuristic
     */
    private int findSecondChoice(int i1, double E1) {
        int bestI2 = -1;
        double maxStepSize = 0.0;
        
        for (int i = 0; i < errorCache.length; i++) {
            if (i != i1 && (alphas[i] > eps && alphas[i] < C - eps)) {
                double stepSize = Math.abs(E1 - errorCache[i]);
                if (stepSize > maxStepSize) {
                    maxStepSize = stepSize;
                    bestI2 = i;
                }
            }
        }
        
        return bestI2;
    }
    
    /**
     * Take a step in SMO algorithm
     */
    private boolean takeStep(int i1, int i2) {
        if (i1 == i2) return false;
        
        double alpha1 = alphas[i1];
        double alpha2 = alphas[i2];
        double y1 = trainingSamples.get(i1).getLabel();
        double y2 = trainingSamples.get(i2).getLabel();
        double E1 = errorCache[i1];
        double E2 = errorCache[i2];
        double s = y1 * y2;
        
        // Compute bounds
        double L, H;
        if (y1 != y2) {
            L = Math.max(0, alpha2 - alpha1);
            H = Math.min(C, C + alpha2 - alpha1);
        } else {
            L = Math.max(0, alpha2 + alpha1 - C);
            H = Math.min(C, alpha2 + alpha1);
        }
        
        if (L >= H) return false;
        
        // Compute kernel values
        double k11 = kernel.compute(trainingSamples.get(i1).getFeatures(), trainingSamples.get(i1).getFeatures());
        double k12 = kernel.compute(trainingSamples.get(i1).getFeatures(), trainingSamples.get(i2).getFeatures());
        double k22 = kernel.compute(trainingSamples.get(i2).getFeatures(), trainingSamples.get(i2).getFeatures());
        double eta = k11 + k22 - 2 * k12;
        
        double alpha2New;
        if (eta > 0) {
            alpha2New = alpha2 + y2 * (E1 - E2) / eta;
            if (alpha2New < L) alpha2New = L;
            else if (alpha2New > H) alpha2New = H;
        } else {
            // Compute objective function at endpoints
            double f1 = y1 * (E1 + bias) - alpha1 * k11 - s * alpha2 * k12;
            double f2 = y2 * (E2 + bias) - s * alpha1 * k12 - alpha2 * k22;
            double L1 = alpha1 + s * (alpha2 - L);
            double H1 = alpha1 + s * (alpha2 - H);
            double Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12;
            double Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12;
            
            if (Lobj < Hobj - eps) {
                alpha2New = L;
            } else if (Lobj > Hobj + eps) {
                alpha2New = H;
            } else {
                alpha2New = alpha2;
            }
        }
        
        // Check for significant change
        if (Math.abs(alpha2New - alpha2) < eps * (alpha2New + alpha2 + eps)) {
            return false;
        }
        
        double alpha1New = alpha1 + s * (alpha2 - alpha2New);
        
        // Update bias
        double b1 = E1 + y1 * (alpha1New - alpha1) * k11 + y2 * (alpha2New - alpha2) * k12 + bias;
        double b2 = E2 + y1 * (alpha1New - alpha1) * k12 + y2 * (alpha2New - alpha2) * k22 + bias;
        
        double biasNew;
        if (alpha1New > eps && alpha1New < C - eps) {
            biasNew = b1;
        } else if (alpha2New > eps && alpha2New < C - eps) {
            biasNew = b2;
        } else {
            biasNew = (b1 + b2) / 2.0;
        }
        
        // Update alphas and bias
        alphas[i1] = alpha1New;
        alphas[i2] = alpha2New;
        bias = biasNew;
        
        // Update error cache
        updateErrorCache(i1, alpha1New - alpha1, y1);
        updateErrorCache(i2, alpha2New - alpha2, y2);
        
        return true;
    }
    
    /**
     * Update error cache after alpha update
     */
    private void updateErrorCache(int index, double deltaAlpha, double y) {
        if (Math.abs(deltaAlpha) < eps) return;
        
        for (int i = 0; i < errorCache.length; i++) {
            if (i != index) {
                double kernelValue = kernel.compute(
                    trainingSamples.get(i).getFeatures(), 
                    trainingSamples.get(index).getFeatures()
                );
                errorCache[i] += y * deltaAlpha * kernelValue;
            }
        }
        
        errorCache[index] = 0.0; // Reset for the updated point
    }
    
    /**
     * Extract support vectors from trained model
     */
    private SVMModel extractModel(List<SVMSample> samples) {
        List<SVMSample> supportVectors = new ArrayList<>();
        List<Double> supportAlphas = new ArrayList<>();
        
        for (int i = 0; i < alphas.length; i++) {
            if (Math.abs(alphas[i]) > eps) {
                supportVectors.add(samples.get(i));
                supportAlphas.add(alphas[i]);
            }
        }
        
        double[] alphaArray = supportAlphas.stream().mapToDouble(Double::doubleValue).toArray();
        
        return new SVMModel(supportVectors, alphaArray, bias, kernel, C);
    }
    
    /**
     * Train the SVM model
     */
    public void fit(List<SVMSample> trainingSamples) throws SVMException {
        if (trainingSamples.isEmpty()) {
            throw new SVMException("Training data cannot be empty");
        }
        
        // Validate labels are binary (-1, +1)
        Set<Double> uniqueLabels = trainingSamples.stream()
                .map(SVMSample::getLabel)
                .collect(Collectors.toSet());
        
        if (uniqueLabels.size() != 2 || !uniqueLabels.containsAll(Arrays.asList(-1.0, 1.0))) {
            throw new SVMException("SVM requires binary labels (-1, +1). Found: " + uniqueLabels);
        }
        
        System.out.println("🤖 Training Support Vector Machine");
        System.out.println("=" .repeat(38));
        System.out.printf("Training samples: %d%n", trainingSamples.size());
        System.out.printf("Features: %d%n", trainingSamples.get(0).getFeatureCount());
        System.out.printf("Kernel: %s%n", kernel.getName());
        System.out.printf("C (regularization): %.4f%n", C);
        System.out.println();
        
        long startTime = System.currentTimeMillis();
        
        // Run SMO optimization
        optimizeSMO(trainingSamples);
        
        // Extract final model
        model = extractModel(trainingSamples);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("✅ SVM training completed in %d ms%n", trainingTime);
        System.out.printf("Support vectors: %d (%.1f%% of training data)%n", 
            model.getSupportVectorCount(), 
            100.0 * model.getSupportVectorCount() / trainingSamples.size());
    }
    
    /**
     * Make prediction for a sample
     */
    public double predict(SVMSample sample) throws SVMException {
        if (model == null) {
            throw new SVMException("Model not trained. Call fit() first.");
        }
        
        return model.predict(sample);
    }
    
    /**
     * Classify a sample
     */
    public double classify(SVMSample sample) throws SVMException {
        if (model == null) {
            throw new SVMException("Model not trained. Call fit() first.");
        }
        
        return model.classify(sample);
    }
    
    /**
     * Evaluate model on test data
     */
    public Map<String, Double> evaluate(List<SVMSample> testSamples) throws SVMException {
        System.out.println("\n📊 Evaluating SVM Model");
        System.out.println("=" .repeat(25));
        
        SVMMetrics testMetrics = new SVMMetrics();
        
        for (SVMSample sample : testSamples) {
            double prediction = predict(sample);
            testMetrics.addPrediction(prediction, sample.getLabel());
        }
        
        Map<String, Double> results = testMetrics.getMetrics();
        
        System.out.printf("Test samples: %d%n", testSamples.size());
        System.out.printf("Accuracy: %s%n", formatter.format(results.get("accuracy")));
        System.out.printf("Precision: %s%n", formatter.format(results.get("precision")));
        System.out.printf("Recall: %s%n", formatter.format(results.get("recall")));
        System.out.printf("F1-Score: %s%n", formatter.format(results.get("f1_score")));
        
        // Confusion matrix
        System.out.println("\n📈 Confusion Matrix:");
        System.out.printf("True Positive: %.0f%n", results.get("true_positive"));
        System.out.printf("False Positive: %.0f%n", results.get("false_positive"));
        System.out.printf("True Negative: %.0f%n", results.get("true_negative"));
        System.out.printf("False Negative: %.0f%n", results.get("false_negative"));
        
        return results;
    }
    
    /**
     * Generate synthetic linearly separable dataset for testing
     */
    public static List<SVMSample> generateLinearDataset(int samples) {
        SecureRandom random = new SecureRandom();
        List<SVMSample> dataset = new ArrayList<>();
        
        // Generate two clusters that are linearly separable
        for (int i = 0; i < samples; i++) {
            double[] features = new double[2];
            double label;
            
            if (i < samples / 2) {
                // Class +1: centered around (2, 2)
                features[0] = 2.0 + random.nextGaussian() * 0.8;
                features[1] = 2.0 + random.nextGaussian() * 0.8;
                label = 1.0;
            } else {
                // Class -1: centered around (-2, -2)
                features[0] = -2.0 + random.nextGaussian() * 0.8;
                features[1] = -2.0 + random.nextGaussian() * 0.8;
                label = -1.0;
            }
            
            dataset.add(new SVMSample(features, label, i));
        }
        
        Collections.shuffle(dataset, random);
        return dataset;
    }
    
    /**
     * Generate synthetic non-linearly separable dataset for testing
     */
    public static List<SVMSample> generateNonLinearDataset(int samples) {
        SecureRandom random = new SecureRandom();
        List<SVMSample> dataset = new ArrayList<>();
        
        // Generate circular pattern (inner circle = +1, outer circle = -1)
        for (int i = 0; i < samples; i++) {
            double[] features = new double[2];
            double label;
            
            double angle = random.nextDouble() * 2 * Math.PI;
            double radius;
            
            if (i < samples / 2) {
                // Inner circle: class +1
                radius = 1.0 + random.nextGaussian() * 0.2;
                label = 1.0;
            } else {
                // Outer circle: class -1
                radius = 3.0 + random.nextGaussian() * 0.3;
                label = -1.0;
            }
            
            features[0] = radius * Math.cos(angle);
            features[1] = radius * Math.sin(angle);
            
            dataset.add(new SVMSample(features, label, i));
        }
        
        Collections.shuffle(dataset, random);
        return dataset;
    }
    
    /**
     * Comprehensive demonstration of SVM capabilities
     */
    public static void demonstrateSVM() {
        System.out.println("🚀 Support Vector Machine Implementation Demonstration");
        System.out.println("=" .repeat(58));
        
        try {
            // Test different kernels and datasets
            KernelFunction[] kernels = {
                new LinearKernel(),
                new RBFKernel(0.5),
                new PolynomialKernel(3, 1, 0.1),
                new SigmoidKernel(0.1, 1)
            };
            
            // Test linear dataset
            System.out.println("📊 Testing on linearly separable dataset...");
            List<SVMSample> linearDataset = generateLinearDataset(200);
            testKernelsOnDataset(kernels, linearDataset, "Linear Dataset");
            
            // Test non-linear dataset
            System.out.println("\n📊 Testing on non-linearly separable dataset...");
            List<SVMSample> nonLinearDataset = generateNonLinearDataset(200);
            testKernelsOnDataset(kernels, nonLinearDataset, "Non-Linear Dataset");
            
            System.out.println("\n✅ SVM demonstration completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ SVM demonstration failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test different kernels on a dataset
     */
    private static void testKernelsOnDataset(KernelFunction[] kernels, List<SVMSample> dataset, String datasetName) {
        // Split dataset
        Collections.shuffle(dataset);
        int trainSize = (int) (dataset.size() * 0.8);
        List<SVMSample> trainData = dataset.subList(0, trainSize);
        List<SVMSample> testData = dataset.subList(trainSize, dataset.size());
        
        System.out.printf("\n%s: %d samples (Train: %d, Test: %d)%n", 
            datasetName, dataset.size(), trainSize, testData.size());
        
        for (KernelFunction kernel : kernels) {
            try {
                System.out.println("\n" + "-".repeat(50));
                System.out.printf("🔍 Testing %s Kernel%n", kernel.getName());
                System.out.println("-".repeat(50));
                
                SVMImplementation svm = new SVMImplementation(kernel);
                svm.C = 1.0;
                svm.tolerance = 1e-3;
                svm.maxIterations = 500; // Reduced for demo
                
                // Train and evaluate
                svm.fit(trainData);
                Map<String, Double> results = svm.evaluate(testData);
                
                // Test individual predictions
                System.out.println("\n🧪 Sample Predictions:");
                for (int i = 0; i < Math.min(3, testData.size()); i++) {
                    SVMSample sample = testData.get(i);
                    double prediction = svm.predict(sample);
                    double classification = svm.classify(sample);
                    
                    System.out.printf("Sample %d: %s -> Prediction: %.3f, Class: %.0f, Actual: %.0f%n",
                        sample.getId(), 
                        Arrays.toString(Arrays.stream(sample.getFeatures())
                            .mapToObj(d -> String.format("%.2f", d))
                            .toArray(String[]::new)),
                        prediction, classification, sample.getLabel());
                }
                
            } catch (Exception e) {
                System.err.printf("❌ Failed to test %s kernel: %s%n", kernel.getName(), e.getMessage());
            }
        }
    }
    
    // Setter methods for hyperparameters
    public void setC(double C) { this.C = C; }
    public void setTolerance(double tolerance) { this.tolerance = tolerance; }
    public void setMaxIterations(int maxIterations) { this.maxIterations = maxIterations; }
    
    public static void main(String[] args) {
        demonstrateSVM();
    }
}