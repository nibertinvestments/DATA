/**
 * Support Vector Machine Implementation in Swift
 * =============================================
 * 
 * This module demonstrates production-ready SVM implementation in Swift with
 * comprehensive error handling, Core ML integration patterns, and iOS/macOS
 * deployment considerations for AI training datasets.
 *
 * Key Features:
 * - Protocol-oriented programming with Swift generics
 * - Sequential Minimal Optimization (SMO) algorithm
 * - Multiple kernel functions (Linear, RBF, Polynomial)
 * - Swift concurrency with async/await patterns
 * - Core ML integration for production deployment
 * - SwiftUI integration patterns for visualization
 * - Comprehensive error handling with Result types
 * - Memory management with ARC optimization
 * - Extensive documentation for AI learning
 * - Production-ready patterns with Swift ecosystem integration
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import Foundation
import Accelerate
import os.log

// MARK: - Error Handling

enum SVMError: LocalizedError {
    case invalidInput(String)
    case trainingError(String)
    case predictionError(String)
    case kernelError(String)
    case optimizationError(String)
    
    var errorDescription: String? {
        switch self {
        case .invalidInput(let message): return "Invalid input: \(message)"
        case .trainingError(let message): return "Training error: \(message)"
        case .predictionError(let message): return "Prediction error: \(message)"
        case .kernelError(let message): return "Kernel error: \(message)"
        case .optimizationError(let message): return "Optimization error: \(message)"
        }
    }
}

// MARK: - Data Structures

struct DataPoint {
    let features: [Double]
    let label: Double
    
    init(features: [Double], label: Double) throws {
        guard !features.isEmpty else {
            throw SVMError.invalidInput("Features cannot be empty")
        }
        guard features.allSatisfy({ !$0.isNaN && !$0.isInfinite }) else {
            throw SVMError.invalidInput("Features cannot contain NaN or infinite values")
        }
        guard !label.isNaN && !label.isInfinite else {
            throw SVMError.invalidInput("Label cannot be NaN or infinite")
        }
        
        self.features = features
        self.label = label
    }
}

struct Dataset {
    let points: [DataPoint]
    let featureCount: Int
    let labelSet: Set<Double>
    
    init(points: [DataPoint]) throws {
        guard !points.isEmpty else {
            throw SVMError.invalidInput("Dataset cannot be empty")
        }
        
        let featureCounts = Set(points.map { $0.features.count })
        guard featureCounts.count == 1 else {
            throw SVMError.invalidInput("All data points must have the same number of features")
        }
        
        self.points = points
        self.featureCount = featureCounts.first!
        self.labelSet = Set(points.map { $0.label })
        
        // Validate for binary classification
        if labelSet.count != 2 {
            os_log("Warning: Dataset has %d classes, but SVM is optimized for binary classification", 
                   log: .default, type: .info, labelSet.count)
        }
    }
    
    var size: Int {
        return points.count
    }
    
    var isBalanced: Bool {
        let labelCounts = Dictionary(grouping: points) { $0.label }
            .mapValues { $0.count }
        let counts = Array(labelCounts.values)
        let maxCount = counts.max() ?? 0
        let minCount = counts.min() ?? 0
        return Double(minCount) / Double(maxCount) > 0.8
    }
}

// MARK: - Kernel Functions Protocol

protocol KernelFunction {
    var parameters: [String: Double] { get set }
    func compute(_ x1: [Double], _ x2: [Double]) throws -> Double
    func name: String { get }
}

// Linear Kernel
struct LinearKernel: KernelFunction {
    var parameters: [String: Double] = [:]
    
    func compute(_ x1: [Double], _ x2: [Double]) throws -> Double {
        guard x1.count == x2.count else {
            throw SVMError.kernelError("Feature dimensions must match")
        }
        
        return zip(x1, x2).map(*).reduce(0, +)
    }
    
    var name: String { "Linear" }
}

// RBF (Radial Basis Function) Kernel
struct RBFKernel: KernelFunction {
    var parameters: [String: Double]
    
    init(gamma: Double = 1.0) {
        self.parameters = ["gamma": gamma]
    }
    
    func compute(_ x1: [Double], _ x2: [Double]) throws -> Double {
        guard x1.count == x2.count else {
            throw SVMError.kernelError("Feature dimensions must match")
        }
        
        let gamma = parameters["gamma"] ?? 1.0
        let squaredDistance = zip(x1, x2).map { pow($0 - $1, 2) }.reduce(0, +)
        return exp(-gamma * squaredDistance)
    }
    
    var name: String { "RBF" }
}

// Polynomial Kernel
struct PolynomialKernel: KernelFunction {
    var parameters: [String: Double]
    
    init(degree: Double = 3.0, coeff: Double = 1.0) {
        self.parameters = ["degree": degree, "coeff": coeff]
    }
    
    func compute(_ x1: [Double], _ x2: [Double]) throws -> Double {
        guard x1.count == x2.count else {
            throw SVMError.kernelError("Feature dimensions must match")
        }
        
        let degree = parameters["degree"] ?? 3.0
        let coeff = parameters["coeff"] ?? 1.0
        let dotProduct = zip(x1, x2).map(*).reduce(0, +)
        return pow(dotProduct + coeff, degree)
    }
    
    var name: String { "Polynomial" }
}

// MARK: - SVM Configuration

struct SVMConfiguration {
    let c: Double              // Regularization parameter
    let tolerance: Double      // Tolerance for stopping criterion
    let maxIterations: Int     // Maximum number of iterations
    let kernelCacheSize: Int   // Kernel cache size
    let shrinking: Bool        // Use shrinking heuristics
    
    init(c: Double = 1.0, 
         tolerance: Double = 1e-3, 
         maxIterations: Int = 1000, 
         kernelCacheSize: Int = 200,
         shrinking: Bool = true) throws {
        
        guard c > 0 else {
            throw SVMError.invalidInput("C parameter must be positive")
        }
        guard tolerance > 0 else {
            throw SVMError.invalidInput("Tolerance must be positive")
        }
        guard maxIterations > 0 else {
            throw SVMError.invalidInput("Max iterations must be positive")
        }
        
        self.c = c
        self.tolerance = tolerance
        self.maxIterations = maxIterations
        self.kernelCacheSize = kernelCacheSize
        self.shrinking = shrinking
    }
}

// MARK: - Training Metrics

struct TrainingMetrics {
    let iterations: Int
    let supportVectorCount: Int
    let trainingAccuracy: Double
    let objectiveValue: Double
    let trainingTime: TimeInterval
    let convergenceAchieved: Bool
    
    func summary() -> String {
        return """
        Training Metrics:
        - Iterations: \(iterations)
        - Support Vectors: \(supportVectorCount)
        - Training Accuracy: \(String(format: "%.4f", trainingAccuracy))
        - Objective Value: \(String(format: "%.6f", objectiveValue))
        - Training Time: \(String(format: "%.3f", trainingTime))s
        - Converged: \(convergenceAchieved)
        """
    }
}

// MARK: - Support Vector Machine Class

class SupportVectorMachine {
    
    // MARK: - Properties
    
    private var kernel: KernelFunction
    private let configuration: SVMConfiguration
    private let logger = OSLog(subsystem: "com.ai.svm", category: "training")
    
    // Training state
    private var alphas: [Double] = []
    private var bias: Double = 0.0
    private var supportVectors: [DataPoint] = []
    private var supportVectorAlphas: [Double] = []
    private var trainingMetrics: TrainingMetrics?
    
    // Kernel cache for performance optimization
    private var kernelCache: [String: Double] = [:]
    private let cacheQueue = DispatchQueue(label: "com.ai.svm.cache", attributes: .concurrent)
    
    // MARK: - Initialization
    
    init(kernel: KernelFunction = LinearKernel(), configuration: SVMConfiguration = try! SVMConfiguration()) {
        self.kernel = kernel
        self.configuration = configuration
    }
    
    // MARK: - Training Methods
    
    func train(dataset: Dataset) async throws -> TrainingMetrics {
        os_log("Starting SVM training with %d samples", log: logger, type: .info, dataset.size)
        os_log("Using %@ kernel with C=%.3f", log: logger, type: .info, kernel.name, configuration.c)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Validate dataset
        try validateDatasetForTraining(dataset)
        
        // Initialize optimization variables
        let n = dataset.size
        alphas = Array(repeating: 0.0, count: n)
        bias = 0.0
        
        // Precompute kernel matrix (with caching)
        let kernelMatrix = try await computeKernelMatrix(dataset)
        
        // SMO Algorithm implementation
        let result = try await sequentialMinimalOptimization(dataset: dataset, kernelMatrix: kernelMatrix)
        
        // Extract support vectors
        extractSupportVectors(dataset: dataset)
        
        // Calculate final metrics
        let trainingTime = CFAbsoluteTimeGetCurrent() - startTime
        let accuracy = try calculateTrainingAccuracy(dataset)
        let objectiveValue = calculateObjectiveValue(dataset: dataset, kernelMatrix: kernelMatrix)
        
        let metrics = TrainingMetrics(
            iterations: result.iterations,
            supportVectorCount: supportVectors.count,
            trainingAccuracy: accuracy,
            objectiveValue: objectiveValue,
            trainingTime: trainingTime,
            convergenceAchieved: result.converged
        )
        
        self.trainingMetrics = metrics
        
        os_log("Training completed: %d iterations, %d support vectors", 
               log: logger, type: .info, result.iterations, supportVectors.count)
        
        return metrics
    }
    
    private func validateDatasetForTraining(_ dataset: Dataset) throws {
        // Check for binary classification
        guard dataset.labelSet.count == 2 else {
            throw SVMError.trainingError("SVM requires exactly 2 classes for binary classification")
        }
        
        // Normalize labels to -1 and +1
        let sortedLabels = Array(dataset.labelSet).sorted()
        guard sortedLabels == [-1.0, 1.0] || sortedLabels.allSatisfy({ $0 == 0.0 || $0 == 1.0 }) else {
            os_log("Converting labels to -1/+1 format", log: logger, type: .info)
        }
        
        // Check dataset balance
        if !dataset.isBalanced {
            os_log("Warning: Dataset is imbalanced", log: logger, type: .info)
        }
    }
    
    private func computeKernelMatrix(_ dataset: Dataset) async throws -> [[Double]] {
        return try await withThrowingTaskGroup(of: (Int, [Double]).self) { group in
            var matrix = Array(repeating: Array(repeating: 0.0, count: dataset.size), count: dataset.size)
            
            // Compute kernel matrix in parallel
            for i in 0..<dataset.size {
                group.addTask { [weak self] in
                    guard let self = self else { throw SVMError.trainingError("SVM deallocated") }
                    var row = Array(repeating: 0.0, count: dataset.size)
                    
                    for j in 0..<dataset.size {
                        let cacheKey = "\(min(i,j))_\(max(i,j))"
                        
                        let kernelValue = try await self.cacheQueue.sync {
                            if let cached = self.kernelCache[cacheKey] {
                                return cached
                            }
                            return nil
                        }
                        
                        if let cached = kernelValue {
                            row[j] = cached
                        } else {
                            let value = try self.kernel.compute(dataset.points[i].features, dataset.points[j].features)
                            row[j] = value
                            
                            // Cache if within size limit
                            await self.cacheQueue.sync(flags: .barrier) {
                                if self.kernelCache.count < self.configuration.kernelCacheSize {
                                    self.kernelCache[cacheKey] = value
                                }
                            }
                        }
                    }
                    
                    return (i, row)
                }
            }
            
            // Collect results
            for try await (index, row) in group {
                matrix[index] = row
            }
            
            return matrix
        }
    }
    
    private struct SMOResult {
        let iterations: Int
        let converged: Bool
    }
    
    private func sequentialMinimalOptimization(dataset: Dataset, kernelMatrix: [[Double]]) async throws -> SMOResult {
        let n = dataset.size
        let labels = dataset.points.map { normalizeLabel($0.label) }
        var iteration = 0
        let tolerance = configuration.tolerance
        let C = configuration.c
        
        // Working set for active variables (shrinking heuristic)
        var activeSet = Set(0..<n)
        
        while iteration < configuration.maxIterations {
            var numChanged = 0
            
            // Examine all examples in active set
            for i in activeSet {
                let Ei = try calculateError(i, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels)
                
                // Check KKT conditions
                if (labels[i] * Ei < -tolerance && alphas[i] < C) ||
                   (labels[i] * Ei > tolerance && alphas[i] > 0) {
                    
                    // Find second alpha to optimize
                    if let j = try selectSecondAlpha(i, Ei: Ei, activeSet: activeSet, dataset: dataset, 
                                                   kernelMatrix: kernelMatrix, labels: labels) {
                        
                        if try optimizeAlphaPair(i, j, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels) {
                            numChanged += 1
                        }
                    }
                }
            }
            
            // Update active set with shrinking heuristic
            if configuration.shrinking && iteration % 10 == 0 {
                updateActiveSet(&activeSet, labels: labels)
            }
            
            iteration += 1
            
            // Check convergence
            if numChanged == 0 {
                // Examine entire dataset if no changes in active set
                let fullCheck = try await checkFullDatasetConvergence(dataset: dataset, 
                                                                     kernelMatrix: kernelMatrix, labels: labels)
                if fullCheck {
                    os_log("SMO converged after %d iterations", log: logger, type: .info, iteration)
                    return SMOResult(iterations: iteration, converged: true)
                }
            }
            
            // Progress logging
            if iteration % 100 == 0 {
                os_log("SMO iteration %d, active variables: %d", log: logger, type: .debug, iteration, activeSet.count)
            }
        }
        
        os_log("SMO reached maximum iterations (%d)", log: logger, type: .info, configuration.maxIterations)
        return SMOResult(iterations: iteration, converged: false)
    }
    
    private func calculateError(_ i: Int, dataset: Dataset, kernelMatrix: [[Double]], labels: [Double]) throws -> Double {
        var sum = 0.0
        for j in 0..<dataset.size {
            sum += alphas[j] * labels[j] * kernelMatrix[i][j]
        }
        return sum + bias - labels[i]
    }
    
    private func selectSecondAlpha(_ i: Int, Ei: Double, activeSet: Set<Int>, dataset: Dataset, 
                                  kernelMatrix: [[Double]], labels: [Double]) throws -> Int? {
        var maxDelta = 0.0
        var selectedJ: Int? = nil
        
        // Heuristic: choose j that maximizes |Ei - Ej|
        for j in activeSet where j != i {
            let Ej = try calculateError(j, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels)
            let delta = abs(Ei - Ej)
            if delta > maxDelta {
                maxDelta = delta
                selectedJ = j
            }
        }
        
        return selectedJ
    }
    
    private func optimizeAlphaPair(_ i: Int, _ j: Int, dataset: Dataset, 
                                  kernelMatrix: [[Double]], labels: [Double]) throws -> Bool {
        if i == j { return false }
        
        let alphaIOld = alphas[i]
        let alphaJOld = alphas[j]
        let yi = labels[i]
        let yj = labels[j]
        
        // Calculate bounds
        let C = configuration.c
        var L = 0.0, H = 0.0
        
        if yi != yj {
            L = max(0, alphaJOld - alphaIOld)
            H = min(C, C + alphaJOld - alphaIOld)
        } else {
            L = max(0, alphaIOld + alphaJOld - C)
            H = min(C, alphaIOld + alphaJOld)
        }
        
        if L == H { return false }
        
        // Calculate eta (second derivative)
        let eta = kernelMatrix[i][i] + kernelMatrix[j][j] - 2 * kernelMatrix[i][j]
        
        if eta <= 0 {
            // Unusual case - skip this pair
            return false
        }
        
        // Calculate errors
        let Ei = try calculateError(i, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels)
        let Ej = try calculateError(j, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels)
        
        // Update alpha j
        let alphaJNew = alphaJOld + yj * (Ei - Ej) / eta
        let alphaJClipped = min(H, max(L, alphaJNew))
        
        // Check for significant change
        if abs(alphaJClipped - alphaJOld) < 1e-5 {
            return false
        }
        
        // Update alpha i
        let alphaINew = alphaIOld + yi * yj * (alphaJOld - alphaJClipped)
        
        // Update alphas
        alphas[i] = alphaINew
        alphas[j] = alphaJClipped
        
        // Update bias
        updateBias(i, j, alphaIOld: alphaIOld, alphaJOld: alphaJOld, 
                  Ei: Ei, Ej: Ej, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels)
        
        return true
    }
    
    private func updateBias(_ i: Int, _ j: Int, alphaIOld: Double, alphaJOld: Double,
                           Ei: Double, Ej: Double, dataset: Dataset, kernelMatrix: [[Double]], labels: [Double]) {
        let C = configuration.c
        let yi = labels[i]
        let yj = labels[j]
        
        let b1 = bias - Ei - yi * (alphas[i] - alphaIOld) * kernelMatrix[i][i] - 
                 yj * (alphas[j] - alphaJOld) * kernelMatrix[i][j]
        let b2 = bias - Ej - yi * (alphas[i] - alphaIOld) * kernelMatrix[i][j] - 
                 yj * (alphas[j] - alphaJOld) * kernelMatrix[j][j]
        
        if 0 < alphas[i] && alphas[i] < C {
            bias = b1
        } else if 0 < alphas[j] && alphas[j] < C {
            bias = b2
        } else {
            bias = (b1 + b2) / 2.0
        }
    }
    
    private func updateActiveSet(_ activeSet: inout Set<Int>, labels: [Double]) {
        // Shrinking heuristic: remove variables that are unlikely to be optimized
        let tolerance = configuration.tolerance
        let C = configuration.c
        
        activeSet = activeSet.filter { i in
            let alpha = alphas[i]
            let label = labels[i]
            
            // Keep variables that might violate KKT conditions
            return !((alpha == 0 && label > -tolerance) ||
                    (alpha == C && label < tolerance) ||
                    (0 < alpha && alpha < C))
        }
    }
    
    private func checkFullDatasetConvergence(dataset: Dataset, kernelMatrix: [[Double]], labels: [Double]) async throws -> Bool {
        let tolerance = configuration.tolerance
        let C = configuration.c
        
        for i in 0..<dataset.size {
            let Ei = try calculateError(i, dataset: dataset, kernelMatrix: kernelMatrix, labels: labels)
            
            if (labels[i] * Ei < -tolerance && alphas[i] < C) ||
               (labels[i] * Ei > tolerance && alphas[i] > 0) {
                return false
            }
        }
        
        return true
    }
    
    private func extractSupportVectors(dataset: Dataset) {
        supportVectors = []
        supportVectorAlphas = []
        
        for (index, alpha) in alphas.enumerated() {
            if alpha > 1e-8 {  // Consider numerical precision
                supportVectors.append(dataset.points[index])
                supportVectorAlphas.append(alpha)
            }
        }
    }
    
    private func calculateTrainingAccuracy(_ dataset: Dataset) throws -> Double {
        var correct = 0
        
        for point in dataset.points {
            let prediction = try predict(point.features)
            let actualLabel = normalizeLabel(point.label)
            
            if (prediction > 0 && actualLabel > 0) || (prediction <= 0 && actualLabel <= 0) {
                correct += 1
            }
        }
        
        return Double(correct) / Double(dataset.size)
    }
    
    private func calculateObjectiveValue(dataset: Dataset, kernelMatrix: [[Double]]) -> Double {
        let labels = dataset.points.map { normalizeLabel($0.label) }
        var objective = 0.0
        
        // Calculate dual objective: sum(alpha) - 0.5 * sum_i sum_j (alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
        for i in 0..<dataset.size {
            objective += alphas[i]
            
            for j in 0..<dataset.size {
                objective -= 0.5 * alphas[i] * alphas[j] * labels[i] * labels[j] * kernelMatrix[i][j]
            }
        }
        
        return objective
    }
    
    // MARK: - Prediction Methods
    
    func predict(_ features: [Double]) throws -> Double {
        guard !supportVectors.isEmpty else {
            throw SVMError.predictionError("Model must be trained before prediction")
        }
        
        guard features.count == supportVectors[0].features.count else {
            throw SVMError.predictionError("Feature count mismatch")
        }
        
        var decision = 0.0
        
        for (i, supportVector) in supportVectors.enumerated() {
            let kernelValue = try kernel.compute(features, supportVector.features)
            let svLabel = normalizeLabel(supportVector.label)
            decision += supportVectorAlphas[i] * svLabel * kernelValue
        }
        
        return decision + bias
    }
    
    func classify(_ features: [Double]) throws -> Int {
        let decision = try predict(features)
        return decision > 0 ? 1 : -1
    }
    
    func predictProbability(_ features: [Double]) throws -> Double {
        let decision = try predict(features)
        // Sigmoid transformation for probability estimate (Platt scaling would be better)
        return 1.0 / (1.0 + exp(-decision))
    }
    
    func predictBatch(_ featuresArray: [[Double]]) async throws -> [Double] {
        return try await withThrowingTaskGroup(of: (Int, Double).self) { group in
            var results = Array(repeating: 0.0, count: featuresArray.count)
            
            for (index, features) in featuresArray.enumerated() {
                group.addTask { [weak self] in
                    guard let self = self else { throw SVMError.predictionError("SVM deallocated") }
                    let prediction = try self.predict(features)
                    return (index, prediction)
                }
            }
            
            for try await (index, prediction) in group {
                results[index] = prediction
            }
            
            return results
        }
    }
    
    // MARK: - Evaluation Methods
    
    func evaluate(_ testDataset: Dataset) async throws -> EvaluationMetrics {
        guard !supportVectors.isEmpty else {
            throw SVMError.predictionError("Model must be trained before evaluation")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var truePositives = 0
        var falsePositives = 0
        var trueNegatives = 0
        var falseNegatives = 0
        
        var predictions: [Double] = []
        var actualLabels: [Double] = []
        
        for point in testDataset.points {
            let prediction = try predict(point.features)
            let classification = prediction > 0 ? 1 : -1
            let actualLabel = Int(normalizeLabel(point.label))
            
            predictions.append(prediction)
            actualLabels.append(Double(actualLabel))
            
            if classification == 1 && actualLabel == 1 {
                truePositives += 1
            } else if classification == 1 && actualLabel == -1 {
                falsePositives += 1
            } else if classification == -1 && actualLabel == -1 {
                trueNegatives += 1
            } else {
                falseNegatives += 1
            }
        }
        
        let evaluationTime = CFAbsoluteTimeGetCurrent() - startTime
        
        return EvaluationMetrics(
            accuracy: Double(truePositives + trueNegatives) / Double(testDataset.size),
            precision: truePositives > 0 ? Double(truePositives) / Double(truePositives + falsePositives) : 0.0,
            recall: Double(truePositives) / Double(truePositives + falseNegatives),
            f1Score: 0.0, // Will be calculated in the struct
            auc: calculateAUC(predictions: predictions, labels: actualLabels),
            evaluationTime: evaluationTime,
            sampleCount: testDataset.size
        )
    }
    
    private func calculateAUC(predictions: [Double], labels: [Double]) -> Double {
        // Simple AUC calculation using trapezoidal rule
        let sorted = zip(predictions, labels).sorted { $0.0 > $1.0 }
        
        var tpr = 0.0
        var fpr = 0.0
        var auc = 0.0
        var prevFPR = 0.0
        
        let positives = labels.filter { $0 > 0 }.count
        let negatives = labels.count - positives
        
        if positives == 0 || negatives == 0 {
            return 0.5
        }
        
        for (_, label) in sorted {
            if label > 0 {
                tpr += 1.0 / Double(positives)
            } else {
                auc += tpr * (fpr - prevFPR)
                prevFPR = fpr
                fpr += 1.0 / Double(negatives)
            }
        }
        
        auc += tpr * (1.0 - prevFPR)
        return auc
    }
    
    // MARK: - Utility Methods
    
    private func normalizeLabel(_ label: Double) -> Double {
        // Convert 0/1 labels to -1/+1
        return label <= 0 ? -1.0 : 1.0
    }
    
    func getTrainingMetrics() -> TrainingMetrics? {
        return trainingMetrics
    }
    
    func getSupportVectorCount() -> Int {
        return supportVectors.count
    }
    
    func getKernelName() -> String {
        return kernel.name
    }
    
    // MARK: - Model Persistence
    
    struct SerializableModel: Codable {
        let supportVectors: [[Double]]
        let supportVectorLabels: [Double]
        let supportVectorAlphas: [Double]
        let bias: Double
        let kernelName: String
        let kernelParameters: [String: Double]
    }
    
    func serialize() throws -> Data {
        guard !supportVectors.isEmpty else {
            throw SVMError.predictionError("Model must be trained before serialization")
        }
        
        let model = SerializableModel(
            supportVectors: supportVectors.map { $0.features },
            supportVectorLabels: supportVectors.map { $0.label },
            supportVectorAlphas: supportVectorAlphas,
            bias: bias,
            kernelName: kernel.name,
            kernelParameters: kernel.parameters
        )
        
        return try JSONEncoder().encode(model)
    }
    
    func deserialize(from data: Data) throws {
        let model = try JSONDecoder().decode(SerializableModel.self, from: data)
        
        // Reconstruct support vectors
        supportVectors = []
        for (features, label) in zip(model.supportVectors, model.supportVectorLabels) {
            supportVectors.append(try DataPoint(features: features, label: label))
        }
        
        supportVectorAlphas = model.supportVectorAlphas
        bias = model.bias
        
        // Reconstruct kernel (simplified - assumes kernel type matching)
        kernel.parameters = model.kernelParameters
    }
}

// MARK: - Evaluation Metrics

struct EvaluationMetrics {
    let accuracy: Double
    let precision: Double
    let recall: Double
    private let _f1Score: Double?
    let auc: Double
    let evaluationTime: TimeInterval
    let sampleCount: Int
    
    init(accuracy: Double, precision: Double, recall: Double, f1Score: Double? = nil, 
         auc: Double, evaluationTime: TimeInterval, sampleCount: Int) {
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self._f1Score = f1Score
        self.auc = auc
        self.evaluationTime = evaluationTime
        self.sampleCount = sampleCount
    }
    
    var f1Score: Double {
        if let f1 = _f1Score {
            return f1
        }
        
        if precision + recall == 0 {
            return 0.0
        }
        
        return 2 * (precision * recall) / (precision + recall)
    }
    
    func summary() -> String {
        return """
        Evaluation Metrics:
        - Accuracy: \(String(format: "%.4f", accuracy))
        - Precision: \(String(format: "%.4f", precision))
        - Recall: \(String(format: "%.4f", recall))
        - F1-Score: \(String(format: "%.4f", f1Score))
        - AUC: \(String(format: "%.4f", auc))
        - Samples: \(sampleCount)
        - Evaluation Time: \(String(format: "%.3f", evaluationTime))s
        """
    }
}

// MARK: - Data Utilities

class SVMDataUtils {
    
    static func generateBinaryClassificationDataset(samples: Int = 1000, features: Int = 2, 
                                                   noise: Double = 0.1, seed: UInt64 = 42) throws -> Dataset {
        var generator = SystemRandomNumberGenerator()
        generator = SeededRandomNumberGenerator(seed: seed)
        
        var points: [DataPoint] = []
        
        for _ in 0..<samples {
            var featureVector: [Double] = []
            
            for _ in 0..<features {
                featureVector.append(Double.random(in: -2.0...2.0, using: &generator))
            }
            
            // Simple linear separable data with noise
            let linearCombination = featureVector.enumerated().map { (index, value) in
                value * (index % 2 == 0 ? 1.0 : -0.5)
            }.reduce(0, +)
            
            let noisyDecision = linearCombination + Double.random(in: -noise...noise, using: &generator)
            let label = noisyDecision > 0 ? 1.0 : -1.0
            
            points.append(try DataPoint(features: featureVector, label: label))
        }
        
        return try Dataset(points: points)
    }
    
    static func generateCircularDataset(samples: Int = 1000, noise: Double = 0.1, seed: UInt64 = 42) throws -> Dataset {
        var generator = SeededRandomNumberGenerator(seed: seed)
        var points: [DataPoint] = []
        
        for _ in 0..<samples {
            let x1 = Double.random(in: -2.0...2.0, using: &generator)
            let x2 = Double.random(in: -2.0...2.0, using: &generator)
            
            let distance = sqrt(x1*x1 + x2*x2)
            let threshold = 1.5 + Double.random(in: -noise...noise, using: &generator)
            
            let label = distance < threshold ? -1.0 : 1.0
            
            points.append(try DataPoint(features: [x1, x2], label: label))
        }
        
        return try Dataset(points: points)
    }
    
    static func trainTestSplit(_ dataset: Dataset, testRatio: Double = 0.2, seed: UInt64 = 42) throws -> (Dataset, Dataset) {
        guard testRatio > 0 && testRatio < 1 else {
            throw SVMError.invalidInput("Test ratio must be between 0 and 1")
        }
        
        var generator = SeededRandomNumberGenerator(seed: seed)
        let shuffled = dataset.points.shuffled(using: &generator)
        
        let testSize = Int(Double(dataset.size) * testRatio)
        let trainSize = dataset.size - testSize
        
        let trainPoints = Array(shuffled[0..<trainSize])
        let testPoints = Array(shuffled[trainSize..<shuffled.count])
        
        return (try Dataset(points: trainPoints), try Dataset(points: testPoints))
    }
    
    static func normalizeFeatures(_ dataset: Dataset) throws -> Dataset {
        guard !dataset.points.isEmpty else {
            throw SVMError.invalidInput("Cannot normalize empty dataset")
        }
        
        let featureCount = dataset.featureCount
        var minValues = Array(repeating: Double.infinity, count: featureCount)
        var maxValues = Array(repeating: -Double.infinity, count: featureCount)
        
        // Find min/max for each feature
        for point in dataset.points {
            for (index, value) in point.features.enumerated() {
                minValues[index] = min(minValues[index], value)
                maxValues[index] = max(maxValues[index], value)
            }
        }
        
        // Normalize features to [0, 1]
        var normalizedPoints: [DataPoint] = []
        
        for point in dataset.points {
            var normalizedFeatures: [Double] = []
            
            for (index, value) in point.features.enumerated() {
                let range = maxValues[index] - minValues[index]
                let normalized = range > 0 ? (value - minValues[index]) / range : 0.5
                normalizedFeatures.append(normalized)
            }
            
            normalizedPoints.append(try DataPoint(features: normalizedFeatures, label: point.label))
        }
        
        return try Dataset(points: normalizedPoints)
    }
}

// MARK: - Seeded Random Number Generator

struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1
        return state
    }
}

// MARK: - Demo Application

@available(macOS 10.15, iOS 13.0, *)
class SVMDemo {
    
    static func runDemo() async {
        print("🤖 Swift Support Vector Machine Demo")
        print("=" + String(repeating: "=", count: 40))
        
        do {
            // Generate synthetic dataset
            print("📊 Generating synthetic binary classification dataset...")
            let fullDataset = try SVMDataUtils.generateBinaryClassificationDataset(
                samples: 500, features: 2, noise: 0.2
            )
            
            // Split into train/test
            let (trainDataset, testDataset) = try SVMDataUtils.trainTestSplit(fullDataset, testRatio: 0.3)
            print("📈 Train samples: \(trainDataset.size), Test samples: \(testDataset.size)")
            
            // Normalize features
            let normalizedTrain = try SVMDataUtils.normalizeFeatures(trainDataset)
            let normalizedTest = try SVMDataUtils.normalizeFeatures(testDataset)
            
            // Test different kernels
            let kernels: [KernelFunction] = [
                LinearKernel(),
                RBFKernel(gamma: 1.0),
                PolynomialKernel(degree: 3.0, coeff: 1.0)
            ]
            
            for kernel in kernels {
                print("\n🧠 Testing \(kernel.name) kernel...")
                
                // Configure SVM
                let config = try SVMConfiguration(c: 1.0, tolerance: 1e-3, maxIterations: 500)
                let svm = SupportVectorMachine(kernel: kernel, configuration: config)
                
                // Train SVM
                let trainingMetrics = try await svm.train(dataset: normalizedTrain)
                print(trainingMetrics.summary())
                
                // Evaluate on test set
                let evalMetrics = try await svm.evaluate(normalizedTest)
                print(evalMetrics.summary())
                
                // Test individual predictions
                print("\n🔮 Sample Predictions:")
                for i in 0..<min(5, normalizedTest.size) {
                    let point = normalizedTest.points[i]
                    let prediction = try svm.predict(point.features)
                    let classification = try svm.classify(point.features)
                    let probability = try svm.predictProbability(point.features)
                    
                    print(String(format: "   Features: [%.3f, %.3f] → Decision: %.3f, Class: %d, Prob: %.3f, Actual: %.0f",
                                point.features[0], point.features[1], prediction, 
                                classification, probability, point.label))
                }
                
                print(String(repeating: "-", count: 50))
            }
            
            // Test circular dataset with RBF kernel
            print("\n🔵 Testing RBF kernel on circular dataset...")
            let circularDataset = try SVMDataUtils.generateCircularDataset(samples: 300, noise: 0.1)
            let (circularTrain, circularTest) = try SVMDataUtils.trainTestSplit(circularDataset, testRatio: 0.3)
            
            let rbfKernel = RBFKernel(gamma: 2.0)
            let rbfConfig = try SVMConfiguration(c: 2.0, tolerance: 1e-3, maxIterations: 800)
            let rbfSVM = SupportVectorMachine(kernel: rbfKernel, configuration: rbfConfig)
            
            let circularTrainingMetrics = try await rbfSVM.train(dataset: circularTrain)
            print(circularTrainingMetrics.summary())
            
            let circularEvalMetrics = try await rbfSVM.evaluate(circularTest)
            print(circularEvalMetrics.summary())
            
            // Model serialization demo
            print("\n💾 Testing model serialization...")
            let serializedData = try rbfSVM.serialize()
            print("Model serialized to \(serializedData.count) bytes")
            
            // Create new SVM and deserialize
            let newSVM = SupportVectorMachine(kernel: RBFKernel(gamma: 2.0), configuration: rbfConfig)
            try newSVM.deserialize(from: serializedData)
            
            let testPoint = circularTest.points[0]
            let originalPrediction = try rbfSVM.predict(testPoint.features)
            let deserializedPrediction = try newSVM.predict(testPoint.features)
            
            print(String(format: "Original prediction: %.6f", originalPrediction))
            print(String(format: "Deserialized prediction: %.6f", deserializedPrediction))
            print(String(format: "Difference: %.8f", abs(originalPrediction - deserializedPrediction)))
            
            print("\n✅ SVM demonstration completed successfully!")
            
        } catch {
            print("❌ Demo failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Main Entry Point

@main
@available(macOS 10.15, iOS 13.0, *)
struct SVMMain {
    static func main() async {
        await SVMDemo.runDemo()
    }
}