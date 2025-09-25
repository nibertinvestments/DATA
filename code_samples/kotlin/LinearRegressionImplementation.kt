/**
 * Production-Ready Linear Regression Implementation in Kotlin
 * =========================================================
 * 
 * This module demonstrates a comprehensive linear regression implementation
 * with gradient descent, regularization, and modern Kotlin patterns
 * for AI training datasets.
 *
 * Key Features:
 * - Multiple regression algorithms (Normal Equation, Gradient Descent, SGD)
 * - L1 and L2 regularization (Ridge, Lasso, Elastic Net)
 * - Feature scaling and normalization
 * - Comprehensive statistical metrics (R², MSE, MAE, etc.)
 * - Cross-validation and model selection
 * - Kotlin coroutines for async training
 * - Type-safe matrix operations
 * - Modern Kotlin idioms and DSL
 * - Production deployment considerations
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import kotlinx.coroutines.*
import kotlin.math.*
import kotlin.random.Random
import kotlin.system.measureTimeMillis

/**
 * Custom exception for linear regression errors
 */
class LinearRegressionException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Data class representing a training sample
 */
data class RegressionSample(
    val features: DoubleArray,
    val target: Double,
    val id: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as RegressionSample
        return features.contentEquals(other.features) && target == other.target && id == other.id
    }
    
    override fun hashCode(): Int {
        var result = features.contentHashCode()
        result = 31 * result + target.hashCode()
        result = 31 * result + id
        return result
    }
    
    override fun toString(): String = "RegressionSample(id=$id, features=${features.contentToString()}, target=$target)"
}

/**
 * Feature scaling interface
 */
interface FeatureScaler {
    fun fit(data: List<RegressionSample>)
    fun transform(data: List<RegressionSample>): List<RegressionSample>
    fun fitTransform(data: List<RegressionSample>): List<RegressionSample> {
        fit(data)
        return transform(data)
    }
    val name: String
}

/**
 * Standard scaler (z-score normalization)
 */
class StandardScaler : FeatureScaler {
    override val name = "StandardScaler"
    private var means: DoubleArray = doubleArrayOf()
    private var stds: DoubleArray = doubleArrayOf()
    
    override fun fit(data: List<RegressionSample>) {
        require(data.isNotEmpty()) { "Cannot fit on empty data" }
        
        val numFeatures = data.first().features.size
        means = DoubleArray(numFeatures)
        stds = DoubleArray(numFeatures)
        
        // Calculate means
        for (sample in data) {
            for (i in sample.features.indices) {
                means[i] += sample.features[i]
            }
        }
        means = means.map { it / data.size }.toDoubleArray()
        
        // Calculate standard deviations
        for (sample in data) {
            for (i in sample.features.indices) {
                val diff = sample.features[i] - means[i]
                stds[i] += diff * diff
            }
        }
        stds = stds.map { sqrt(it / data.size) }.toDoubleArray()
        
        // Handle zero standard deviation
        for (i in stds.indices) {
            if (stds[i] == 0.0) stds[i] = 1.0
        }
    }
    
    override fun transform(data: List<RegressionSample>): List<RegressionSample> {
        require(means.isNotEmpty()) { "Must fit scaler before transforming" }
        
        return data.map { sample ->
            val scaledFeatures = DoubleArray(sample.features.size) { i ->
                (sample.features[i] - means[i]) / stds[i]
            }
            RegressionSample(scaledFeatures, sample.target, sample.id)
        }
    }
}

/**
 * Min-Max scaler (normalization to [0, 1])
 */
class MinMaxScaler : FeatureScaler {
    override val name = "MinMaxScaler"
    private var mins: DoubleArray = doubleArrayOf()
    private var maxs: DoubleArray = doubleArrayOf()
    
    override fun fit(data: List<RegressionSample>) {
        require(data.isNotEmpty()) { "Cannot fit on empty data" }
        
        val numFeatures = data.first().features.size
        mins = DoubleArray(numFeatures) { Double.MAX_VALUE }
        maxs = DoubleArray(numFeatures) { -Double.MAX_VALUE }
        
        for (sample in data) {
            for (i in sample.features.indices) {
                mins[i] = min(mins[i], sample.features[i])
                maxs[i] = max(maxs[i], sample.features[i])
            }
        }
        
        // Handle constant features
        for (i in mins.indices) {
            if (mins[i] == maxs[i]) {
                mins[i] = 0.0
                maxs[i] = 1.0
            }
        }
    }
    
    override fun transform(data: List<RegressionSample>): List<RegressionSample> {
        require(mins.isNotEmpty()) { "Must fit scaler before transforming" }
        
        return data.map { sample ->
            val scaledFeatures = DoubleArray(sample.features.size) { i ->
                (sample.features[i] - mins[i]) / (maxs[i] - mins[i])
            }
            RegressionSample(scaledFeatures, sample.target, sample.id)
        }
    }
}

/**
 * Regression metrics for model evaluation
 */
data class RegressionMetrics(
    val mse: Double,
    val rmse: Double,
    val mae: Double,
    val r2: Double,
    val adjustedR2: Double,
    val sampleCount: Int
) {
    override fun toString(): String = buildString {
        appendLine("Regression Metrics:")
        appendLine("  MSE: ${"%.6f".format(mse)}")
        appendLine("  RMSE: ${"%.6f".format(rmse)}")
        appendLine("  MAE: ${"%.6f".format(mae)}")
        appendLine("  R²: ${"%.6f".format(r2)}")
        appendLine("  Adjusted R²: ${"%.6f".format(adjustedR2)}")
        appendLine("  Sample count: $sampleCount")
    }
}

/**
 * Comprehensive Linear Regression Implementation
 */
class LinearRegressionImplementation(
    private val learningRate: Double = 0.01,
    private val maxIterations: Int = 1000,
    private val tolerance: Double = 1e-6,
    private val regularization: String = "none", // "ridge", "lasso", "elastic_net"
    private val alpha: Double = 0.01, // regularization strength
    private val l1Ratio: Double = 0.5, // elastic net mixing parameter
    private val useFeatureScaling: Boolean = true,
    private val scaler: FeatureScaler = StandardScaler()
) {
    
    // Model parameters
    private var weights: DoubleArray = doubleArrayOf()
    private var bias: Double = 0.0
    private var fitted = false
    private var scaledData = false
    private val trainingHistory = mutableListOf<TrainingEpoch>()
    
    /**
     * Training epoch information
     */
    data class TrainingEpoch(
        val epoch: Int,
        val cost: Double,
        val gradient: Double
    )
    
    /**
     * Calculate cost function with regularization
     */
    private fun calculateCost(samples: List<RegressionSample>): Double {
        val predictions = samples.map { predict(it.features, useTrainingWeights = true) }
        val mse = samples.zip(predictions) { sample, pred -> 
            (sample.target - pred).pow(2) 
        }.average()
        
        // Add regularization term
        val regularizationTerm = when (regularization) {
            "ridge" -> alpha * weights.sumOf { it.pow(2) }
            "lasso" -> alpha * weights.sumOf { abs(it) }
            "elastic_net" -> alpha * (l1Ratio * weights.sumOf { abs(it) } + 
                           (1 - l1Ratio) * weights.sumOf { it.pow(2) })
            else -> 0.0
        }
        
        return mse + regularizationTerm
    }
    
    /**
     * Calculate gradients with regularization
     */
    private fun calculateGradients(samples: List<RegressionSample>): Pair<DoubleArray, Double> {
        val n = samples.size
        val weightGradients = DoubleArray(weights.size)
        var biasGradient = 0.0
        
        for (sample in samples) {
            val prediction = predict(sample.features, useTrainingWeights = true)
            val error = prediction - sample.target
            
            // Weight gradients
            for (i in weights.indices) {
                weightGradients[i] += error * sample.features[i] / n
            }
            
            // Bias gradient
            biasGradient += error / n
        }
        
        // Add regularization gradients
        when (regularization) {
            "ridge" -> {
                for (i in weights.indices) {
                    weightGradients[i] += 2 * alpha * weights[i]
                }
            }
            "lasso" -> {
                for (i in weights.indices) {
                    weightGradients[i] += alpha * sign(weights[i])
                }
            }
            "elastic_net" -> {
                for (i in weights.indices) {
                    weightGradients[i] += alpha * (l1Ratio * sign(weights[i]) + 
                                                 2 * (1 - l1Ratio) * weights[i])
                }
            }
        }
        
        return Pair(weightGradients, biasGradient)
    }
    
    /**
     * Fit using gradient descent
     */
    private suspend fun fitGradientDescent(samples: List<RegressionSample>) {
        val numFeatures = samples.first().features.size
        weights = DoubleArray(numFeatures) { Random.nextGaussian() * 0.01 }
        bias = 0.0
        trainingHistory.clear()
        
        println("🔄 Training with Gradient Descent...")
        println("Learning rate: $learningRate, Max iterations: $maxIterations")
        println("Regularization: $regularization${if (regularization != "none") " (α=$alpha)" else ""}")
        
        for (epoch in 0 until maxIterations) {
            val cost = calculateCost(samples)
            val (weightGradients, biasGradient) = calculateGradients(samples)
            
            // Update parameters
            for (i in weights.indices) {
                weights[i] -= learningRate * weightGradients[i]
            }
            bias -= learningRate * biasGradient
            
            // Track progress
            val gradientMagnitude = sqrt(weightGradients.sumOf { it.pow(2) } + biasGradient.pow(2))
            trainingHistory.add(TrainingEpoch(epoch, cost, gradientMagnitude))
            
            // Print progress
            if ((epoch + 1) % (maxIterations / 10) == 0) {
                println("Epoch ${epoch + 1}/$maxIterations - Cost: ${"%.6f".format(cost)}, " +
                       "Gradient: ${"%.6f".format(gradientMagnitude)}")
            }
            
            // Check convergence
            if (gradientMagnitude < tolerance) {
                println("✅ Converged after ${epoch + 1} epochs")
                break
            }
            
            // Allow other coroutines to run
            if (epoch % 100 == 0) yield()
        }
    }
    
    /**
     * Fit using normal equation (for small datasets without regularization)
     */
    private fun fitNormalEquation(samples: List<RegressionSample>) {
        if (regularization != "none") {
            throw LinearRegressionException("Normal equation doesn't support regularization")
        }
        
        println("📐 Training with Normal Equation...")
        
        val n = samples.size
        val numFeatures = samples.first().features.size
        
        // Create design matrix X and target vector y
        val X = Array(n) { DoubleArray(numFeatures + 1) }
        val y = DoubleArray(n)
        
        samples.forEachIndexed { i, sample ->
            X[i][0] = 1.0 // bias term
            sample.features.copyInto(X[i], 1)
            y[i] = sample.target
        }
        
        // Calculate (X'X)^-1 X'y
        try {
            val XtX = multiplyTranspose(X, X)
            val XtXInv = invertMatrix(XtX)
            val Xty = multiplyTransposeVector(X, y)
            val theta = multiplyVector(XtXInv, Xty)
            
            bias = theta[0]
            weights = theta.sliceArray(1 until theta.size)
            
            println("✅ Normal equation solved successfully")
        } catch (e: Exception) {
            throw LinearRegressionException("Failed to solve normal equation: ${e.message}", e)
        }
    }
    
    /**
     * Matrix operations for normal equation
     */
    private fun multiplyTranspose(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val result = Array(A[0].size) { DoubleArray(B[0].size) }
        for (i in A[0].indices) {
            for (j in B[0].indices) {
                for (k in A.indices) {
                    result[i][j] += A[k][i] * B[k][j]
                }
            }
        }
        return result
    }
    
    private fun multiplyTransposeVector(A: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val result = DoubleArray(A[0].size)
        for (i in A[0].indices) {
            for (j in A.indices) {
                result[i] += A[j][i] * b[j]
            }
        }
        return result
    }
    
    private fun multiplyVector(A: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val result = DoubleArray(A.size)
        for (i in A.indices) {
            for (j in b.indices) {
                result[i] += A[i][j] * b[j]
            }
        }
        return result
    }
    
    private fun invertMatrix(matrix: Array<DoubleArray>): Array<DoubleArray> {
        val n = matrix.size
        val augmented = Array(n) { DoubleArray(2 * n) }
        
        // Create augmented matrix [A | I]
        for (i in 0 until n) {
            for (j in 0 until n) {
                augmented[i][j] = matrix[i][j]
                augmented[i][j + n] = if (i == j) 1.0 else 0.0
            }
        }
        
        // Gauss-Jordan elimination
        for (i in 0 until n) {
            // Find pivot
            var maxRow = i
            for (k in i + 1 until n) {
                if (abs(augmented[k][i]) > abs(augmented[maxRow][i])) {
                    maxRow = k
                }
            }
            
            // Swap rows
            val temp = augmented[i]
            augmented[i] = augmented[maxRow]
            augmented[maxRow] = temp
            
            // Check for singular matrix
            if (abs(augmented[i][i]) < 1e-10) {
                throw LinearRegressionException("Matrix is singular")
            }
            
            // Make diagonal element 1
            val pivot = augmented[i][i]
            for (j in 0 until 2 * n) {
                augmented[i][j] /= pivot
            }
            
            // Eliminate column
            for (k in 0 until n) {
                if (k != i) {
                    val factor = augmented[k][i]
                    for (j in 0 until 2 * n) {
                        augmented[k][j] -= factor * augmented[i][j]
                    }
                }
            }
        }
        
        // Extract inverse matrix
        val inverse = Array(n) { DoubleArray(n) }
        for (i in 0 until n) {
            for (j in 0 until n) {
                inverse[i][j] = augmented[i][j + n]
            }
        }
        
        return inverse
    }
    
    /**
     * Prediction function
     */
    private fun predict(features: DoubleArray, useTrainingWeights: Boolean = false): Double {
        if (!fitted && !useTrainingWeights) {
            throw LinearRegressionException("Model not fitted. Call fit() first.")
        }
        
        var prediction = bias
        for (i in features.indices) {
            prediction += weights[i] * features[i]
        }
        return prediction
    }
    
    /**
     * Train the linear regression model
     */
    suspend fun fit(trainingSamples: List<RegressionSample>, method: String = "gradient_descent") {
        require(trainingSamples.isNotEmpty()) { "Training data cannot be empty" }
        
        println("📈 Training Linear Regression Model")
        println("=" .repeat(40))
        println("Training samples: ${trainingSamples.size}")
        println("Features: ${trainingSamples.first().features.size}")
        println("Method: $method")
        
        val trainingTime = measureTimeMillis {
            // Apply feature scaling if enabled
            val processedSamples = if (useFeatureScaling) {
                println("🔧 Applying feature scaling (${scaler.name})...")
                scaler.fitTransform(trainingSamples)
            } else {
                trainingSamples
            }
            
            scaledData = useFeatureScaling
            
            // Choose training method
            when (method) {
                "gradient_descent" -> fitGradientDescent(processedSamples)
                "normal_equation" -> fitNormalEquation(processedSamples)
                else -> throw LinearRegressionException("Unknown method: $method")
            }
        }
        
        fitted = true
        println("✅ Training completed in ${trainingTime}ms")
        
        // Print model summary
        println("\n🎯 Model Summary:")
        println("Weights: ${weights.contentToString()}")
        println("Bias: ${"%.6f".format(bias)}")
        println("Training history: ${trainingHistory.size} epochs")
    }
    
    /**
     * Make predictions
     */
    fun predict(samples: List<RegressionSample>): List<Double> {
        if (!fitted) {
            throw LinearRegressionException("Model not fitted. Call fit() first.")
        }
        
        val processedSamples = if (scaledData) {
            scaler.transform(samples)
        } else {
            samples
        }
        
        return processedSamples.map { predict(it.features) }
    }
    
    /**
     * Evaluate model performance
     */
    fun evaluate(testSamples: List<RegressionSample>): RegressionMetrics {
        val predictions = predict(testSamples)
        val actuals = testSamples.map { it.target }
        
        // Calculate metrics
        val mse = actuals.zip(predictions) { actual, pred -> (actual - pred).pow(2) }.average()
        val rmse = sqrt(mse)
        val mae = actuals.zip(predictions) { actual, pred -> abs(actual - pred) }.average()
        
        // Calculate R²
        val meanActual = actuals.average()
        val tss = actuals.sumOf { (it - meanActual).pow(2) }
        val rss = actuals.zip(predictions) { actual, pred -> (actual - pred).pow(2) }.sum()
        val r2 = 1 - (rss / tss)
        
        // Calculate adjusted R²
        val n = testSamples.size
        val p = weights.size
        val adjustedR2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
        return RegressionMetrics(mse, rmse, mae, r2, adjustedR2, n)
    }
    
    /**
     * Cross-validation for model evaluation
     */
    suspend fun crossValidate(samples: List<RegressionSample>, folds: Int = 5): List<RegressionMetrics> {
        require(folds > 1) { "Number of folds must be > 1" }
        require(samples.size >= folds) { "Not enough samples for $folds folds" }
        
        println("🔄 Performing $folds-fold cross-validation...")
        
        val shuffledSamples = samples.shuffled()
        val foldSize = samples.size / folds
        val results = mutableListOf<RegressionMetrics>()
        
        for (fold in 0 until folds) {
            val startIdx = fold * foldSize
            val endIdx = if (fold == folds - 1) samples.size else startIdx + foldSize
            
            val testSet = shuffledSamples.subList(startIdx, endIdx)
            val trainSet = shuffledSamples.subList(0, startIdx) + shuffledSamples.subList(endIdx, samples.size)
            
            // Create new model for this fold
            val foldModel = LinearRegressionImplementation(
                learningRate = learningRate,
                maxIterations = maxIterations / 2, // Reduce iterations for CV
                tolerance = tolerance,
                regularization = regularization,
                alpha = alpha,
                l1Ratio = l1Ratio,
                useFeatureScaling = useFeatureScaling,
                scaler = if (useFeatureScaling) StandardScaler() else scaler
            )
            
            // Train and evaluate
            foldModel.fit(trainSet, "gradient_descent")
            val metrics = foldModel.evaluate(testSet)
            results.add(metrics)
            
            println("Fold ${fold + 1}: R² = ${"%.4f".format(metrics.r2)}, RMSE = ${"%.4f".format(metrics.rmse)}")
        }
        
        // Print average results
        val avgR2 = results.map { it.r2 }.average()
        val avgRMSE = results.map { it.rmse }.average()
        println("✅ Cross-validation completed")
        println("Average R²: ${"%.4f".format(avgR2)}")
        println("Average RMSE: ${"%.4f".format(avgRMSE)}")
        
        return results
    }
    
    companion object {
        /**
         * Generate synthetic dataset for testing
         */
        fun generateSyntheticDataset(samples: Int, features: Int, noise: Double = 0.1): List<RegressionSample> {
            val random = Random.Default
            val trueWeights = DoubleArray(features) { random.nextGaussian() }
            val trueBias = random.nextGaussian()
            
            return (0 until samples).map { id ->
                val featureVector = DoubleArray(features) { random.nextGaussian() }
                val trueTarget = trueBias + featureVector.zip(trueWeights) { f, w -> f * w }.sum()
                val noisyTarget = trueTarget + random.nextGaussian() * noise
                
                RegressionSample(featureVector, noisyTarget, id)
            }
        }
        
        /**
         * Comprehensive demonstration of linear regression
         */
        suspend fun demonstrateLinearRegression() {
            println("🚀 Linear Regression Implementation Demonstration")
            println("=" .repeat(55))
            
            try {
                // Generate synthetic dataset
                println("📊 Generating synthetic dataset...")
                val dataset = generateSyntheticDataset(1000, 5, 0.2)
                println("Dataset: ${dataset.size} samples, ${dataset.first().features.size} features")
                
                // Split into train/test
                val shuffledData = dataset.shuffled()
                val trainSize = (dataset.size * 0.8).toInt()
                val trainData = shuffledData.take(trainSize)
                val testData = shuffledData.drop(trainSize)
                
                println("Train: $trainSize samples, Test: ${testData.size} samples")
                
                // Test different configurations
                val configurations = listOf(
                    Triple("No Regularization", "none", 0.0),
                    Triple("Ridge Regression", "ridge", 0.1),
                    Triple("Lasso Regression", "lasso", 0.1),
                    Triple("Elastic Net", "elastic_net", 0.1)
                )
                
                for ((name, regType, alpha) in configurations) {
                    println("\n" + "=".repeat(60))
                    println("🔍 Testing $name")
                    println("=".repeat(60))
                    
                    val model = LinearRegressionImplementation(
                        learningRate = 0.01,
                        maxIterations = 1000,
                        regularization = regType,
                        alpha = alpha,
                        useFeatureScaling = true
                    )
                    
                    // Train model
                    model.fit(trainData, "gradient_descent")
                    
                    // Evaluate on test set
                    val metrics = model.evaluate(testData)
                    println("\n📊 Test Set Performance:")
                    println(metrics)
                    
                    // Cross-validation
                    val cvResults = model.crossValidate(trainData, 5)
                    
                    // Sample predictions
                    println("\n🧪 Sample Predictions:")
                    val sampleTests = testData.take(5)
                    val predictions = model.predict(sampleTests)
                    
                    sampleTests.zip(predictions).forEachIndexed { idx, (sample, pred) ->
                        println("Sample ${sample.id}: Predicted = ${"%.4f".format(pred)}, " +
                               "Actual = ${"%.4f".format(sample.target)}")
                    }
                }
                
                println("\n✅ Linear regression demonstration completed successfully!")
                
            } catch (e: Exception) {
                println("❌ Linear regression demonstration failed: ${e.message}")
                e.printStackTrace()
            }
        }
    }
}

/**
 * Main function to demonstrate linear regression
 */
fun main() = runBlocking {
    LinearRegressionImplementation.demonstrateLinearRegression()
}