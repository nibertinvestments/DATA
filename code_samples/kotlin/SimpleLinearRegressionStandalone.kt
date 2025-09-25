/**
 * Production-Ready Linear Regression Implementation in Kotlin (Standalone)
 * =======================================================================
 * 
 * Simplified version that demonstrates linear regression without external dependencies.
 * This implementation focuses on core ML algorithms and Kotlin language features.
 */

import kotlin.math.*
import kotlin.random.Random

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
 * Regression metrics for model evaluation
 */
data class RegressionMetrics(
    val mse: Double,
    val rmse: Double,
    val mae: Double,
    val r2: Double,
    val sampleCount: Int
) {
    override fun toString(): String = buildString {
        appendLine("Regression Metrics:")
        appendLine("  MSE: ${"%.6f".format(mse)}")
        appendLine("  RMSE: ${"%.6f".format(rmse)}")
        appendLine("  MAE: ${"%.6f".format(mae)}")
        appendLine("  R²: ${"%.6f".format(r2)}")
        appendLine("  Sample count: $sampleCount")
    }
}

/**
 * Simple Linear Regression Implementation
 */
class SimpleLinearRegression(
    private val learningRate: Double = 0.01,
    private val maxIterations: Int = 1000,
    private val tolerance: Double = 1e-6
) {
    
    private var weights: DoubleArray = doubleArrayOf()
    private var bias: Double = 0.0
    private var fitted = false
    
    /**
     * Calculate cost function (MSE)
     */
    private fun calculateCost(samples: List<RegressionSample>): Double {
        val predictions = samples.map { predict(it.features) }
        return samples.zip(predictions) { sample, pred -> 
            (sample.target - pred).pow(2) 
        }.average()
    }
    
    /**
     * Calculate gradients
     */
    private fun calculateGradients(samples: List<RegressionSample>): Pair<DoubleArray, Double> {
        val n = samples.size
        val weightGradients = DoubleArray(weights.size)
        var biasGradient = 0.0
        
        for (sample in samples) {
            val prediction = predict(sample.features)
            val error = prediction - sample.target
            
            for (i in weights.indices) {
                weightGradients[i] += error * sample.features[i] / n
            }
            biasGradient += error / n
        }
        
        return Pair(weightGradients, biasGradient)
    }
    
    /**
     * Prediction function
     */
    private fun predict(features: DoubleArray): Double {
        if (weights.isEmpty()) {
            // During training, weights might not be initialized yet
            return 0.0
        }
        
        var prediction = bias
        for (i in features.indices) {
            prediction += weights[i] * features[i]
        }
        return prediction
    }
    
    /**
     * Train using gradient descent
     */
    fun fit(trainingSamples: List<RegressionSample>) {
        require(trainingSamples.isNotEmpty()) { "Training data cannot be empty" }
        
        println("📈 Training Linear Regression Model")
        println("=" .repeat(40))
        println("Training samples: ${trainingSamples.size}")
        println("Features: ${trainingSamples.first().features.size}")
        println("Learning rate: $learningRate")
        println("Max iterations: $maxIterations")
        
        val numFeatures = trainingSamples.first().features.size
        weights = DoubleArray(numFeatures) { Random.nextDouble(-0.01, 0.01) }
        bias = 0.0
        
        var previousCost = Double.MAX_VALUE
        
        println("\n🔄 Training with Gradient Descent...")
        
        for (epoch in 0 until maxIterations) {
            val cost = calculateCost(trainingSamples)
            val (weightGradients, biasGradient) = calculateGradients(trainingSamples)
            
            // Update parameters
            for (i in weights.indices) {
                weights[i] -= learningRate * weightGradients[i]
            }
            bias -= learningRate * biasGradient
            
            // Print progress
            if ((epoch + 1) % (maxIterations / 10) == 0) {
                println("Epoch ${epoch + 1}/$maxIterations - Cost: ${"%.6f".format(cost)}")
            }
            
            // Check convergence
            if (abs(previousCost - cost) < tolerance) {
                println("✅ Converged after ${epoch + 1} epochs")
                break
            }
            
            previousCost = cost
        }
        
        fitted = true
        println("\n✅ Training completed!")
        println("Final weights: ${weights.contentToString()}")
        println("Final bias: ${"%.6f".format(bias)}")
    }
    
    /**
     * Make predictions
     */
    fun predict(samples: List<RegressionSample>): List<Double> {
        if (!fitted) throw IllegalStateException("Model not fitted. Call fit() first.")
        return samples.map { predictSingle(it.features) }
    }
    
    /**
     * Public prediction function for single sample
     */
    private fun predictSingle(features: DoubleArray): Double {
        var prediction = bias
        for (i in features.indices) {
            prediction += weights[i] * features[i]
        }
        return prediction
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
        
        return RegressionMetrics(mse, rmse, mae, r2, testSamples.size)
    }
    
    companion object {
        /**
         * Generate synthetic dataset for testing
         */
        fun generateSyntheticDataset(samples: Int, features: Int, noise: Double = 0.1): List<RegressionSample> {
            val random = Random.Default
            val trueWeights = DoubleArray(features) { random.nextDouble(-2.0, 2.0) }
            val trueBias = random.nextDouble(-1.0, 1.0)
            
            return (0 until samples).map { id ->
                val featureVector = DoubleArray(features) { random.nextDouble(-2.0, 2.0) }
                val trueTarget = trueBias + featureVector.zip(trueWeights) { f, w -> f * w }.sum()
                val noisyTarget = trueTarget + random.nextDouble(-noise, noise)
                
                RegressionSample(featureVector, noisyTarget, id)
            }
        }
        
        /**
         * Demonstrate linear regression
         */
        fun demonstrateLinearRegression() {
            println("🚀 Linear Regression Implementation Demonstration")
            println("=" .repeat(55))
            
            try {
                // Generate synthetic dataset
                println("📊 Generating synthetic dataset...")
                val dataset = generateSyntheticDataset(1000, 3, 0.2)
                println("Dataset: ${dataset.size} samples, ${dataset.first().features.size} features")
                
                // Split into train/test
                val shuffledData = dataset.shuffled()
                val trainSize = (dataset.size * 0.8).toInt()
                val trainData = shuffledData.take(trainSize)
                val testData = shuffledData.drop(trainSize)
                
                println("Train: $trainSize samples, Test: ${testData.size} samples")
                
                // Create and train model
                val model = SimpleLinearRegression(
                    learningRate = 0.01,
                    maxIterations = 1000,
                    tolerance = 1e-6
                )
                
                model.fit(trainData)
                
                // Evaluate on test set
                val metrics = model.evaluate(testData)
                println("\n📊 Test Set Performance:")
                println(metrics)
                
                // Sample predictions
                println("\n🧪 Sample Predictions:")
                val sampleTests = testData.take(5)
                val predictions = model.predict(sampleTests)
                
                sampleTests.zip(predictions).forEach { (sample, pred) ->
                    println("Sample ${sample.id}: Predicted = ${"%.4f".format(pred)}, " +
                           "Actual = ${"%.4f".format(sample.target)}, " +
                           "Error = ${"%.4f".format(abs(pred - sample.target))}")
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
fun main() {
    SimpleLinearRegression.demonstrateLinearRegression()
}