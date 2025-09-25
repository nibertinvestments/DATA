/**
 * Production-Ready Naive Bayes Implementation in Kotlin
 * ====================================================
 * 
 * This module demonstrates a comprehensive Naive Bayes classifier
 * with Gaussian, Multinomial, and Bernoulli variants using modern
 * Kotlin patterns for AI training datasets.
 *
 * Key Features:
 * - Multiple Naive Bayes variants (Gaussian, Multinomial, Bernoulli)
 * - Laplace smoothing for numerical stability
 * - Incremental learning support
 * - Feature selection and importance scoring
 * - Cross-validation and model evaluation
 * - Type-safe sealed classes for variants
 * - Kotlin coroutines for parallel processing
 * - Comprehensive statistical analysis
 * - Memory-efficient probability calculations
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import kotlinx.coroutines.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Custom exception for Naive Bayes errors
 */
class NaiveBayesException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Data class representing a classification sample
 */
data class ClassificationSample(
    val features: DoubleArray,
    val label: String,
    val id: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ClassificationSample
        return features.contentEquals(other.features) && label == other.label && id == other.id
    }
    
    override fun hashCode(): Int {
        var result = features.contentHashCode()
        result = 31 * result + label.hashCode()
        result = 31 * result + id
        return result
    }
    
    override fun toString(): String = 
        "ClassificationSample(id=$id, features=${features.contentToString()}, label='$label')"
}

/**
 * Sealed class for different Naive Bayes variants
 */
sealed class NaiveBayesVariant {
    object Gaussian : NaiveBayesVariant()
    object Multinomial : NaiveBayesVariant()
    object Bernoulli : NaiveBayesVariant()
    
    val name: String
        get() = when (this) {
            is Gaussian -> "Gaussian"
            is Multinomial -> "Multinomial"
            is Bernoulli -> "Bernoulli"
        }
}

/**
 * Feature statistics for each class
 */
data class FeatureStatistics(
    val mean: Double = 0.0,
    val variance: Double = 0.0,
    val count: Int = 0,
    val sum: Double = 0.0
) {
    fun update(value: Double): FeatureStatistics {
        val newCount = count + 1
        val newSum = sum + value
        val newMean = newSum / newCount
        val delta = value - mean
        val newVariance = if (newCount > 1) {
            ((count * variance + delta * (value - newMean)) / newCount)
        } else {
            0.0
        }
        return FeatureStatistics(newMean, newVariance, newCount, newSum)
    }
}

/**
 * Class information containing statistics and probabilities
 */
data class ClassInfo(
    val className: String,
    val featureStats: MutableList<FeatureStatistics> = mutableListOf(),
    val sampleCount: Int = 0,
    val logPrior: Double = 0.0
) {
    fun updateSampleCount(newCount: Int): ClassInfo = 
        copy(sampleCount = newCount)
    
    fun updateLogPrior(totalSamples: Int, smoothing: Double = 1.0): ClassInfo = 
        copy(logPrior = ln((sampleCount + smoothing) / (totalSamples + smoothing * 2)))
}

/**
 * Naive Bayes evaluation metrics
 */
data class ClassificationMetrics(
    val accuracy: Double,
    val precision: Map<String, Double>,
    val recall: Map<String, Double>,
    val f1Score: Map<String, Double>,
    val macroPrecision: Double,
    val macroRecall: Double,
    val macroF1: Double,
    val confusionMatrix: Map<String, Map<String, Int>>,
    val sampleCount: Int
) {
    override fun toString(): String = buildString {
        appendLine("Classification Metrics:")
        appendLine("  Accuracy: ${"%.6f".format(accuracy)}")
        appendLine("  Macro Precision: ${"%.6f".format(macroPrecision)}")
        appendLine("  Macro Recall: ${"%.6f".format(macroRecall)}")
        appendLine("  Macro F1-Score: ${"%.6f".format(macroF1)}")
        appendLine("  Sample count: $sampleCount")
        appendLine()
        appendLine("Per-class Metrics:")
        precision.keys.sorted().forEach { className ->
            appendLine("  Class '$className':")
            appendLine("    Precision: ${"%.6f".format(precision[className] ?: 0.0)}")
            appendLine("    Recall: ${"%.6f".format(recall[className] ?: 0.0)}")
            appendLine("    F1-Score: ${"%.6f".format(f1Score[className] ?: 0.0)}")
        }
    }
}

/**
 * Prediction result with probabilities
 */
data class PredictionResult(
    val predictedClass: String,
    val probability: Double,
    val classProbabilities: Map<String, Double>
) {
    override fun toString(): String = 
        "PredictionResult(class='$predictedClass', prob=${"%.4f".format(probability)})"
}

/**
 * Comprehensive Naive Bayes Classifier Implementation
 */
class NaiveBayesImplementation(
    private val variant: NaiveBayesVariant = NaiveBayesVariant.Gaussian,
    private val smoothing: Double = 1.0, // Laplace smoothing
    private val priors: Map<String, Double>? = null // Custom class priors
) {
    
    private val classInfoMap = mutableMapOf<String, ClassInfo>()
    private var totalSamples = 0
    private var numFeatures = 0
    private var fitted = false
    private val featureNames = mutableListOf<String>()
    
    /**
     * Calculate Gaussian probability density
     */
    private fun gaussianProbability(x: Double, mean: Double, variance: Double): Double {
        if (variance <= 0.0) return if (x == mean) 1.0 else 1e-9
        
        val coefficient = 1.0 / sqrt(2 * PI * variance)
        val exponent = -(x - mean).pow(2) / (2 * variance)
        return coefficient * exp(exponent)
    }
    
    /**
     * Calculate multinomial log probability
     */
    private fun multinomialLogProbability(x: Double, classStats: FeatureStatistics, 
                                        featureIndex: Int, totalFeatures: Int): Double {
        val featureCount = classStats.sum
        val totalCount = classStats.count * totalFeatures
        val probability = (featureCount + smoothing) / (totalCount + smoothing * totalFeatures)
        return x * ln(probability)
    }
    
    /**
     * Calculate Bernoulli log probability
     */
    private fun bernoulliLogProbability(x: Double, probability: Double): Double {
        val p = max(1e-9, min(1.0 - 1e-9, probability)) // Clamp to avoid log(0)
        return if (x > 0.5) ln(p) else ln(1 - p)
    }
    
    /**
     * Calculate feature probability for a class
     */
    private fun calculateFeatureProbability(
        featureValue: Double, 
        classInfo: ClassInfo, 
        featureIndex: Int
    ): Double {
        if (featureIndex >= classInfo.featureStats.size) return 1e-9
        
        val stats = classInfo.featureStats[featureIndex]
        
        return when (variant) {
            is NaiveBayesVariant.Gaussian -> {
                val adjustedVariance = max(stats.variance, 1e-9) // Prevent division by zero
                gaussianProbability(featureValue, stats.mean, adjustedVariance)
            }
            
            is NaiveBayesVariant.Multinomial -> {
                exp(multinomialLogProbability(featureValue, stats, featureIndex, numFeatures))
            }
            
            is NaiveBayesVariant.Bernoulli -> {
                val probability = stats.mean // For Bernoulli, mean is the probability
                exp(bernoulliLogProbability(featureValue, probability))
            }
        }
    }
    
    /**
     * Update class statistics with a new sample
     */
    private fun updateClassStatistics(sample: ClassificationSample) {
        val classInfo = classInfoMap.getOrPut(sample.label) { 
            ClassInfo(sample.label, MutableList(numFeatures) { FeatureStatistics() }) 
        }
        
        // Ensure feature stats list is properly sized
        while (classInfo.featureStats.size < numFeatures) {
            classInfo.featureStats.add(FeatureStatistics())
        }
        
        // Update feature statistics
        for (i in sample.features.indices) {
            val currentStats = classInfo.featureStats[i]
            classInfo.featureStats[i] = currentStats.update(sample.features[i])
        }
        
        // Update sample count
        classInfoMap[sample.label] = classInfo.updateSampleCount(classInfo.sampleCount + 1)
    }
    
    /**
     * Update class priors
     */
    private fun updatePriors() {
        for (className in classInfoMap.keys) {
            val classInfo = classInfoMap[className]!!
            classInfoMap[className] = classInfo.updateLogPrior(totalSamples, smoothing)
        }
    }
    
    /**
     * Fit the Naive Bayes model
     */
    suspend fun fit(trainingSamples: List<ClassificationSample>) {
        require(trainingSamples.isNotEmpty()) { "Training data cannot be empty" }
        
        println("🤖 Training Naive Bayes Classifier (${variant.name})")
        println("=" .repeat(50))
        println("Training samples: ${trainingSamples.size}")
        println("Features: ${trainingSamples.first().features.size}")
        println("Classes: ${trainingSamples.map { it.label }.toSet().size}")
        println("Smoothing: $smoothing")
        
        val trainingTime = kotlin.system.measureTimeMillis {
            // Initialize
            numFeatures = trainingSamples.first().features.size
            totalSamples = trainingSamples.size
            classInfoMap.clear()
            
            // Create feature names if not provided
            if (featureNames.isEmpty()) {
                repeat(numFeatures) { featureNames.add("feature_$it") }
            }
            
            // Process samples
            trainingSamples.forEach { sample ->
                updateClassStatistics(sample)
                
                // Yield periodically for coroutine cooperation
                if (sample.id % 100 == 0) yield()
            }
            
            // Update priors
            updatePriors()
        }
        
        fitted = true
        
        println("✅ Training completed in ${trainingTime}ms")
        println("\n🎯 Model Summary:")
        println("Classes trained: ${classInfoMap.size}")
        classInfoMap.values.sortedBy { it.className }.forEach { classInfo ->
            val prior = exp(classInfo.logPrior)
            println("  Class '${classInfo.className}': ${classInfo.sampleCount} samples " +
                   "(prior: ${"%.4f".format(prior)})")
        }
    }
    
    /**
     * Incremental fit for online learning
     */
    fun partialFit(samples: List<ClassificationSample>) {
        if (!fitted) {
            throw NaiveBayesException("Must call fit() before partialFit()")
        }
        
        samples.forEach { sample ->
            updateClassStatistics(sample)
            totalSamples++
        }
        
        updatePriors()
    }
    
    /**
     * Predict class probabilities for a sample
     */
    fun predictProbabilities(sample: ClassificationSample): Map<String, Double> {
        if (!fitted) {
            throw NaiveBayesException("Model not fitted. Call fit() first.")
        }
        
        val logProbabilities = mutableMapOf<String, Double>()
        
        for ((className, classInfo) in classInfoMap) {
            var logProb = classInfo.logPrior
            
            // Calculate feature likelihoods
            for (i in sample.features.indices) {
                val featureProb = calculateFeatureProbability(sample.features[i], classInfo, i)
                logProb += ln(max(featureProb, 1e-300)) // Prevent log(0)
            }
            
            logProbabilities[className] = logProb
        }
        
        // Convert to probabilities using log-sum-exp trick for numerical stability
        val maxLogProb = logProbabilities.values.maxOrNull() ?: 0.0
        val probabilities = logProbabilities.mapValues { (_, logProb) ->
            exp(logProb - maxLogProb)
        }
        
        // Normalize
        val sumProb = probabilities.values.sum()
        return probabilities.mapValues { (_, prob) -> prob / sumProb }
    }
    
    /**
     * Predict class for a sample
     */
    fun predict(sample: ClassificationSample): PredictionResult {
        val classProbabilities = predictProbabilities(sample)
        val (predictedClass, probability) = classProbabilities.maxByOrNull { it.value }
            ?: throw NaiveBayesException("No prediction could be made")
        
        return PredictionResult(predictedClass, probability, classProbabilities)
    }
    
    /**
     * Predict classes for multiple samples
     */
    fun predict(samples: List<ClassificationSample>): List<PredictionResult> {
        return samples.map { predict(it) }
    }
    
    /**
     * Calculate feature importance based on variance between classes
     */
    fun calculateFeatureImportance(): List<Pair<String, Double>> {
        if (!fitted) {
            throw NaiveBayesException("Model not fitted. Call fit() first.")
        }
        
        val importances = mutableListOf<Double>()
        
        for (featureIndex in 0 until numFeatures) {
            val classMeans = classInfoMap.values.map { classInfo ->
                if (featureIndex < classInfo.featureStats.size) {
                    classInfo.featureStats[featureIndex].mean
                } else {
                    0.0
                }
            }
            
            // Calculate variance between class means as importance measure
            val overallMean = classMeans.average()
            val betweenClassVariance = classMeans.sumOf { (it - overallMean).pow(2) } / classMeans.size
            
            importances.add(betweenClassVariance)
        }
        
        // Normalize importances
        val maxImportance = importances.maxOrNull() ?: 1.0
        val normalizedImportances = importances.map { it / maxImportance }
        
        return featureNames.zip(normalizedImportances).sortedByDescending { it.second }
    }
    
    /**
     * Evaluate model performance
     */
    fun evaluate(testSamples: List<ClassificationSample>): ClassificationMetrics {
        val predictions = predict(testSamples)
        val actualLabels = testSamples.map { it.label }
        val predictedLabels = predictions.map { it.predictedClass }
        
        // Calculate confusion matrix
        val confusionMatrix = mutableMapOf<String, MutableMap<String, Int>>()
        actualLabels.zip(predictedLabels).forEach { (actual, predicted) ->
            confusionMatrix.getOrPut(actual) { mutableMapOf() }
                .merge(predicted, 1) { old, new -> old + new }
        }
        
        // Calculate per-class metrics
        val classes = (actualLabels + predictedLabels).toSet()
        val precision = mutableMapOf<String, Double>()
        val recall = mutableMapOf<String, Double>()
        val f1Score = mutableMapOf<String, Double>()
        
        for (className in classes) {
            val tp = confusionMatrix[className]?.get(className) ?: 0
            val fp = classes.sumOf { other -> 
                if (other != className) confusionMatrix[other]?.get(className) ?: 0 else 0
            }
            val fn = confusionMatrix[className]?.values?.sumOf { it } ?: 0 - tp
            
            val precisionValue = if (tp + fp > 0) tp.toDouble() / (tp + fp) else 0.0
            val recallValue = if (tp + fn > 0) tp.toDouble() / (tp + fn) else 0.0
            val f1Value = if (precisionValue + recallValue > 0) {
                2 * precisionValue * recallValue / (precisionValue + recallValue)
            } else {
                0.0
            }
            
            precision[className] = precisionValue
            recall[className] = recallValue
            f1Score[className] = f1Value
        }
        
        // Calculate macro averages
        val macroPrecision = precision.values.average()
        val macroRecall = recall.values.average()
        val macroF1 = f1Score.values.average()
        
        // Calculate accuracy
        val correct = actualLabels.zip(predictedLabels).count { (actual, predicted) -> actual == predicted }
        val accuracy = correct.toDouble() / testSamples.size
        
        return ClassificationMetrics(
            accuracy = accuracy,
            precision = precision,
            recall = recall,
            f1Score = f1Score,
            macroPrecision = macroPrecision,
            macroRecall = macroRecall,
            macroF1 = macroF1,
            confusionMatrix = confusionMatrix.mapValues { it.value.toMap() },
            sampleCount = testSamples.size
        )
    }
    
    /**
     * Cross-validation for model evaluation
     */
    suspend fun crossValidate(samples: List<ClassificationSample>, folds: Int = 5): List<ClassificationMetrics> {
        require(folds > 1) { "Number of folds must be > 1" }
        require(samples.size >= folds) { "Not enough samples for $folds folds" }
        
        println("🔄 Performing $folds-fold cross-validation...")
        
        val shuffledSamples = samples.shuffled()
        val foldSize = samples.size / folds
        val results = mutableListOf<ClassificationMetrics>()
        
        for (fold in 0 until folds) {
            val startIdx = fold * foldSize
            val endIdx = if (fold == folds - 1) samples.size else startIdx + foldSize
            
            val testSet = shuffledSamples.subList(startIdx, endIdx)
            val trainSet = shuffledSamples.subList(0, startIdx) + shuffledSamples.subList(endIdx, samples.size)
            
            // Create new model for this fold
            val foldModel = NaiveBayesImplementation(variant, smoothing, priors)
            
            // Train and evaluate
            foldModel.fit(trainSet)
            val metrics = foldModel.evaluate(testSet)
            results.add(metrics)
            
            println("Fold ${fold + 1}: Accuracy = ${"%.4f".format(metrics.accuracy)}, " +
                   "Macro F1 = ${"%.4f".format(metrics.macroF1)}")
        }
        
        // Print average results
        val avgAccuracy = results.map { it.accuracy }.average()
        val avgF1 = results.map { it.macroF1 }.average()
        println("✅ Cross-validation completed")
        println("Average Accuracy: ${"%.4f".format(avgAccuracy)}")
        println("Average Macro F1: ${"%.4f".format(avgF1)}")
        
        return results
    }
    
    companion object {
        /**
         * Generate synthetic Iris-like dataset
         */
        fun generateIrisDataset(samples: Int): List<ClassificationSample> {
            val random = Random.Default
            val classes = listOf("setosa", "versicolor", "virginica")
            val dataset = mutableListOf<ClassificationSample>()
            
            repeat(samples) { id ->
                val classIndex = id % 3
                val className = classes[classIndex]
                
                val features = when (classIndex) {
                    0 -> doubleArrayOf( // Setosa
                        5.0 + random.nextGaussian() * 0.5,
                        3.5 + random.nextGaussian() * 0.3,
                        1.5 + random.nextGaussian() * 0.2,
                        0.3 + random.nextGaussian() * 0.1
                    )
                    1 -> doubleArrayOf( // Versicolor
                        6.0 + random.nextGaussian() * 0.5,
                        2.8 + random.nextGaussian() * 0.3,
                        4.3 + random.nextGaussian() * 0.4,
                        1.3 + random.nextGaussian() * 0.2
                    )
                    else -> doubleArrayOf( // Virginica
                        6.5 + random.nextGaussian() * 0.5,
                        3.0 + random.nextGaussian() * 0.3,
                        5.5 + random.nextGaussian() * 0.4,
                        2.0 + random.nextGaussian() * 0.3
                    )
                }
                
                // Ensure non-negative values
                features.indices.forEach { i ->
                    features[i] = maxOf(0.1, features[i])
                }
                
                dataset.add(ClassificationSample(features, className, id))
            }
            
            return dataset.shuffled()
        }
        
        /**
         * Generate text classification dataset (for Multinomial/Bernoulli)
         */
        fun generateTextDataset(samples: Int): List<ClassificationSample> {
            val random = Random.Default
            val topics = listOf("sports", "technology", "politics")
            val dataset = mutableListOf<ClassificationSample>()
            
            // Vocabulary features (word counts/presence)
            val vocabularySize = 20
            
            repeat(samples) { id ->
                val topicIndex = id % 3
                val topic = topics[topicIndex]
                
                val features = DoubleArray(vocabularySize) { featureIndex ->
                    // Generate topic-specific word distributions
                    val baseRate = when (topicIndex) {
                        0 -> if (featureIndex < 7) random.nextDouble() * 5 else random.nextDouble()
                        1 -> if (featureIndex in 5..12) random.nextDouble() * 5 else random.nextDouble()
                        else -> if (featureIndex > 10) random.nextDouble() * 5 else random.nextDouble()
                    }
                    
                    // For Bernoulli variant, convert to binary
                    if (random.nextDouble() < 0.7) baseRate else 0.0
                }
                
                dataset.add(ClassificationSample(features, topic, id))
            }
            
            return dataset.shuffled()
        }
        
        /**
         * Comprehensive demonstration of Naive Bayes
         */
        suspend fun demonstrateNaiveBayes() {
            println("🚀 Naive Bayes Implementation Demonstration")
            println("=" .repeat(50))
            
            try {
                val variants = listOf(
                    NaiveBayesVariant.Gaussian,
                    NaiveBayesVariant.Multinomial,
                    NaiveBayesVariant.Bernoulli
                )
                
                for (variant in variants) {
                    println("\n" + "=".repeat(60))
                    println("🔍 Testing ${variant.name} Naive Bayes")
                    println("=".repeat(60))
                    
                    // Choose appropriate dataset
                    val dataset = when (variant) {
                        is NaiveBayesVariant.Gaussian -> generateIrisDataset(300)
                        else -> generateTextDataset(300)
                    }
                    
                    // Split dataset
                    val trainSize = (dataset.size * 0.8).toInt()
                    val trainData = dataset.take(trainSize)
                    val testData = dataset.drop(trainSize)
                    
                    println("Dataset: ${dataset.size} samples (Train: $trainSize, Test: ${testData.size})")
                    
                    // Create and train model
                    val model = NaiveBayesImplementation(variant, smoothing = 1.0)
                    model.fit(trainData)
                    
                    // Evaluate performance
                    println("\n📊 Evaluating model performance...")
                    val metrics = model.evaluate(testData)
                    println(metrics)
                    
                    // Feature importance
                    val featureImportance = model.calculateFeatureImportance()
                    println("🎯 Top 5 Most Important Features:")
                    featureImportance.take(5).forEach { (feature, importance) ->
                        println("  $feature: ${"%.4f".format(importance)}")
                    }
                    
                    // Sample predictions
                    println("\n🧪 Sample Predictions:")
                    testData.take(3).forEach { sample ->
                        val prediction = model.predict(sample)
                        val topProbs = prediction.classProbabilities.entries
                            .sortedByDescending { it.value }.take(2)
                        
                        println("Sample ${sample.id}: Predicted='${prediction.predictedClass}', " +
                               "Actual='${sample.label}'")
                        topProbs.forEach { (className, prob) ->
                            println("  $className: ${"%.4f".format(prob)}")
                        }
                    }
                    
                    // Cross-validation
                    model.crossValidate(trainData, 5)
                }
                
                println("\n✅ Naive Bayes demonstration completed successfully!")
                
            } catch (e: Exception) {
                println("❌ Naive Bayes demonstration failed: ${e.message}")
                e.printStackTrace()
            }
        }
    }
}

/**
 * Main function to demonstrate Naive Bayes
 */
fun main() = runBlocking {
    NaiveBayesImplementation.demonstrateNaiveBayes()
}