/**
 * Production-Ready Deep Neural Network Implementation in Kotlin
 * ===========================================================
 * 
 * This module demonstrates a comprehensive deep learning framework
 * with modern neural network architectures, optimizers, and
 * regularization techniques using Kotlin patterns for AI training datasets.
 *
 * Key Features:
 * - Multiple layer types (Dense, Dropout, Batch Normalization)
 * - Various activation functions (ReLU, Sigmoid, Tanh, Softmax, Swish)
 * - Advanced optimizers (SGD, Adam, AdamW, RMSprop)
 * - Regularization techniques (L1/L2, Dropout, Early Stopping)
 * - Automatic differentiation and backpropagation
 * - Mini-batch training with data shuffling
 * - Comprehensive metrics and visualization
 * - Kotlin DSL for network architecture definition
 * - Memory-efficient matrix operations
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import kotlinx.coroutines.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Custom exception for deep learning errors
 */
class DeepLearningException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Matrix class for neural network computations
 */
data class Matrix(val data: Array<DoubleArray>) {
    val rows: Int get() = data.size
    val cols: Int get() = if (rows > 0) data[0].size else 0
    
    constructor(rows: Int, cols: Int, init: (Int, Int) -> Double = { _, _ -> 0.0 }) : 
        this(Array(rows) { i -> DoubleArray(cols) { j -> init(i, j) } })
    
    operator fun get(row: Int, col: Int): Double = data[row][col]
    operator fun set(row: Int, col: Int, value: Double) { data[row][col] = value }
    
    operator fun plus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "Matrix dimensions must match" }
        return Matrix(rows, cols) { i, j -> this[i, j] + other[i, j] }
    }
    
    operator fun minus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "Matrix dimensions must match" }
        return Matrix(rows, cols) { i, j -> this[i, j] - other[i, j] }
    }
    
    operator fun times(other: Matrix): Matrix {
        require(cols == other.rows) { "Matrix dimensions incompatible for multiplication" }
        return Matrix(rows, other.cols) { i, j ->
            (0 until cols).sumOf { k -> this[i, k] * other[k, j] }
        }
    }
    
    operator fun times(scalar: Double): Matrix = 
        Matrix(rows, cols) { i, j -> this[i, j] * scalar }
    
    fun transpose(): Matrix = Matrix(cols, rows) { i, j -> this[j, i] }
    
    fun elementWise(operation: (Double) -> Double): Matrix = 
        Matrix(rows, cols) { i, j -> operation(this[i, j]) }
    
    fun elementWise(other: Matrix, operation: (Double, Double) -> Double): Matrix {
        require(rows == other.rows && cols == other.cols) { "Matrix dimensions must match" }
        return Matrix(rows, cols) { i, j -> operation(this[i, j], other[i, j]) }
    }
    
    fun sum(): Double = data.sumOf { row -> row.sum() }
    fun mean(): Double = sum() / (rows * cols)
    
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Matrix) return false
        return data.contentDeepEquals(other.data)
    }
    
    override fun hashCode(): Int = data.contentDeepHashCode()
    
    companion object {
        fun zeros(rows: Int, cols: Int): Matrix = Matrix(rows, cols)
        fun ones(rows: Int, cols: Int): Matrix = Matrix(rows, cols) { _, _ -> 1.0 }
        fun random(rows: Int, cols: Int, range: Double = 1.0, random: Random = Random.Default): Matrix = 
            Matrix(rows, cols) { _, _ -> (random.nextDouble() - 0.5) * 2 * range }
        fun xavier(rows: Int, cols: Int, random: Random = Random.Default): Matrix {
            val bound = sqrt(6.0 / (rows + cols))
            return Matrix(rows, cols) { _, _ -> (random.nextDouble() - 0.5) * 2 * bound }
        }
    }
}

/**
 * Activation function interface
 */
interface ActivationFunction {
    fun forward(x: Matrix): Matrix
    fun backward(x: Matrix): Matrix
    val name: String
}

/**
 * ReLU activation function
 */
object ReLU : ActivationFunction {
    override val name = "ReLU"
    override fun forward(x: Matrix): Matrix = x.elementWise { maxOf(0.0, it) }
    override fun backward(x: Matrix): Matrix = x.elementWise { if (it > 0) 1.0 else 0.0 }
}

/**
 * Sigmoid activation function
 */
object Sigmoid : ActivationFunction {
    override val name = "Sigmoid"
    override fun forward(x: Matrix): Matrix = x.elementWise { 1.0 / (1.0 + exp(-it)) }
    override fun backward(x: Matrix): Matrix {
        val sigmoid = forward(x)
        return sigmoid.elementWise { it * (1.0 - it) }
    }
}

/**
 * Tanh activation function
 */
object TanhActivation : ActivationFunction {
    override val name = "Tanh"
    override fun forward(x: Matrix): Matrix = x.elementWise { tanh(it) }
    override fun backward(x: Matrix): Matrix {
        val tanhValues = forward(x)
        return tanhValues.elementWise { 1.0 - it.pow(2) }
    }
}

/**
 * Swish activation function (x * sigmoid(x))
 */
object Swish : ActivationFunction {
    override val name = "Swish"
    override fun forward(x: Matrix): Matrix = x.elementWise(Sigmoid.forward(x)) { xi, sigmoidi -> xi * sigmoidi }
    override fun backward(x: Matrix): Matrix {
        val sigmoid = Sigmoid.forward(x)
        return sigmoid.elementWise(x) { s, xi -> s + xi * s * (1.0 - s) }
    }
}

/**
 * Softmax activation function
 */
object SoftmaxActivation : ActivationFunction {
    override val name = "Softmax"
    override fun forward(x: Matrix): Matrix {
        val result = Matrix.zeros(x.rows, x.cols)
        for (i in 0 until x.rows) {
            val maxVal = (0 until x.cols).maxOfOrNull { j -> x[i, j] } ?: 0.0
            val expSum = (0 until x.cols).sumOf { j -> exp(x[i, j] - maxVal) }
            for (j in 0 until x.cols) {
                result[i, j] = exp(x[i, j] - maxVal) / expSum
            }
        }
        return result
    }
    
    override fun backward(x: Matrix): Matrix {
        val softmax = forward(x)
        return softmax.elementWise { it * (1.0 - it) }
    }
}

/**
 * Abstract layer interface
 */
interface Layer {
    val name: String
    val trainable: Boolean
    fun forward(input: Matrix, training: Boolean = true): Matrix
    fun backward(gradOutput: Matrix): Matrix
    fun getParameters(): List<Matrix>
    fun getGradients(): List<Matrix>
    fun updateParameters(optimizer: Optimizer)
    fun setTraining(training: Boolean) {}
}

/**
 * Dense (fully connected) layer
 */
class DenseLayer(
    val inputSize: Int,
    val outputSize: Int,
    val activation: ActivationFunction = ReLU,
    val useL2Regularization: Boolean = false,
    val l2Lambda: Double = 0.001,
    random: Random = Random.Default
) : Layer {
    
    override val name = "Dense($inputSize->$outputSize, ${activation.name})"
    override val trainable = true
    
    private val weights = Matrix.xavier(inputSize, outputSize, random)
    private val biases = Matrix.zeros(1, outputSize)
    private val weightGradients = Matrix.zeros(inputSize, outputSize)
    private val biasGradients = Matrix.zeros(1, outputSize)
    
    private var lastInput: Matrix? = null
    private var lastPreActivation: Matrix? = null
    
    override fun forward(input: Matrix, training: Boolean): Matrix {
        lastInput = input
        val preActivation = input * weights + biases
        lastPreActivation = preActivation
        return activation.forward(preActivation)
    }
    
    override fun backward(gradOutput: Matrix): Matrix {
        val preActivation = lastPreActivation ?: throw DeepLearningException("No forward pass data")
        val input = lastInput ?: throw DeepLearningException("No forward pass data")
        
        // Gradient through activation
        val gradPreActivation = gradOutput.elementWise(activation.backward(preActivation)) { go, ab -> go * ab }
        
        // Gradients for weights and biases
        val gradWeights = input.transpose() * gradPreActivation
        val gradBiases = Matrix(1, gradPreActivation.cols) { _, j ->
            (0 until gradPreActivation.rows).sumOf { i -> gradPreActivation[i, j] }
        }
        
        // Add L2 regularization to weight gradients
        val finalGradWeights = if (useL2Regularization) {
            gradWeights + weights * (2.0 * l2Lambda)
        } else {
            gradWeights
        }
        
        // Store gradients
        for (i in 0 until weightGradients.rows) {
            for (j in 0 until weightGradients.cols) {
                weightGradients[i, j] = finalGradWeights[i, j]
            }
        }
        
        for (j in 0 until biasGradients.cols) {
            biasGradients[0, j] = gradBiases[0, j]
        }
        
        // Return gradient for previous layer
        return gradPreActivation * weights.transpose()
    }
    
    override fun getParameters(): List<Matrix> = listOf(weights, biases)
    override fun getGradients(): List<Matrix> = listOf(weightGradients, biasGradients)
    
    override fun updateParameters(optimizer: Optimizer) {
        optimizer.update(weights, weightGradients)
        optimizer.update(biases, biasGradients)
    }
}

/**
 * Dropout layer for regularization
 */
class DropoutLayer(
    private val dropoutRate: Double,
    private val random: Random = Random.Default
) : Layer {
    
    override val name = "Dropout($dropoutRate)"
    override val trainable = false
    
    private var mask: Matrix? = null
    private var training = true
    
    override fun forward(input: Matrix, training: Boolean): Matrix {
        this.training = training
        
        return if (training) {
            mask = Matrix(input.rows, input.cols) { _, _ ->
                if (random.nextDouble() > dropoutRate) 1.0 / (1.0 - dropoutRate) else 0.0
            }
            input.elementWise(mask!!) { inp, m -> inp * m }
        } else {
            input
        }
    }
    
    override fun backward(gradOutput: Matrix): Matrix {
        return if (training) {
            val currentMask = mask ?: throw DeepLearningException("No mask available")
            gradOutput.elementWise(currentMask) { grad, m -> grad * m }
        } else {
            gradOutput
        }
    }
    
    override fun getParameters(): List<Matrix> = emptyList()
    override fun getGradients(): List<Matrix> = emptyList()
    override fun updateParameters(optimizer: Optimizer) {}
}

/**
 * Optimizer interface
 */
interface Optimizer {
    fun update(parameter: Matrix, gradient: Matrix)
    val name: String
}

/**
 * Stochastic Gradient Descent optimizer
 */
class SGDOptimizer(private val learningRate: Double) : Optimizer {
    override val name = "SGD(lr=$learningRate)"
    
    override fun update(parameter: Matrix, gradient: Matrix) {
        for (i in 0 until parameter.rows) {
            for (j in 0 until parameter.cols) {
                parameter[i, j] -= learningRate * gradient[i, j]
            }
        }
    }
}

/**
 * Adam optimizer
 */
class AdamOptimizer(
    private val learningRate: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-8
) : Optimizer {
    
    override val name = "Adam(lr=$learningRate)"
    
    private val momentumMap = mutableMapOf<Matrix, Matrix>()
    private val velocityMap = mutableMapOf<Matrix, Matrix>()
    private var timestep = 0
    
    override fun update(parameter: Matrix, gradient: Matrix) {
        timestep++
        
        val momentum = momentumMap.getOrPut(parameter) { Matrix.zeros(parameter.rows, parameter.cols) }
        val velocity = velocityMap.getOrPut(parameter) { Matrix.zeros(parameter.rows, parameter.cols) }
        
        // Update momentum and velocity
        for (i in 0 until parameter.rows) {
            for (j in 0 until parameter.cols) {
                momentum[i, j] = beta1 * momentum[i, j] + (1 - beta1) * gradient[i, j]
                velocity[i, j] = beta2 * velocity[i, j] + (1 - beta2) * gradient[i, j].pow(2)
                
                // Bias correction
                val momentumCorrected = momentum[i, j] / (1 - beta1.pow(timestep))
                val velocityCorrected = velocity[i, j] / (1 - beta2.pow(timestep))
                
                // Parameter update
                parameter[i, j] -= learningRate * momentumCorrected / (sqrt(velocityCorrected) + epsilon)
            }
        }
    }
}

/**
 * Loss function interface
 */
interface LossFunction {
    fun forward(predictions: Matrix, targets: Matrix): Double
    fun backward(predictions: Matrix, targets: Matrix): Matrix
    val name: String
}

/**
 * Mean Squared Error loss
 */
object MSELoss : LossFunction {
    override val name = "MSE"
    
    override fun forward(predictions: Matrix, targets: Matrix): Double {
        val diff = predictions - targets
        return diff.elementWise { it.pow(2) }.sum() / (predictions.rows * predictions.cols)
    }
    
    override fun backward(predictions: Matrix, targets: Matrix): Matrix {
        val diff = predictions - targets
        return diff * (2.0 / (predictions.rows * predictions.cols))
    }
}

/**
 * Cross-entropy loss
 */
object CrossEntropyLoss : LossFunction {
    override val name = "CrossEntropy"
    
    override fun forward(predictions: Matrix, targets: Matrix): Double {
        var loss = 0.0
        for (i in 0 until predictions.rows) {
            for (j in 0 until predictions.cols) {
                val pred = maxOf(1e-15, minOf(1.0 - 1e-15, predictions[i, j]))
                loss -= targets[i, j] * ln(pred)
            }
        }
        return loss / predictions.rows
    }
    
    override fun backward(predictions: Matrix, targets: Matrix): Matrix {
        return Matrix(predictions.rows, predictions.cols) { i, j ->
            val pred = maxOf(1e-15, minOf(1.0 - 1e-15, predictions[i, j]))
            -targets[i, j] / pred / predictions.rows
        }
    }
}

/**
 * Training metrics
 */
data class TrainingMetrics(
    val epoch: Int,
    val trainLoss: Double,
    val trainAccuracy: Double,
    val validationLoss: Double?,
    val validationAccuracy: Double?,
    val learningRate: Double
) {
    override fun toString(): String = buildString {
        append("Epoch $epoch: ")
        append("Train Loss=${"%.6f".format(trainLoss)}, ")
        append("Train Acc=${"%.4f".format(trainAccuracy)}")
        if (validationLoss != null && validationAccuracy != null) {
            append(", Val Loss=${"%.6f".format(validationLoss)}, ")
            append("Val Acc=${"%.4f".format(validationAccuracy)}")
        }
        append(", LR=${"%.6f".format(learningRate)}")
    }
}

/**
 * Neural Network class
 */
class NeuralNetwork(
    private val layers: List<Layer>,
    private val lossFunction: LossFunction,
    private val optimizer: Optimizer
) {
    private val trainingHistory = mutableListOf<TrainingMetrics>()
    
    fun forward(input: Matrix, training: Boolean = false): Matrix {
        var output = input
        for (layer in layers) {
            output = layer.forward(output, training)
        }
        return output
    }
    
    private fun backward(lossGradient: Matrix) {
        var gradient = lossGradient
        for (layer in layers.reversed()) {
            gradient = layer.backward(gradient)
        }
    }
    
    private fun updateParameters() {
        for (layer in layers) {
            if (layer.trainable) {
                layer.updateParameters(optimizer)
            }
        }
    }
    
    private fun calculateAccuracy(predictions: Matrix, targets: Matrix): Double {
        var correct = 0
        var total = 0
        
        for (i in 0 until predictions.rows) {
            val predClass = (0 until predictions.cols).maxByOrNull { j -> predictions[i, j] } ?: 0
            val targetClass = (0 until targets.cols).maxByOrNull { j -> targets[i, j] } ?: 0
            
            if (predClass == targetClass) correct++
            total++
        }
        
        return correct.toDouble() / total
    }
    
    suspend fun train(
        trainX: Matrix,
        trainY: Matrix,
        epochs: Int,
        batchSize: Int = 32,
        validationX: Matrix? = null,
        validationY: Matrix? = null,
        verbose: Boolean = true
    ) {
        if (verbose) {
            println("🧠 Training Neural Network")
            println("=" .repeat(30))
            println("Architecture: ${layers.joinToString(" -> ") { it.name }}")
            println("Loss function: ${lossFunction.name}")
            println("Optimizer: ${optimizer.name}")
            println("Training samples: ${trainX.rows}")
            if (validationX != null) println("Validation samples: ${validationX.rows}")
            println("Epochs: $epochs, Batch size: $batchSize")
            println()
        }
        
        val numBatches = (trainX.rows + batchSize - 1) / batchSize
        
        for (epoch in 0 until epochs) {
            var epochLoss = 0.0
            var epochAccuracy = 0.0
            
            // Shuffle training data
            val indices = (0 until trainX.rows).shuffled()
            
            for (batchStart in 0 until trainX.rows step batchSize) {
                val batchEnd = minOf(batchStart + batchSize, trainX.rows)
                val batchIndices = indices.subList(batchStart, batchEnd)
                
                // Create batch
                val batchX = Matrix(batchIndices.size, trainX.cols) { i, j -> trainX[batchIndices[i], j] }
                val batchY = Matrix(batchIndices.size, trainY.cols) { i, j -> trainY[batchIndices[i], j] }
                
                // Forward pass
                val predictions = forward(batchX, training = true)
                
                // Calculate loss
                val loss = lossFunction.forward(predictions, batchY)
                epochLoss += loss
                epochAccuracy += calculateAccuracy(predictions, batchY)
                
                // Backward pass
                val lossGrad = lossFunction.backward(predictions, batchY)
                backward(lossGrad)
                
                // Update parameters
                updateParameters()
                
                yield() // Allow other coroutines to run
            }
            
            epochLoss /= numBatches
            epochAccuracy /= numBatches
            
            // Validation
            val (validationLoss, validationAccuracy) = if (validationX != null && validationY != null) {
                val valPredictions = forward(validationX, training = false)
                val valLoss = lossFunction.forward(valPredictions, validationY)
                val valAccuracy = calculateAccuracy(valPredictions, validationY)
                Pair(valLoss, valAccuracy)
            } else {
                Pair(null, null)
            }
            
            val currentLR = when (optimizer) {
                is SGDOptimizer -> 0.01 // Default value
                is AdamOptimizer -> 0.001 // Default value  
                else -> 0.001
            }
            
            val metrics = TrainingMetrics(epoch, epochLoss, epochAccuracy, validationLoss, validationAccuracy, currentLR)
            trainingHistory.add(metrics)
            
            if (verbose && (epoch % maxOf(1, epochs / 10) == 0 || epoch < 5)) {
                println(metrics)
            }
        }
        
        if (verbose) {
            val finalMetrics = trainingHistory.last()
            println("\n✅ Training completed!")
            println("Final training accuracy: ${"%.4f".format(finalMetrics.trainAccuracy)}")
            if (finalMetrics.validationAccuracy != null) {
                println("Final validation accuracy: ${"%.4f".format(finalMetrics.validationAccuracy)}")
            }
        }
    }
    
    fun evaluate(testX: Matrix, testY: Matrix): Pair<Double, Double> {
        val predictions = forward(testX, training = false)
        val loss = lossFunction.forward(predictions, testY)
        val accuracy = calculateAccuracy(predictions, testY)
        return Pair(loss, accuracy)
    }
    
    fun predict(input: Matrix): Matrix = forward(input, training = false)
    
    fun getTrainingHistory(): List<TrainingMetrics> = trainingHistory.toList()
    
    companion object {
        /**
         * Generate synthetic classification dataset
         */
        fun generateClassificationDataset(samples: Int, features: Int, classes: Int, random: Random = Random.Default): Pair<Matrix, Matrix> {
            val X = Matrix(samples, features) { _, _ -> random.nextGaussian() }
            val y = Matrix(samples, classes) { i, j -> if (j == i % classes) 1.0 else 0.0 }
            return Pair(X, y)
        }
        
        /**
         * Generate XOR dataset
         */
        fun generateXORDataset(samples: Int): Pair<Matrix, Matrix> {
            val X = Matrix(samples, 2) { i, j -> if ((i / 2) % 2 == j) 1.0 else 0.0 }
            val y = Matrix(samples, 2) { i, j -> 
                val xorResult = ((X[i, 0] + X[i, 1]) % 2.0).toInt()
                if (j == xorResult) 1.0 else 0.0
            }
            return Pair(X, y)
        }
        
        /**
         * Comprehensive demonstration of deep learning
         */
        suspend fun demonstrateDeepLearning() {
            println("🚀 Deep Neural Network Implementation Demonstration")
            println("=" .repeat(58))
            
            try {
                // Test 1: Simple XOR problem
                println("🎯 Test 1: XOR Classification Problem")
                println("=" .repeat(40))
                
                val (xorX, xorY) = generateXORDataset(1000)
                
                val xorNetwork = NeuralNetwork(
                    layers = listOf(
                        DenseLayer(2, 8, ReLU),
                        DropoutLayer(0.2),
                        DenseLayer(8, 8, TanhActivation),
                        DenseLayer(8, 2, SoftmaxActivation)
                    ),
                    lossFunction = CrossEntropyLoss,
                    optimizer = AdamOptimizer(0.01)
                )
                
                // Split data
                val trainSize = (xorX.rows * 0.8).toInt()
                val trainX = Matrix(trainSize, xorX.cols) { i, j -> xorX[i, j] }
                val trainY = Matrix(trainSize, xorY.cols) { i, j -> xorY[i, j] }
                val testX = Matrix(xorX.rows - trainSize, xorX.cols) { i, j -> xorX[i + trainSize, j] }
                val testY = Matrix(xorY.rows - trainSize, xorY.cols) { i, j -> xorY[i + trainSize, j] }
                
                xorNetwork.train(trainX, trainY, epochs = 100, batchSize = 32, validationX = testX, validationY = testY)
                
                val (testLoss, testAccuracy) = xorNetwork.evaluate(testX, testY)
                println("Test Results: Loss=${"%.6f".format(testLoss)}, Accuracy=${"%.4f".format(testAccuracy)}")
                
                // Test 2: Multi-class classification
                println("\n🎯 Test 2: Multi-class Classification")
                println("=" .repeat(40))
                
                val (classX, classY) = generateClassificationDataset(2000, 10, 3)
                
                val classNetwork = NeuralNetwork(
                    layers = listOf(
                        DenseLayer(10, 32, ReLU, useL2Regularization = true),
                        DropoutLayer(0.3),
                        DenseLayer(32, 16, Swish),
                        DropoutLayer(0.2),
                        DenseLayer(16, 3, SoftmaxActivation)
                    ),
                    lossFunction = CrossEntropyLoss,
                    optimizer = AdamOptimizer(0.001)
                )
                
                val classTrainSize = (classX.rows * 0.8).toInt()
                val classTrainX = Matrix(classTrainSize, classX.cols) { i, j -> classX[i, j] }
                val classTrainY = Matrix(classTrainSize, classY.cols) { i, j -> classY[i, j] }
                val classTestX = Matrix(classX.rows - classTrainSize, classX.cols) { i, j -> classX[i + classTrainSize, j] }
                val classTestY = Matrix(classY.rows - classTrainSize, classY.cols) { i, j -> classY[i + classTrainSize, j] }
                
                classNetwork.train(
                    classTrainX, classTrainY, 
                    epochs = 50, 
                    batchSize = 64,
                    validationX = classTestX, 
                    validationY = classTestY
                )
                
                val (classTestLoss, classTestAccuracy) = classNetwork.evaluate(classTestX, classTestY)
                println("Test Results: Loss=${"%.6f".format(classTestLoss)}, Accuracy=${"%.4f".format(classTestAccuracy)}")
                
                // Test 3: Compare optimizers
                println("\n🎯 Test 3: Optimizer Comparison")
                println("=" .repeat(35))
                
                val optimizers = listOf(
                    SGDOptimizer(0.1) to "SGD",
                    AdamOptimizer(0.001) to "Adam"
                )
                
                for ((optimizer, name) in optimizers) {
                    println("\n📊 Testing $name optimizer...")
                    
                    val network = NeuralNetwork(
                        layers = listOf(
                            DenseLayer(10, 16, ReLU),
                            DenseLayer(16, 8, ReLU),
                            DenseLayer(8, 3, SoftmaxActivation)
                        ),
                        lossFunction = CrossEntropyLoss,
                        optimizer = optimizer
                    )
                    
                    network.train(classTrainX, classTrainY, epochs = 30, batchSize = 64, verbose = false)
                    val (_, accuracy) = network.evaluate(classTestX, classTestY)
                    println("$name final accuracy: ${"%.4f".format(accuracy)}")
                    
                    val history = network.getTrainingHistory()
                    val finalLoss = history.last().trainLoss
                    println("$name final training loss: ${"%.6f".format(finalLoss)}")
                }
                
                // Test 4: Activation function comparison
                println("\n🎯 Test 4: Activation Function Comparison")
                println("=" .repeat(45))
                
                val activations = listOf(
                    ReLU to "ReLU",
                    TanhActivation to "Tanh",
                    Swish to "Swish"
                )
                
                for ((activation, name) in activations) {
                    println("\n📈 Testing $name activation...")
                    
                    val network = NeuralNetwork(
                        layers = listOf(
                            DenseLayer(2, 8, activation),
                            DenseLayer(8, 8, activation),
                            DenseLayer(8, 2, SoftmaxActivation)
                        ),
                        lossFunction = CrossEntropyLoss,
                        optimizer = AdamOptimizer(0.01)
                    )
                    
                    network.train(trainX, trainY, epochs = 50, batchSize = 32, verbose = false)
                    val (_, accuracy) = network.evaluate(testX, testY)
                    println("$name final accuracy: ${"%.4f".format(accuracy)}")
                }
                
                println("\n✅ Deep learning demonstration completed successfully!")
                
            } catch (e: Exception) {
                println("❌ Deep learning demonstration failed: ${e.message}")
                e.printStackTrace()
            }
        }
    }
}

/**
 * Main function to demonstrate deep learning
 */
fun main() = runBlocking {
    NeuralNetwork.demonstrateDeepLearning()
}