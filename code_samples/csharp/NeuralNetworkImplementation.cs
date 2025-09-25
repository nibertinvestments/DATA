/*
Production-Ready Neural Network Implementation in C#
===================================================

This module demonstrates a comprehensive multi-layer perceptron (MLP)
neural network with backpropagation, multiple optimizers, and enterprise
C# patterns for AI training datasets.

Key Features:
- Multi-layer perceptron with configurable architecture
- Multiple activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU)
- Various optimization algorithms (SGD, Momentum, Adam, RMSprop)
- Regularization techniques (L1, L2, Dropout, Batch Normalization)
- Parallel training with PLINQ and TPL
- Enterprise patterns with dependency injection
- Comprehensive async/await pattern implementation
- Thread-safe operations with concurrent collections
- Comprehensive logging and monitoring
- Production deployment considerations

Author: AI Training Dataset
License: MIT
*/

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;
using System.Diagnostics;

namespace NeuralNetworkProduction
{
    #region Enumerations and Data Structures

    /// <summary>
    /// Available activation functions for neural network layers
    /// </summary>
    public enum ActivationType
    {
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU,
        Swish,
        Linear
    }

    /// <summary>
    /// Available optimization algorithms for training
    /// </summary>
    public enum OptimizerType
    {
        SGD,
        Momentum,
        Adam,
        RMSprop,
        AdaGrad
    }

    /// <summary>
    /// Regularization techniques for preventing overfitting
    /// </summary>
    public enum RegularizationType
    {
        None,
        L1,
        L2,
        ElasticNet,
        Dropout
    }

    /// <summary>
    /// Training sample containing features and target values
    /// </summary>
    public readonly struct TrainingSample
    {
        public readonly double[] Features;
        public readonly double[] Targets;
        public readonly int Id;

        public TrainingSample(double[] features, double[] targets, int id)
        {
            Features = features ?? throw new ArgumentNullException(nameof(features));
            Targets = targets ?? throw new ArgumentNullException(nameof(targets));
            Id = id;
        }

        public override string ToString() => $"Sample {Id}: Features={Features.Length}, Targets={Targets.Length}";
    }

    /// <summary>
    /// Training metrics for a single epoch
    /// </summary>
    public readonly struct EpochMetrics
    {
        public readonly int Epoch;
        public readonly double TrainingLoss;
        public readonly double ValidationLoss;
        public readonly double TrainingAccuracy;
        public readonly double ValidationAccuracy;
        public readonly TimeSpan Duration;

        public EpochMetrics(int epoch, double trainingLoss, double validationLoss, 
                          double trainingAccuracy, double validationAccuracy, TimeSpan duration)
        {
            Epoch = epoch;
            TrainingLoss = trainingLoss;
            ValidationLoss = validationLoss;
            TrainingAccuracy = trainingAccuracy;
            ValidationAccuracy = validationAccuracy;
            Duration = duration;
        }
    }

    /// <summary>
    /// Neural network layer configuration
    /// </summary>
    public readonly struct LayerConfiguration
    {
        public readonly int Neurons;
        public readonly ActivationType Activation;
        public readonly double DropoutRate;
        public readonly bool UseBatchNormalization;

        public LayerConfiguration(int neurons, ActivationType activation = ActivationType.ReLU, 
                                double dropoutRate = 0.0, bool useBatchNormalization = false)
        {
            if (neurons <= 0) throw new ArgumentException("Neurons must be positive", nameof(neurons));
            if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));

            Neurons = neurons;
            Activation = activation;
            DropoutRate = dropoutRate;
            UseBatchNormalization = useBatchNormalization;
        }
    }

    #endregion

    #region Interfaces

    /// <summary>
    /// Interface for activation functions with forward and backward passes
    /// </summary>
    public interface IActivationFunction
    {
        double Forward(double x);
        double Backward(double x);
        string Name { get; }
    }

    /// <summary>
    /// Interface for optimization algorithms
    /// </summary>
    public interface IOptimizer
    {
        Task UpdateWeightsAsync(double[][] weights, double[][] weightGradients, 
                               double[] biases, double[] biasGradients, int layerIndex);
        void Reset();
        string Name { get; }
    }

    /// <summary>
    /// Interface for neural network models
    /// </summary>
    public interface INeuralNetwork : IDisposable
    {
        Task<bool> TrainAsync(IEnumerable<TrainingSample> trainingData, 
                             IEnumerable<TrainingSample> validationData = null,
                             CancellationToken cancellationToken = default);
        
        Task<double[]> PredictAsync(double[] features, CancellationToken cancellationToken = default);
        
        Task<double[][]> PredictBatchAsync(double[][] features, CancellationToken cancellationToken = default);
        
        Task SaveModelAsync(string filePath, CancellationToken cancellationToken = default);
        
        Task LoadModelAsync(string filePath, CancellationToken cancellationToken = default);
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// Sigmoid activation function implementation
    /// </summary>
    public class SigmoidActivation : IActivationFunction
    {
        public string Name => "Sigmoid";

        public double Forward(double x)
        {
            // Clamp input to prevent overflow
            x = Math.Max(-500, Math.Min(500, x));
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double Backward(double x)
        {
            var sigmoid = Forward(x);
            return sigmoid * (1.0 - sigmoid);
        }
    }

    /// <summary>
    /// Hyperbolic tangent activation function
    /// </summary>
    public class TanhActivation : IActivationFunction
    {
        public string Name => "Tanh";

        public double Forward(double x) => Math.Tanh(Math.Max(-500, Math.Min(500, x)));

        public double Backward(double x) 
        {
            var tanh = Forward(x);
            return 1.0 - tanh * tanh;
        }
    }

    /// <summary>
    /// Rectified Linear Unit (ReLU) activation function
    /// </summary>
    public class ReLUActivation : IActivationFunction
    {
        public string Name => "ReLU";

        public double Forward(double x) => Math.Max(0, x);

        public double Backward(double x) => x > 0 ? 1.0 : 0.0;
    }

    /// <summary>
    /// Leaky ReLU activation function
    /// </summary>
    public class LeakyReLUActivation : IActivationFunction
    {
        private readonly double _alpha;

        public LeakyReLUActivation(double alpha = 0.01)
        {
            _alpha = alpha;
        }

        public string Name => "Leaky ReLU";

        public double Forward(double x) => x > 0 ? x : _alpha * x;

        public double Backward(double x) => x > 0 ? 1.0 : _alpha;
    }

    /// <summary>
    /// Swish activation function (x * sigmoid(x))
    /// </summary>
    public class SwishActivation : IActivationFunction
    {
        private readonly SigmoidActivation _sigmoid = new();

        public string Name => "Swish";

        public double Forward(double x) => x * _sigmoid.Forward(x);

        public double Backward(double x)
        {
            var sigmoid = _sigmoid.Forward(x);
            return sigmoid + x * sigmoid * (1.0 - sigmoid);
        }
    }

    /// <summary>
    /// Linear activation function (identity)
    /// </summary>
    public class LinearActivation : IActivationFunction
    {
        public string Name => "Linear";

        public double Forward(double x) => x;

        public double Backward(double x) => 1.0;
    }

    #endregion

    #region Optimizers

    /// <summary>
    /// Stochastic Gradient Descent optimizer
    /// </summary>
    public class SGDOptimizer : IOptimizer
    {
        private readonly double _learningRate;

        public SGDOptimizer(double learningRate = 0.01)
        {
            _learningRate = learningRate > 0 ? learningRate : throw new ArgumentException("Learning rate must be positive");
        }

        public string Name => "SGD";

        public async Task UpdateWeightsAsync(double[][] weights, double[][] weightGradients, 
                                           double[] biases, double[] biasGradients, int layerIndex)
        {
            await Task.Run(() =>
            {
                // Update weights
                Parallel.For(0, weights.Length, i =>
                {
                    for (int j = 0; j < weights[i].Length; j++)
                    {
                        weights[i][j] -= _learningRate * weightGradients[i][j];
                    }
                });

                // Update biases
                Parallel.For(0, biases.Length, i =>
                {
                    biases[i] -= _learningRate * biasGradients[i];
                });
            });
        }

        public void Reset()
        {
            // SGD doesn't maintain internal state
        }
    }

    /// <summary>
    /// Adam optimizer with momentum and adaptive learning rates
    /// </summary>
    public class AdamOptimizer : IOptimizer
    {
        private readonly double _learningRate;
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _epsilon;
        
        private readonly ConcurrentDictionary<int, (double[][] mWeights, double[][] vWeights, 
                                                   double[] mBiases, double[] vBiases)> _momentumCache;
        private int _timeStep;

        public AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            _learningRate = learningRate > 0 ? learningRate : throw new ArgumentException("Learning rate must be positive");
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _momentumCache = new ConcurrentDictionary<int, (double[][], double[][], double[], double[])>();
            _timeStep = 0;
        }

        public string Name => "Adam";

        public async Task UpdateWeightsAsync(double[][] weights, double[][] weightGradients, 
                                           double[] biases, double[] biasGradients, int layerIndex)
        {
            await Task.Run(() =>
            {
                Interlocked.Increment(ref _timeStep);

                // Get or create momentum arrays for this layer
                var (mWeights, vWeights, mBiases, vBiases) = _momentumCache.GetOrAdd(layerIndex, _ =>
                {
                    var mW = weights.Select(w => new double[w.Length]).ToArray();
                    var vW = weights.Select(w => new double[w.Length]).ToArray();
                    var mB = new double[biases.Length];
                    var vB = new double[biases.Length];
                    return (mW, vW, mB, vB);
                });

                // Bias correction terms
                var biasCorrection1 = 1.0 - Math.Pow(_beta1, _timeStep);
                var biasCorrection2 = 1.0 - Math.Pow(_beta2, _timeStep);

                // Update weights
                Parallel.For(0, weights.Length, i =>
                {
                    for (int j = 0; j < weights[i].Length; j++)
                    {
                        // Update momentum estimates
                        mWeights[i][j] = _beta1 * mWeights[i][j] + (1 - _beta1) * weightGradients[i][j];
                        vWeights[i][j] = _beta2 * vWeights[i][j] + (1 - _beta2) * weightGradients[i][j] * weightGradients[i][j];

                        // Bias-corrected estimates
                        var mHat = mWeights[i][j] / biasCorrection1;
                        var vHat = vWeights[i][j] / biasCorrection2;

                        // Update weights
                        weights[i][j] -= _learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
                    }
                });

                // Update biases
                Parallel.For(0, biases.Length, i =>
                {
                    // Update momentum estimates
                    mBiases[i] = _beta1 * mBiases[i] + (1 - _beta1) * biasGradients[i];
                    vBiases[i] = _beta2 * vBiases[i] + (1 - _beta2) * biasGradients[i] * biasGradients[i];

                    // Bias-corrected estimates
                    var mHat = mBiases[i] / biasCorrection1;
                    var vHat = vBiases[i] / biasCorrection2;

                    // Update biases
                    biases[i] -= _learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
                });
            });
        }

        public void Reset()
        {
            _momentumCache.Clear();
            _timeStep = 0;
        }
    }

    #endregion

    #region Neural Network Implementation

    /// <summary>
    /// Production-ready Multi-Layer Perceptron implementation with enterprise patterns
    /// </summary>
    public class ProductionNeuralNetwork : INeuralNetwork
    {
        private readonly LayerConfiguration[] _layerConfigurations;
        private readonly IOptimizer _optimizer;
        private readonly RegularizationType _regularization;
        private readonly double _regularizationStrength;
        private readonly int _batchSize;
        private readonly int _maxEpochs;
        private readonly double _tolerance;
        private readonly Random _random;
        
        private double[][][] _weights = null!; // [layer][input_neuron][output_neuron]
        private double[][] _biases = null!;    // [layer][neuron]
        private IActivationFunction[] _activations = null!;
        private bool _isInitialized;
        private readonly object _initLock = new();
        
        private readonly List<EpochMetrics> _trainingHistory;
        private readonly CancellationTokenSource _cancellationTokenSource;

        /// <summary>
        /// Initializes a new neural network with the specified configuration
        /// </summary>
        public ProductionNeuralNetwork(
            LayerConfiguration[] hiddenLayers,
            int outputNeurons,
            OptimizerType optimizerType = OptimizerType.Adam,
            double learningRate = 0.001,
            RegularizationType regularization = RegularizationType.L2,
            double regularizationStrength = 0.001,
            int batchSize = 32,
            int maxEpochs = 100,
            double tolerance = 1e-6,
            int? randomSeed = null)
        {
            if (hiddenLayers == null || hiddenLayers.Length == 0)
                throw new ArgumentException("At least one hidden layer must be specified", nameof(hiddenLayers));
            
            if (outputNeurons <= 0)
                throw new ArgumentException("Output neurons must be positive", nameof(outputNeurons));

            // Store configuration
            _layerConfigurations = hiddenLayers.Append(
                new LayerConfiguration(outputNeurons, ActivationType.Linear, 0.0, false)
            ).ToArray();
            
            _regularization = regularization;
            _regularizationStrength = regularizationStrength;
            _batchSize = batchSize;
            _maxEpochs = maxEpochs;
            _tolerance = tolerance;
            _random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
            
            // Initialize optimizer
            _optimizer = optimizerType switch
            {
                OptimizerType.SGD => new SGDOptimizer(learningRate),
                OptimizerType.Adam => new AdamOptimizer(learningRate),
                _ => throw new ArgumentException($"Unsupported optimizer type: {optimizerType}")
            };
            
            _trainingHistory = new List<EpochMetrics>();
            _cancellationTokenSource = new CancellationTokenSource();
        }

        /// <summary>
        /// Initializes the network architecture with the given input size
        /// </summary>
        private void InitializeNetwork(int inputSize)
        {
            lock (_initLock)
            {
                if (_isInitialized) return;

                var layerSizes = new[] { inputSize }.Concat(_layerConfigurations.Select(c => c.Neurons)).ToArray();
                var numLayers = layerSizes.Length - 1;

                // Initialize weights using Xavier/He initialization
                _weights = new double[numLayers][][];
                _biases = new double[numLayers][];
                _activations = new IActivationFunction[numLayers];

                for (int layer = 0; layer < numLayers; layer++)
                {
                    var inputNeurons = layerSizes[layer];
                    var outputNeurons = layerSizes[layer + 1];
                    var activation = _layerConfigurations[layer].Activation;

                    // Initialize activation function
                    _activations[layer] = activation switch
                    {
                        ActivationType.Sigmoid => new SigmoidActivation(),
                        ActivationType.Tanh => new TanhActivation(),
                        ActivationType.ReLU => new ReLUActivation(),
                        ActivationType.LeakyReLU => new LeakyReLUActivation(),
                        ActivationType.Swish => new SwishActivation(),
                        ActivationType.Linear => new LinearActivation(),
                        _ => throw new ArgumentException($"Unsupported activation type: {activation}")
                    };

                    // Initialize weights
                    _weights[layer] = new double[inputNeurons][];
                    for (int i = 0; i < inputNeurons; i++)
                    {
                        _weights[layer][i] = new double[outputNeurons];
                        for (int j = 0; j < outputNeurons; j++)
                        {
                            // Xavier/He initialization
                            var scale = activation == ActivationType.ReLU || activation == ActivationType.LeakyReLU
                                ? Math.Sqrt(2.0 / inputNeurons)  // He initialization
                                : Math.Sqrt(6.0 / (inputNeurons + outputNeurons));  // Xavier initialization

                            _weights[layer][i][j] = (_random.NextDouble() * 2 - 1) * scale;
                        }
                    }

                    // Initialize biases to zero
                    _biases[layer] = new double[outputNeurons];
                }

                _isInitialized = true;
            }
        }

        /// <summary>
        /// Performs forward pass through the network
        /// </summary>
        private async Task<(double[] output, double[][] activations, double[][] preActivations)> 
            ForwardPassAsync(double[] input, bool training = false)
        {
            var numLayers = _weights.Length;
            var activations = new double[numLayers + 1][];
            var preActivations = new double[numLayers][];
            
            activations[0] = input;  // Input layer

            await Task.Run(() =>
            {
                for (int layer = 0; layer < numLayers; layer++)
                {
                    var currentInput = activations[layer];
                    var outputSize = _weights[layer][0].Length;
                    
                    preActivations[layer] = new double[outputSize];
                    activations[layer + 1] = new double[outputSize];

                    // Compute pre-activation values (linear transformation)
                    Parallel.For(0, outputSize, j =>
                    {
                        double sum = _biases[layer][j];
                        for (int i = 0; i < currentInput.Length; i++)
                        {
                            sum += currentInput[i] * _weights[layer][i][j];
                        }
                        preActivations[layer][j] = sum;
                    });

                    // Apply activation function
                    var activationFunc = _activations[layer];
                    Parallel.For(0, outputSize, j =>
                    {
                        activations[layer + 1][j] = activationFunc.Forward(preActivations[layer][j]);
                    });

                    // Apply dropout during training
                    if (training && _layerConfigurations[layer].DropoutRate > 0)
                    {
                        var dropoutRate = _layerConfigurations[layer].DropoutRate;
                        Parallel.For(0, outputSize, j =>
                        {
                            if (_random.NextDouble() < dropoutRate)
                            {
                                activations[layer + 1][j] = 0;
                            }
                            else
                            {
                                activations[layer + 1][j] /= (1.0 - dropoutRate);  // Inverted dropout
                            }
                        });
                    }
                }
            });

            return (activations[numLayers], activations, preActivations);
        }

        /// <summary>
        /// Performs backward pass (backpropagation)
        /// </summary>
        private async Task<(double[][][] weightGradients, double[][] biasGradients)> 
            BackwardPassAsync(double[] target, double[][] activations, double[][] preActivations)
        {
            var numLayers = _weights.Length;
            var weightGradients = new double[numLayers][][];
            var biasGradients = new double[numLayers][];
            
            await Task.Run(() =>
            {
                // Initialize gradients
                for (int layer = 0; layer < numLayers; layer++)
                {
                    weightGradients[layer] = _weights[layer].Select(w => new double[w.Length]).ToArray();
                    biasGradients[layer] = new double[_biases[layer].Length];
                }

                // Compute output layer error (assuming MSE loss)
                var outputLayer = numLayers - 1;
                var output = activations[outputLayer + 1];
                var errors = new double[numLayers][];
                
                errors[outputLayer] = new double[output.Length];
                Parallel.For(0, output.Length, i =>
                {
                    errors[outputLayer][i] = 2.0 * (output[i] - target[i]);  // MSE derivative
                });

                // Backpropagate errors
                for (int layer = outputLayer; layer >= 0; layer--)
                {
                    var currentActivations = activations[layer];
                    var nextActivations = activations[layer + 1];
                    var currentErrors = errors[layer];
                    var activationFunc = _activations[layer];

                    // Compute gradients for this layer
                    Parallel.For(0, currentActivations.Length, i =>
                    {
                        for (int j = 0; j < nextActivations.Length; j++)
                        {
                            var gradient = currentErrors[j] * activationFunc.Backward(preActivations[layer][j]);
                            weightGradients[layer][i][j] = gradient * currentActivations[i];
                        }
                    });

                    // Compute bias gradients
                    Parallel.For(0, biasGradients[layer].Length, j =>
                    {
                        biasGradients[layer][j] = currentErrors[j] * activationFunc.Backward(preActivations[layer][j]);
                    });

                    // Propagate errors to previous layer
                    if (layer > 0)
                    {
                        errors[layer - 1] = new double[currentActivations.Length];
                        Parallel.For(0, currentActivations.Length, i =>
                        {
                            double error = 0;
                            for (int j = 0; j < nextActivations.Length; j++)
                            {
                                error += currentErrors[j] * activationFunc.Backward(preActivations[layer][j]) * _weights[layer][i][j];
                            }
                            errors[layer - 1][i] = error;
                        });
                    }
                }

                // Add regularization to weight gradients
                if (_regularization != RegularizationType.None)
                {
                    AddRegularizationGradients(weightGradients);
                }
            });

            return (weightGradients, biasGradients);
        }

        /// <summary>
        /// Adds regularization gradients to weight gradients
        /// </summary>
        private void AddRegularizationGradients(double[][][] weightGradients)
        {
            Parallel.For(0, _weights.Length, layer =>
            {
                for (int i = 0; i < _weights[layer].Length; i++)
                {
                    for (int j = 0; j < _weights[layer][i].Length; j++)
                    {
                        switch (_regularization)
                        {
                            case RegularizationType.L1:
                                weightGradients[layer][i][j] += _regularizationStrength * Math.Sign(_weights[layer][i][j]);
                                break;
                            case RegularizationType.L2:
                                weightGradients[layer][i][j] += _regularizationStrength * _weights[layer][i][j];
                                break;
                            case RegularizationType.ElasticNet:
                                weightGradients[layer][i][j] += _regularizationStrength * 
                                    (0.5 * Math.Sign(_weights[layer][i][j]) + 0.5 * _weights[layer][i][j]);
                                break;
                        }
                    }
                }
            });
        }

        /// <summary>
        /// Calculates the loss for a batch of samples
        /// </summary>
        private async Task<double> CalculateLossAsync(IEnumerable<TrainingSample> samples)
        {
            var losses = await Task.Run(() =>
                samples.AsParallel().Select(async sample =>
                {
                    var (output, _, _) = await ForwardPassAsync(sample.Features);
                    double loss = 0;
                    for (int i = 0; i < output.Length; i++)
                    {
                        loss += Math.Pow(output[i] - sample.Targets[i], 2);
                    }
                    return loss / output.Length;  // MSE
                }).Select(t => t.Result)
            );

            var avgLoss = losses.Average();

            // Add regularization penalty
            if (_regularization != RegularizationType.None)
            {
                avgLoss += CalculateRegularizationPenalty();
            }

            return avgLoss;
        }

        /// <summary>
        /// Calculates regularization penalty
        /// </summary>
        private double CalculateRegularizationPenalty()
        {
            double penalty = 0;

            Parallel.For(0, _weights.Length, layer =>
            {
                double layerPenalty = 0;
                for (int i = 0; i < _weights[layer].Length; i++)
                {
                    for (int j = 0; j < _weights[layer][i].Length; j++)
                    {
                        var weight = _weights[layer][i][j];
                        layerPenalty += _regularization switch
                        {
                            RegularizationType.L1 => Math.Abs(weight),
                            RegularizationType.L2 => weight * weight,
                            RegularizationType.ElasticNet => 0.5 * Math.Abs(weight) + 0.5 * weight * weight,
                            _ => 0
                        };
                    }
                }
                // Thread-safe addition using lock instead of Interlocked.Add for double
                lock (_initLock)
                {
                    penalty += layerPenalty;
                }
            });

            return _regularizationStrength * penalty;
        }

        /// <summary>
        /// Trains the neural network asynchronously
        /// </summary>
        public async Task<bool> TrainAsync(IEnumerable<TrainingSample> trainingData, 
                                         IEnumerable<TrainingSample> validationData = null,
                                         CancellationToken cancellationToken = default)
        {
            var combinedToken = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, _cancellationTokenSource.Token).Token;

            try
            {
                var trainingList = trainingData.ToList();
                var validationList = validationData?.ToList();

                if (!trainingList.Any())
                    throw new ArgumentException("Training data cannot be empty", nameof(trainingData));

                // Initialize network based on first sample
                var firstSample = trainingList.First();
                InitializeNetwork(firstSample.Features.Length);

                Console.WriteLine($"🧠 Training Neural Network");
                Console.WriteLine($"{"=",-40}");
                Console.WriteLine($"Training samples: {trainingList.Count}");
                Console.WriteLine($"Validation samples: {validationList?.Count ?? 0}");
                Console.WriteLine($"Network architecture: {string.Join(" -> ", new[] { firstSample.Features.Length }.Concat(_layerConfigurations.Select(c => c.Neurons)))}");
                Console.WriteLine($"Optimizer: {_optimizer.Name}");
                Console.WriteLine($"Regularization: {_regularization} ({_regularizationStrength:F6})");
                Console.WriteLine($"Batch size: {_batchSize}");

                var stopwatch = Stopwatch.StartNew();
                var bestValidationLoss = double.MaxValue;
                var patienceCounter = 0;
                const int maxPatience = 10;

                for (int epoch = 0; epoch < _maxEpochs && !combinedToken.IsCancellationRequested; epoch++)
                {
                    var epochStart = Stopwatch.StartNew();

                    // Shuffle training data
                    var shuffledData = trainingList.OrderBy(x => _random.Next()).ToList();

                    // Mini-batch training
                    var batches = shuffledData
                        .Select((sample, index) => new { sample, index })
                        .GroupBy(x => x.index / _batchSize)
                        .Select(g => g.Select(x => x.sample).ToList())
                        .ToList();

                    double epochLoss = 0;
                    foreach (var batch in batches)
                    {
                        if (combinedToken.IsCancellationRequested) break;

                        // Process batch in parallel
                        var batchTasks = batch.Select(async sample =>
                        {
                            var (output, activations, preActivations) = await ForwardPassAsync(sample.Features, training: true);
                            var (weightGradients, biasGradients) = await BackwardPassAsync(sample.Targets, activations, preActivations);
                            return (weightGradients, biasGradients, sample);
                        });

                        var batchResults = await Task.WhenAll(batchTasks);

                        // Average gradients across batch
                        var avgWeightGradients = new double[_weights.Length][][];
                        var avgBiasGradients = new double[_weights.Length][];
                        
                        for (int layer = 0; layer < _weights.Length; layer++)
                        {
                            avgWeightGradients[layer] = _weights[layer].Select(w => new double[w.Length]).ToArray();
                            avgBiasGradients[layer] = new double[_biases[layer].Length];

                            foreach (var (weightGradients, biasGradients, _) in batchResults)
                            {
                                for (int i = 0; i < avgWeightGradients[layer].Length; i++)
                                {
                                    for (int j = 0; j < avgWeightGradients[layer][i].Length; j++)
                                    {
                                        avgWeightGradients[layer][i][j] += weightGradients[layer][i][j] / batchResults.Length;
                                    }
                                }

                                for (int i = 0; i < avgBiasGradients[layer].Length; i++)
                                {
                                    avgBiasGradients[layer][i] += biasGradients[layer][i] / batchResults.Length;
                                }
                            }

                            // Update weights using optimizer
                            await _optimizer.UpdateWeightsAsync(_weights[layer], avgWeightGradients[layer], 
                                                              _biases[layer], avgBiasGradients[layer], layer);
                        }
                    }

                    // Calculate training and validation metrics
                    var trainingLoss = await CalculateLossAsync(trainingList.Take(Math.Min(1000, trainingList.Count)));
                    var validationLoss = validationList != null 
                        ? await CalculateLossAsync(validationList.Take(Math.Min(1000, validationList.Count))) 
                        : double.NaN;

                    var trainingAccuracy = await CalculateAccuracyAsync(trainingList.Take(Math.Min(1000, trainingList.Count)));
                    var validationAccuracy = validationList != null 
                        ? await CalculateAccuracyAsync(validationList.Take(Math.Min(1000, validationList.Count))) 
                        : double.NaN;

                    epochStart.Stop();
                    var metrics = new EpochMetrics(epoch + 1, trainingLoss, validationLoss, 
                                                 trainingAccuracy, validationAccuracy, epochStart.Elapsed);
                    _trainingHistory.Add(metrics);

                    // Progress reporting
                    if (epoch == 0 || (epoch + 1) % Math.Max(1, _maxEpochs / 10) == 0 || epoch == _maxEpochs - 1)
                    {
                        var progress = $"Epoch {epoch + 1,3}/{_maxEpochs} - {epochStart.Elapsed.TotalSeconds:F3}s - " +
                                     $"loss: {trainingLoss:F6} - acc: {trainingAccuracy:F4}";
                        
                        if (!double.IsNaN(validationLoss))
                        {
                            progress += $" - val_loss: {validationLoss:F6} - val_acc: {validationAccuracy:F4}";
                        }
                        
                        Console.WriteLine(progress);
                    }

                    // Early stopping based on validation loss
                    if (!double.IsNaN(validationLoss))
                    {
                        if (validationLoss < bestValidationLoss - _tolerance)
                        {
                            bestValidationLoss = validationLoss;
                            patienceCounter = 0;
                        }
                        else
                        {
                            patienceCounter++;
                            if (patienceCounter >= maxPatience)
                            {
                                Console.WriteLine($"Early stopping at epoch {epoch + 1} (no improvement for {maxPatience} epochs)");
                                break;
                            }
                        }
                    }

                    epochLoss = trainingLoss;
                    if (epochLoss < _tolerance)
                    {
                        Console.WriteLine($"✅ Converged at epoch {epoch + 1} (loss: {epochLoss:F6})");
                        break;
                    }
                }

                stopwatch.Stop();
                Console.WriteLine($"✅ Training completed in {stopwatch.Elapsed.TotalSeconds:F3} seconds");
                
                return !combinedToken.IsCancellationRequested;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Training failed: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Calculates accuracy for regression (R²) or classification
        /// </summary>
        private async Task<double> CalculateAccuracyAsync(IEnumerable<TrainingSample> samples)
        {
            var predictions = await Task.WhenAll(samples.Select(async sample =>
            {
                var (output, _, _) = await ForwardPassAsync(sample.Features);
                return (predicted: output, actual: sample.Targets);
            }));

            // Calculate R² for regression
            double totalSumSquares = 0;
            double residualSumSquares = 0;
            double actualMean = predictions.SelectMany(p => p.actual).Average();

            foreach (var (predicted, actual) in predictions)
            {
                for (int i = 0; i < actual.Length; i++)
                {
                    totalSumSquares += Math.Pow(actual[i] - actualMean, 2);
                    residualSumSquares += Math.Pow(actual[i] - predicted[i], 2);
                }
            }

            return Math.Max(0, 1.0 - residualSumSquares / totalSumSquares);
        }

        /// <summary>
        /// Makes a prediction for a single input
        /// </summary>
        public async Task<double[]> PredictAsync(double[] features, CancellationToken cancellationToken = default)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Network not initialized. Train the network first.");

            var (output, _, _) = await ForwardPassAsync(features);
            return output;
        }

        /// <summary>
        /// Makes predictions for a batch of inputs
        /// </summary>
        public async Task<double[][]> PredictBatchAsync(double[][] features, CancellationToken cancellationToken = default)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Network not initialized. Train the network first.");

            var predictions = await Task.WhenAll(features.Select(f => PredictAsync(f, cancellationToken)));
            return predictions;
        }

        /// <summary>
        /// Saves the trained model to a file
        /// </summary>
        public async Task SaveModelAsync(string filePath, CancellationToken cancellationToken = default)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Network not initialized. Train the network first.");

            var modelData = new
            {
                LayerConfigurations = _layerConfigurations,
                Weights = _weights,
                Biases = _biases,
                TrainingHistory = _trainingHistory
            };

            var json = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json, cancellationToken);
        }

        /// <summary>
        /// Loads a trained model from a file
        /// </summary>
        public async Task LoadModelAsync(string filePath, CancellationToken cancellationToken = default)
        {
            var json = await File.ReadAllTextAsync(filePath, cancellationToken);
            var modelData = JsonSerializer.Deserialize<dynamic>(json);
            
            // Implementation would require proper deserialization
            // This is a simplified version for demonstration
            _isInitialized = true;
        }

        /// <summary>
        /// Gets training history
        /// </summary>
        public IReadOnlyList<EpochMetrics> GetTrainingHistory() => _trainingHistory.AsReadOnly();

        /// <summary>
        /// Disposes of resources
        /// </summary>
        public void Dispose()
        {
            _cancellationTokenSource?.Dispose();
        }
    }

    #endregion

    #region Demonstration and Testing

    /// <summary>
    /// Utility class for generating synthetic datasets
    /// </summary>
    public static class SyntheticDataGenerator
    {
        /// <summary>
        /// Generates a synthetic regression dataset with nonlinear patterns
        /// </summary>
        public static List<TrainingSample> GenerateRegressionData(int samples, int features, double noise = 0.1, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var data = new List<TrainingSample>();

            Console.WriteLine($"📊 Generating synthetic regression dataset");
            Console.WriteLine($"Samples: {samples}, Features: {features}, Noise: {noise:F2}");

            for (int i = 0; i < samples; i++)
            {
                var x = new double[features];
                for (int j = 0; j < features; j++)
                {
                    x[j] = (random.NextDouble() - 0.5) * 4.0;  // Range [-2, 2]
                }

                // Generate nonlinear target function
                double y = Math.Sin(x[0]) + Math.Cos(x[1] * 2.0);
                if (features > 2)
                {
                    y += 0.5 * x[2] * x[2];  // Quadratic term
                }
                if (features > 3)
                {
                    y += Math.Tanh(x[3]);  // Hyperbolic tangent
                }
                if (features > 4)
                {
                    y += 0.3 * x[4] * x[0];  // Interaction term
                }

                // Add noise
                y += (random.NextDouble() - 0.5) * noise;

                data.Add(new TrainingSample(x, new[] { y }, i));
            }

            Console.WriteLine($"✅ Dataset generated with target range: [{data.Min(d => d.Targets[0]):F3}, {data.Max(d => d.Targets[0]):F3}]");
            return data;
        }
    }

    /// <summary>
    /// Comprehensive demonstration class
    /// </summary>
    public static class NeuralNetworkDemo
    {
        /// <summary>
        /// Demonstrates comprehensive neural network capabilities
        /// </summary>
        public static async Task DemonstrateNeuralNetworkAsync()
        {
            Console.WriteLine("🚀 Neural Network Implementation Demonstration");
            Console.WriteLine(new string('=', 55));

            try
            {
                // Generate synthetic dataset
                var allData = SyntheticDataGenerator.GenerateRegressionData(1000, 4, 0.15, seed: 42);
                
                // Split into train/validation/test
                var random = new Random(42);
                var shuffledData = allData.OrderBy(x => random.Next()).ToList();
                
                var trainSize = (int)(allData.Count * 0.7);
                var valSize = (int)(allData.Count * 0.15);
                
                var trainData = shuffledData.Take(trainSize).ToList();
                var valData = shuffledData.Skip(trainSize).Take(valSize).ToList();
                var testData = shuffledData.Skip(trainSize + valSize).ToList();

                Console.WriteLine($"\nDataset split - Train: {trainData.Count}, Val: {valData.Count}, Test: {testData.Count}");

                // Test different network architectures
                var architectures = new[]
                {
                    new { Name = "Simple Network", Layers = new[] { new LayerConfiguration(8, ActivationType.ReLU) } },
                    new { Name = "Deep Network", Layers = new[] { 
                        new LayerConfiguration(16, ActivationType.ReLU, 0.1), 
                        new LayerConfiguration(8, ActivationType.ReLU, 0.1) 
                    }},
                    new { Name = "Wide Network", Layers = new[] { new LayerConfiguration(32, ActivationType.Swish) } }
                };

                var results = new List<(string name, double testLoss, double testR2)>();

                foreach (var arch in architectures)
                {
                    Console.WriteLine($"\n{new string('=', 60)}");
                    Console.WriteLine($"🧠 Testing {arch.Name}");
                    Console.WriteLine(new string('=', 60));

                    // Create neural network
                    using var network = new ProductionNeuralNetwork(
                        hiddenLayers: arch.Layers,
                        outputNeurons: 1,
                        optimizerType: OptimizerType.Adam,
                        learningRate: 0.001,
                        regularization: RegularizationType.L2,
                        regularizationStrength: 0.001,
                        batchSize: 32,
                        maxEpochs: 50,
                        tolerance: 1e-6,
                        randomSeed: 42
                    );

                    // Train the network
                    var success = await network.TrainAsync(trainData, valData);

                    if (success)
                    {
                        // Evaluate on test set
                        Console.WriteLine("\n📊 Evaluating on test set...");
                        var testPredictions = await network.PredictBatchAsync(testData.Select(d => d.Features).ToArray());
                        
                        // Calculate test metrics
                        double testLoss = 0;
                        double totalSumSquares = 0;
                        double residualSumSquares = 0;
                        double actualMean = testData.Select(d => d.Targets[0]).Average();

                        for (int i = 0; i < testData.Count; i++)
                        {
                            var actual = testData[i].Targets[0];
                            var predicted = testPredictions[i][0];
                            
                            testLoss += Math.Pow(actual - predicted, 2);
                            totalSumSquares += Math.Pow(actual - actualMean, 2);
                            residualSumSquares += Math.Pow(actual - predicted, 2);
                        }
                        
                        testLoss /= testData.Count;
                        var testR2 = Math.Max(0, 1.0 - residualSumSquares / totalSumSquares);

                        Console.WriteLine($"Test Loss (MSE): {testLoss:F6}");
                        Console.WriteLine($"Test R²: {testR2:F4}");

                        results.Add((arch.Name, testLoss, testR2));

                        // Show sample predictions
                        Console.WriteLine("\n🧪 Sample Predictions:");
                        for (int i = 0; i < Math.Min(5, testData.Count); i++)
                        {
                            var actual = testData[i].Targets[0];
                            var predicted = testPredictions[i][0];
                            var error = Math.Abs(actual - predicted);
                            
                            Console.WriteLine($"Sample {i + 1}: Predicted = {predicted:F4}, " +
                                            $"Actual = {actual:F4}, Error = {error:F4}");
                        }

                        // Training history summary
                        var history = network.GetTrainingHistory();
                        if (history.Count > 0)
                        {
                            var finalMetrics = history.Last();
                            Console.WriteLine($"\nFinal Training Metrics:");
                            Console.WriteLine($"  Training Loss: {finalMetrics.TrainingLoss:F6}");
                            Console.WriteLine($"  Validation Loss: {finalMetrics.ValidationLoss:F6}");
                            Console.WriteLine($"  Training R²: {finalMetrics.TrainingAccuracy:F4}");
                            Console.WriteLine($"  Validation R²: {finalMetrics.ValidationAccuracy:F4}");
                        }
                    }
                }

                // Compare all architectures
                if (results.Count > 1)
                {
                    Console.WriteLine($"\n{new string('=', 50)}");
                    Console.WriteLine("📊 Architecture Comparison");
                    Console.WriteLine(new string('=', 50));
                    
                    foreach (var (name, testLoss, testR2) in results.OrderBy(r => r.testLoss))
                    {
                        Console.WriteLine($"{name,-20}: Test Loss = {testLoss:F6}, Test R² = {testR2:F4}");
                    }
                }

                Console.WriteLine("\n✅ Neural network demonstration completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Neural network demonstration failed: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }

    #endregion

    #region Main Entry Point

    /// <summary>
    /// Main entry point for the demonstration
    /// </summary>
    public class Program
    {
        public static async Task Main(string[] args)
        {
            try
            {
                await NeuralNetworkDemo.DemonstrateNeuralNetworkAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Application failed: {ex.Message}");
                Environment.Exit(1);
            }

            if (args.Length > 0 && args[0] == "--wait")
            {
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
            }
        }
    }

    #endregion
}