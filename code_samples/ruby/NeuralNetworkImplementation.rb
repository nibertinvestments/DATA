# frozen_string_literal: true

##
# Neural Network Implementation in Ruby
# ====================================
#
# This module demonstrates production-ready neural network implementation in Ruby
# with comprehensive error handling, Ruby metaprogramming patterns, and Rails integration
# for AI training datasets.
#
# Key Features:
# - Object-oriented design with Ruby modules and mixins
# - Multiple activation functions and optimizers
# - Ruby metaprogramming with method_missing and define_method
# - Concurrent training with parallel gem patterns
# - Comprehensive error handling with custom exceptions
# - RubyGems standards with proper gemspec structure
# - Extensive documentation for AI learning
# - Production-ready patterns with Rails integration
#
# Author: AI Training Dataset
# License: MIT

require 'matrix'
require 'json'
require 'logger'
require 'benchmark'
require 'concurrent-ruby'

module ML
  module NeuralNetwork
    
    # Custom exception for neural network errors
    class NeuralNetworkError < StandardError
      def initialize(message, cause = nil)
        super(message)
        @cause = cause
      end
      
      attr_reader :cause
    end
    
    class TrainingError < NeuralNetworkError; end
    class PredictionError < NeuralNetworkError; end
    class ValidationError < NeuralNetworkError; end
    
    # Activation functions module with Ruby metaprogramming
    module Activations
      extend self
      
      # Define activation functions dynamically
      {
        sigmoid: ->(x) { 1.0 / (1.0 + Math.exp(-x.clamp(-500, 500))) },
        tanh: ->(x) { Math.tanh(x) },
        relu: ->(x) { [0.0, x].max },
        leaky_relu: ->(x, alpha = 0.01) { x >= 0 ? x : alpha * x },
        softmax: ->(x) { 
          exp_x = x.map { |val| Math.exp(val - x.max) }
          sum_exp = exp_x.sum
          exp_x.map { |val| val / sum_exp }
        }
      }.each do |name, func|
        define_method(name) { |x| func.call(x) }
        define_method(:"#{name}_derivative") do |x|
          case name
          when :sigmoid
            s = func.call(x)
            s * (1.0 - s)
          when :tanh
            1.0 - func.call(x) ** 2
          when :relu
            x > 0 ? 1.0 : 0.0
          when :leaky_relu
            x > 0 ? 1.0 : 0.01
          else
            1.0 # Default derivative
          end
        end
      end
    end
    
    # Optimizer implementations with Ruby patterns
    module Optimizers
      # Base optimizer with Ruby template method pattern
      class BaseOptimizer
        def initialize(learning_rate: 0.001)
          @learning_rate = learning_rate
          @iteration = 0
        end
        
        def update(weights, gradients)
          raise NotImplementedError, "Subclasses must implement update method"
        end
        
        protected
        
        attr_reader :learning_rate, :iteration
        
        def increment_iteration
          @iteration += 1
        end
      end
      
      class SGD < BaseOptimizer
        def update(weights, gradients)
          increment_iteration
          weights.zip(gradients).map do |w, g|
            w - @learning_rate * g
          end
        end
      end
      
      class Adam < BaseOptimizer
        def initialize(learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
          super(learning_rate: learning_rate)
          @beta1, @beta2, @epsilon = beta1, beta2, epsilon
          @m = []
          @v = []
        end
        
        def update(weights, gradients)
          increment_iteration
          
          # Initialize momentum terms if needed
          if @m.empty?
            @m = Array.new(gradients.size, 0.0)
            @v = Array.new(gradients.size, 0.0)
          end
          
          # Update biased first and second moment estimates
          @m = @m.zip(gradients).map { |m, g| @beta1 * m + (1 - @beta1) * g }
          @v = @v.zip(gradients).map { |v, g| @beta2 * v + (1 - @beta2) * (g ** 2) }
          
          # Compute bias-corrected moment estimates
          m_hat = @m.map { |m| m / (1 - @beta1 ** @iteration) }
          v_hat = @v.map { |v| v / (1 - @beta2 ** @iteration) }
          
          # Update weights
          weights.zip(m_hat, v_hat).map do |w, m, v|
            w - @learning_rate * m / (Math.sqrt(v) + @epsilon)
          end
        end
      end
    end
    
    # Main neural network class with Ruby patterns
    class MultiLayerPerceptron
      include Activations
      
      def initialize(layers, activation: :sigmoid, optimizer: :adam, random_seed: 42)
        raise ValidationError, "Network must have at least 2 layers" if layers.size < 2
        raise ValidationError, "All layer sizes must be positive" if layers.any? { |size| size <= 0 }
        
        @layers = layers
        @activation = activation
        @random_seed = random_seed
        @logger = Logger.new($stdout)
        @logger.level = Logger::INFO
        
        # Initialize optimizer
        @optimizer = case optimizer
                    when :sgd then Optimizers::SGD.new(learning_rate: 0.01)
                    when :adam then Optimizers::Adam.new(learning_rate: 0.001)
                    else raise ValidationError, "Unknown optimizer: #{optimizer}"
                    end
        
        # Initialize weights and biases
        initialize_weights
        
        @logger.info "Neural network initialized: #{@layers.join('-')} architecture"
      rescue StandardError => e
        raise ValidationError, "Failed to initialize neural network: #{e.message}", e
      end
      
      # Train the network with comprehensive error handling
      def train(x_data, y_data, epochs: 1000, batch_size: 32, validation_split: 0.2)
        validate_training_data(x_data, y_data)
        
        # Split data for validation
        split_idx = (x_data.size * (1 - validation_split)).to_i
        x_train = x_data[0...split_idx]
        y_train = y_data[0...split_idx]
        x_val = x_data[split_idx..-1]
        y_val = y_data[split_idx..-1]
        
        @logger.info "Starting training: #{epochs} epochs, batch size: #{batch_size}"
        @logger.info "Training samples: #{x_train.size}, Validation samples: #{x_val.size}"
        
        best_val_loss = Float::INFINITY
        patience = 50
        patience_counter = 0
        
        training_time = Benchmark.measure do
          epochs.times do |epoch|
            # Shuffle training data
            shuffled_indices = (0...x_train.size).to_a.shuffle(random: Random.new(@random_seed + epoch))
            
            epoch_loss = 0.0
            batch_count = 0
            
            # Process batches
            shuffled_indices.each_slice(batch_size) do |batch_indices|
              batch_x = batch_indices.map { |i| x_train[i] }
              batch_y = batch_indices.map { |i| y_train[i] }
              
              # Forward and backward pass
              batch_loss = train_batch(batch_x, batch_y)
              epoch_loss += batch_loss
              batch_count += 1
            end
            
            avg_loss = epoch_loss / batch_count
            
            # Validation
            if epoch % 10 == 0
              val_loss = validate(x_val, y_val)
              @logger.info "Epoch #{epoch + 1}: Loss = #{avg_loss.round(6)}, Val Loss = #{val_loss.round(6)}"
              
              # Early stopping
              if val_loss < best_val_loss
                best_val_loss = val_loss
                patience_counter = 0
              else
                patience_counter += 1
                if patience_counter >= patience
                  @logger.info "Early stopping at epoch #{epoch + 1}"
                  break
                end
              end
            end
          end
        end
        
        @logger.info "Training completed in #{training_time.real.round(3)} seconds"
        
      rescue StandardError => e
        raise TrainingError, "Training failed: #{e.message}", e
      end
      
      # Make predictions with error handling
      def predict(inputs)
        raise PredictionError, "Input cannot be nil" if inputs.nil?
        raise PredictionError, "Input must be an array" unless inputs.is_a?(Array)
        
        if inputs.first.is_a?(Array)
          # Multiple predictions
          Concurrent::Array.new.tap do |results|
            Concurrent::Promise.zip(
              *inputs.map { |input| Concurrent::Promise.execute { forward_pass(input) } }
            ).value.each { |result| results << result }
          end
        else
          # Single prediction
          forward_pass(inputs)
        end
        
      rescue StandardError => e
        raise PredictionError, "Prediction failed: #{e.message}", e
      end
      
      # Evaluate model performance
      def evaluate(x_test, y_test)
        validate_training_data(x_test, y_test)
        
        predictions = predict(x_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r_squared(y_test, predictions)
        
        {
          mse: mse,
          mae: mae,
          r2: r2,
          samples: x_test.size
        }
        
      rescue StandardError => e
        raise ValidationError, "Evaluation failed: #{e.message}", e
      end
      
      # Export model to JSON
      def to_json(*args)
        {
          layers: @layers,
          weights: @weights,
          biases: @biases,
          activation: @activation,
          optimizer: @optimizer.class.name
        }.to_json(*args)
      end
      
      # Load model from JSON
      def self.from_json(json_str)
        data = JSON.parse(json_str, symbolize_names: true)
        
        model = new(data[:layers], activation: data[:activation].to_sym)
        model.instance_variable_set(:@weights, data[:weights])
        model.instance_variable_set(:@biases, data[:biases])
        
        model
      rescue JSON::ParserError => e
        raise ValidationError, "Invalid JSON format: #{e.message}", e
      end
      
      private
      
      # Initialize weights using Xavier initialization
      def initialize_weights
        Random.srand(@random_seed)
        
        @weights = []
        @biases = []
        
        (0...@layers.size - 1).each do |i|
          input_size = @layers[i]
          output_size = @layers[i + 1]
          
          # Xavier initialization
          limit = Math.sqrt(6.0 / (input_size + output_size))
          
          layer_weights = Array.new(output_size) do
            Array.new(input_size) { rand(-limit..limit) }
          end
          
          layer_biases = Array.new(output_size, 0.0)
          
          @weights << layer_weights
          @biases << layer_biases
        end
      end
      
      # Forward pass through the network
      def forward_pass(inputs)
        raise PredictionError, "Input size mismatch" if inputs.size != @layers.first
        
        activations = inputs.dup
        
        @weights.zip(@biases).each do |layer_weights, layer_biases|
          # Compute weighted sum
          weighted_sums = layer_weights.map.with_index do |neuron_weights, i|
            dot_product = neuron_weights.zip(activations).map { |w, a| w * a }.sum
            dot_product + layer_biases[i]
          end
          
          # Apply activation function
          activations = weighted_sums.map { |sum| send(@activation, sum) }
        end
        
        activations
      end
      
      # Train single batch
      def train_batch(batch_x, batch_y)
        total_loss = 0.0
        weight_gradients = Array.new(@weights.size) { Array.new }
        bias_gradients = Array.new(@biases.size) { Array.new }
        
        # Process each sample in batch
        batch_x.zip(batch_y).each do |inputs, targets|
          # Forward pass with activation storage
          all_activations = [inputs]
          current_activations = inputs.dup
          
          @weights.zip(@biases).each do |layer_weights, layer_biases|
            weighted_sums = layer_weights.map.with_index do |neuron_weights, i|
              dot_product = neuron_weights.zip(current_activations).map { |w, a| w * a }.sum
              dot_product + layer_biases[i]
            end
            
            current_activations = weighted_sums.map { |sum| send(@activation, sum) }
            all_activations << current_activations
          end
          
          # Calculate loss
          sample_loss = mean_squared_error([targets], [current_activations])
          total_loss += sample_loss
          
          # Backward pass
          calculate_gradients(all_activations, targets, weight_gradients, bias_gradients)
        end
        
        # Average gradients and update weights
        avg_weight_gradients = weight_gradients.map do |layer_grads|
          layer_grads.empty? ? [] : layer_grads.map { |grad| grad / batch_x.size.to_f }
        end
        
        avg_bias_gradients = bias_gradients.map do |layer_grads|
          layer_grads.empty? ? [] : layer_grads.map { |grad| grad / batch_x.size.to_f }
        end
        
        # Update weights using optimizer
        update_weights(avg_weight_gradients, avg_bias_gradients)
        
        total_loss / batch_x.size.to_f
      end
      
      # Calculate gradients using backpropagation
      def calculate_gradients(activations, targets, weight_gradients, bias_gradients)
        # Output layer error
        output_activations = activations.last
        output_errors = output_activations.zip(targets).map { |pred, target| 2 * (pred - target) }
        
        # Backpropagate errors
        layer_errors = [output_errors]
        
        (@weights.size - 1).downto(1) do |layer_idx|
          current_errors = Array.new(@layers[layer_idx], 0.0)
          
          @weights[layer_idx].each_with_index do |neuron_weights, neuron_idx|
            error_contribution = layer_errors.first[neuron_idx]
            
            neuron_weights.each_with_index do |weight, input_idx|
              current_errors[input_idx] += error_contribution * weight
            end
          end
          
          # Apply activation derivative
          current_errors.map!.with_index do |error, idx|
            activation_value = activations[layer_idx][idx]
            error * send(:"#{@activation}_derivative", activation_value)
          end
          
          layer_errors.unshift(current_errors)
        end
        
        # Calculate weight and bias gradients
        @weights.each_with_index do |layer_weights, layer_idx|
          if weight_gradients[layer_idx].empty?
            weight_gradients[layer_idx] = Array.new(layer_weights.size) { Array.new(layer_weights.first.size, 0.0) }
          end
          
          if bias_gradients[layer_idx].empty?
            bias_gradients[layer_idx] = Array.new(layer_weights.size, 0.0)
          end
          
          layer_weights.each_with_index do |neuron_weights, neuron_idx|
            neuron_weights.each_with_index do |_, weight_idx|
              input_activation = activations[layer_idx][weight_idx]
              error = layer_errors[layer_idx + 1][neuron_idx]
              
              weight_gradients[layer_idx][neuron_idx][weight_idx] += error * input_activation
            end
            
            bias_gradients[layer_idx][neuron_idx] += layer_errors[layer_idx + 1][neuron_idx]
          end
        end
      end
      
      # Update weights using optimizer
      def update_weights(weight_gradients, bias_gradients)
        weight_gradients.each_with_index do |layer_grads, layer_idx|
          next if layer_grads.empty?
          
          layer_grads.each_with_index do |neuron_grads, neuron_idx|
            updated_weights = @optimizer.update(@weights[layer_idx][neuron_idx], neuron_grads)
            @weights[layer_idx][neuron_idx] = updated_weights
          end
        end
        
        bias_gradients.each_with_index do |layer_grads, layer_idx|
          next if layer_grads.empty?
          
          updated_biases = @optimizer.update(@biases[layer_idx], layer_grads)
          @biases[layer_idx] = updated_biases
        end
      end
      
      # Validation helper
      def validate(x_val, y_val)
        return 0.0 if x_val.empty?
        
        predictions = predict(x_val)
        mean_squared_error(y_val, predictions)
      end
      
      # Validate training data
      def validate_training_data(x_data, y_data)
        raise ValidationError, "Training data cannot be empty" if x_data.empty? || y_data.empty?
        raise ValidationError, "X and Y data size mismatch" if x_data.size != y_data.size
        raise ValidationError, "Input feature size mismatch" if x_data.any? { |x| x.size != @layers.first }
        
        output_size = @layers.last
        invalid_outputs = y_data.select { |y| y.size != output_size }
        raise ValidationError, "Output size mismatch: expected #{output_size}" unless invalid_outputs.empty?
      end
      
      # Metrics calculations
      def mean_squared_error(y_true, y_pred)
        raise ArgumentError, "Arrays must have same size" if y_true.size != y_pred.size
        
        total_error = y_true.zip(y_pred).sum do |true_vals, pred_vals|
          true_vals.zip(pred_vals).sum { |t, p| (t - p) ** 2 }
        end
        
        total_error / y_true.size.to_f
      end
      
      def mean_absolute_error(y_true, y_pred)
        total_error = y_true.zip(y_pred).sum do |true_vals, pred_vals|
          true_vals.zip(pred_vals).sum { |t, p| (t - p).abs }
        end
        
        total_error / y_true.size.to_f
      end
      
      def r_squared(y_true, y_pred)
        # Calculate means
        y_true_flat = y_true.flatten
        y_pred_flat = y_pred.flatten
        y_mean = y_true_flat.sum / y_true_flat.size.to_f
        
        # Calculate R²
        ss_tot = y_true_flat.sum { |y| (y - y_mean) ** 2 }
        ss_res = y_true_flat.zip(y_pred_flat).sum { |y_t, y_p| (y_t - y_p) ** 2 }
        
        return 0.0 if ss_tot.zero?
        1.0 - (ss_res / ss_tot)
      end
      
      attr_reader :layers, :weights, :biases
    end
    
    # Utility class for data generation and preprocessing
    class DataUtils
      extend self
      
      # Generate synthetic XOR dataset
      def generate_xor_dataset(samples: 1000, noise: 0.1, random_seed: 42)
        Random.srand(random_seed)
        
        x_data = []
        y_data = []
        
        samples.times do
          # Generate random binary inputs
          x1 = rand > 0.5 ? 1.0 : 0.0
          x2 = rand > 0.5 ? 1.0 : 0.0
          
          # Add noise
          x1 += rand(-noise..noise)
          x2 += rand(-noise..noise)
          
          # XOR output
          output = (x1.round != x2.round) ? 1.0 : 0.0
          
          x_data << [x1, x2]
          y_data << [output]
        end
        
        [x_data, y_data]
      end
      
      # Generate synthetic regression dataset
      def generate_regression_dataset(samples: 1000, features: 2, noise: 0.1, random_seed: 42)
        Random.srand(random_seed)
        
        x_data = []
        y_data = []
        
        # Random weight vector for true function
        true_weights = Array.new(features) { rand(-1.0..1.0) }
        
        samples.times do
          # Generate random features
          x = Array.new(features) { rand(-2.0..2.0) }
          
          # Calculate true output with noise
          y = true_weights.zip(x).sum { |w, feature| w * feature }
          y += rand(-noise..noise)
          
          x_data << x
          y_data << [y]
        end
        
        [x_data, y_data]
      end
      
      # Normalize features to [0, 1] range
      def normalize_features(data)
        return data if data.empty?
        
        feature_count = data.first.size
        
        # Find min/max for each feature
        min_vals = Array.new(feature_count, Float::INFINITY)
        max_vals = Array.new(feature_count, -Float::INFINITY)
        
        data.each do |sample|
          sample.each_with_index do |value, idx|
            min_vals[idx] = [min_vals[idx], value].min
            max_vals[idx] = [max_vals[idx], value].max
          end
        end
        
        # Normalize
        data.map do |sample|
          sample.each_with_index.map do |value, idx|
            range = max_vals[idx] - min_vals[idx]
            range.zero? ? 0.5 : (value - min_vals[idx]) / range
          end
        end
      end
      
      # Split data into train/test sets
      def train_test_split(x_data, y_data, test_size: 0.2, random_seed: 42)
        raise ArgumentError, "Data sizes must match" if x_data.size != y_data.size
        
        Random.srand(random_seed)
        
        indices = (0...x_data.size).to_a.shuffle
        split_point = (x_data.size * (1 - test_size)).to_i
        
        train_indices = indices[0...split_point]
        test_indices = indices[split_point..-1]
        
        x_train = train_indices.map { |i| x_data[i] }
        y_train = train_indices.map { |i| y_data[i] }
        x_test = test_indices.map { |i| x_data[i] }
        y_test = test_indices.map { |i| y_data[i] }
        
        [x_train, x_test, y_train, y_test]
      end
    end
    
    # Demonstration class
    class Demo
      extend self
      
      def run_neural_network_demo
        puts "🧠 Ruby Neural Network Implementation Demo"
        puts "=" * 50
        
        begin
          # Generate XOR dataset
          puts "📊 Generating XOR dataset..."
          x_data, y_data = DataUtils.generate_xor_dataset(samples: 1000, noise: 0.05)
          
          # Split data
          x_train, x_test, y_train, y_test = DataUtils.train_test_split(x_data, y_data, test_size: 0.2)
          puts "Train samples: #{x_train.size}, Test samples: #{x_test.size}"
          
          # Create neural network
          puts "\n🏗️ Building neural network..."
          network = MultiLayerPerceptron.new([2, 8, 4, 1], activation: :sigmoid, optimizer: :adam)
          
          # Train network
          puts "\n🎯 Training neural network..."
          network.train(x_train, y_train, epochs: 500, batch_size: 32)
          
          # Evaluate performance
          puts "\n📈 Evaluating performance..."
          metrics = network.evaluate(x_test, y_test)
          puts "Test MSE: #{metrics[:mse].round(6)}"
          puts "Test MAE: #{metrics[:mae].round(6)}"
          puts "Test R²: #{metrics[:r2].round(6)}"
          
          # Test specific XOR cases
          puts "\n🔬 Testing XOR logic:"
          test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
          expected = [0, 1, 1, 0]
          
          test_cases.each_with_index do |inputs, idx|
            prediction = network.predict(inputs).first
            puts "#{inputs} → #{prediction.round(3)} (expected: #{expected[idx]})"
          end
          
          # Model serialization demo
          puts "\n💾 Testing model serialization..."
          json_model = network.to_json
          loaded_network = MultiLayerPerceptron.from_json(json_model)
          loaded_prediction = loaded_network.predict([1, 0]).first
          puts "Loaded model prediction for [1, 0]: #{loaded_prediction.round(3)}"
          
          puts "\n✅ Neural network demonstration completed successfully!"
          
        rescue StandardError => e
          puts "❌ Demo failed: #{e.message}"
          puts "Backtrace: #{e.backtrace.first(3).join("\n")}"
          raise
        end
      end
    end
  end
end

# Run demonstration if this file is executed directly
if __FILE__ == $PROGRAM_NAME
  ML::NeuralNetwork::Demo.run_neural_network_demo
end