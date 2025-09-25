/**
 * Production-Ready Machine Learning Patterns in JavaScript/TypeScript
 * ==================================================================
 * 
 * This module demonstrates industry-standard ML patterns in JavaScript/TypeScript
 * with proper error handling, validation, type safety, and production deployment
 * considerations for AI training datasets.
 *
 * Key Features:
 * - TypeScript interfaces for type safety
 * - Comprehensive error handling and validation  
 * - Async/await patterns for ML operations
 * - Memory-efficient data processing
 * - Integration with popular JS ML libraries
 * - Extensive documentation for AI learning
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

// Type definitions for better code clarity
interface DataPoint {
    features: number[];
    target?: number | string;
    id?: string | number;
}

interface ModelMetrics {
    accuracy: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    confusionMatrix?: number[][];
    trainingTime: number;
    predictionTime: number;
    modelSize: number;
}

interface ValidationResult {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    rowCount: number;
    columnCount: number;
    missingValues: { [key: string]: number };
}

interface ModelConfig {
    modelType: 'regression' | 'classification';
    hyperparameters: { [key: string]: any };
    enableValidation: boolean;
    enableMonitoring: boolean;
    randomSeed?: number;
}

// Custom error classes
class DataValidationError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'DataValidationError';
    }
}

class ModelTrainingError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'ModelTrainingError';
    }
}

class PredictionError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'PredictionError';
    }
}

/**
 * Timing decorator for performance monitoring
 */
function timing(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
        const startTime = performance.now();
        try {
            const result = await method.apply(this, args);
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            console.log(`${propertyName} executed in ${executionTime.toFixed(4)}ms`);
            return result;
        } catch (error) {
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            console.error(`${propertyName} failed after ${executionTime.toFixed(4)}ms:`, error);
            throw error;
        }
    };
}

/**
 * Validation decorator for input data
 */
function validateInput(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    
    descriptor.value = function (...args: any[]) {
        // Basic validation for ML function patterns
        for (let i = 0; i < args.length; i++) {
            const arg = args[i];
            if (Array.isArray(arg)) {
                if (arg.length === 0) {
                    throw new DataValidationError(`Empty array at position ${i}`);
                }
                // Check for nested arrays (matrix validation)
                if (Array.isArray(arg[0])) {
                    const expectedLength = arg[0].length;
                    for (const row of arg) {
                        if (!Array.isArray(row) || row.length !== expectedLength) {
                            throw new DataValidationError(`Inconsistent matrix dimensions at position ${i}`);
                        }
                    }
                }
            }
        }
        return method.apply(this, args);
    };
}

/**
 * Comprehensive data validator for JavaScript ML pipelines
 */
class DataValidator {
    private minRows: number;
    private maxMissingRatio: number;
    private requiredColumns: string[];

    constructor(
        minRows: number = 10,
        maxMissingRatio: number = 0.3,
        requiredColumns: string[] = []
    ) {
        this.minRows = minRows;
        this.maxMissingRatio = maxMissingRatio;
        this.requiredColumns = requiredColumns;
    }

    /**
     * Validate a dataset represented as an array of objects
     * @param data - Array of data objects to validate
     * @returns ValidationResult with detailed assessment
     */
    validateDataset(data: DataPoint[]): ValidationResult {
        if (!Array.isArray(data)) {
            throw new TypeError("Input must be an array of data points");
        }

        const result: ValidationResult = {
            isValid: true,
            errors: [],
            warnings: [],
            rowCount: data.length,
            columnCount: 0,
            missingValues: {}
        };

        // Check minimum rows
        if (data.length < this.minRows) {
            result.errors.push(`Insufficient data: ${data.length} rows < ${this.minRows}`);
            result.isValid = false;
        }

        // Check for empty dataset
        if (data.length === 0) {
            result.errors.push("Dataset is empty");
            result.isValid = false;
            return result;
        }

        // Analyze data structure
        const columns = new Set<string>();
        const columnMissingCounts: { [key: string]: number } = {};

        data.forEach((row, index) => {
            if (typeof row !== 'object' || row === null) {
                result.errors.push(`Row ${index} is not a valid object`);
                result.isValid = false;
                return;
            }

            // Collect all column names
            Object.keys(row).forEach(col => columns.add(col));
        });

        const columnNames = Array.from(columns);
        result.columnCount = columnNames.length;

        // Check for required columns
        const missingColumns = this.requiredColumns.filter(col => !columns.has(col));
        if (missingColumns.length > 0) {
            result.errors.push(`Missing required columns: ${missingColumns.join(', ')}`);
            result.isValid = false;
        }

        // Count missing values for each column
        columnNames.forEach(col => {
            columnMissingCounts[col] = 0;
            data.forEach(row => {
                const value = (row as any)[col];
                if (value === null || value === undefined || value === '') {
                    columnMissingCounts[col]++;
                }
            });

            const missingRatio = columnMissingCounts[col] / data.length;
            result.missingValues[col] = columnMissingCounts[col];

            if (missingRatio > this.maxMissingRatio) {
                result.errors.push(
                    `Column '${col}' has ${(missingRatio * 100).toFixed(2)}% missing values ` +
                    `(max allowed: ${(this.maxMissingRatio * 100).toFixed(2)}%)`
                );
                result.isValid = false;
            }
        });

        // Check for duplicate rows
        const uniqueRows = new Set(data.map(row => JSON.stringify(row)));
        if (uniqueRows.size !== data.length) {
            const duplicateCount = data.length - uniqueRows.size;
            result.warnings.push(`Found ${duplicateCount} duplicate rows`);
        }

        return result;
    }

    /**
     * Validate features and target arrays for ML training
     * @param X - Feature matrix
     * @param y - Target array
     * @returns ValidationResult with validation details
     */
    validateFeaturesTarget(X: number[][], y: number[]): ValidationResult {
        const result: ValidationResult = {
            isValid: true,
            errors: [],
            warnings: [],
            rowCount: X.length,
            columnCount: X.length > 0 ? X[0].length : 0,
            missingValues: {}
        };

        // Check shapes
        if (X.length !== y.length) {
            result.errors.push(
                `Feature and target length mismatch: ${X.length} != ${y.length}`
            );
            result.isValid = false;
        }

        // Check for invalid values in features
        let nanCount = 0;
        let infCount = 0;
        
        X.forEach((row, i) => {
            row.forEach((value, j) => {
                if (isNaN(value)) nanCount++;
                if (!isFinite(value)) infCount++;
            });
        });

        if (nanCount > 0) {
            result.errors.push(`Features contain ${nanCount} NaN values`);
            result.isValid = false;
        }

        if (infCount > 0) {
            result.errors.push(`Features contain ${infCount} infinite values`);
            result.isValid = false;
        }

        // Check target values
        const targetNanCount = y.filter(val => isNaN(val)).length;
        if (targetNanCount > 0) {
            result.errors.push(`Target contains ${targetNanCount} NaN values`);
            result.isValid = false;
        }

        return result;
    }
}

/**
 * Simple linear regression implementation for demonstration
 */
class SimpleLinearRegression {
    private weights: number[] = [];
    private bias: number = 0;
    private isTrained: boolean = false;
    private trainingTime: number = 0;

    @timing
    @validateInput
    fit(X: number[][], y: number[]): Promise<SimpleLinearRegression> {
        return new Promise((resolve, reject) => {
            try {
                const startTime = performance.now();
                
                // Validate input
                const validator = new DataValidator();
                const validationResult = validator.validateFeaturesTarget(X, y);
                if (!validationResult.isValid) {
                    throw new DataValidationError(
                        `Data validation failed: ${validationResult.errors.join(', ')}`
                    );
                }

                const numFeatures = X[0].length;
                const numSamples = X.length;

                // Initialize weights with small random values
                this.weights = Array(numFeatures).fill(0).map(() => Math.random() * 0.01);
                this.bias = 0;

                // Simple gradient descent implementation
                const learningRate = 0.01;
                const epochs = 1000;

                for (let epoch = 0; epoch < epochs; epoch++) {
                    // Forward pass
                    const predictions = X.map(row => 
                        row.reduce((sum, feature, idx) => sum + feature * this.weights[idx], this.bias)
                    );

                    // Calculate gradients
                    const weightGradients = Array(numFeatures).fill(0);
                    let biasGradient = 0;

                    for (let i = 0; i < numSamples; i++) {
                        const error = predictions[i] - y[i];
                        biasGradient += error;
                        
                        for (let j = 0; j < numFeatures; j++) {
                            weightGradients[j] += error * X[i][j];
                        }
                    }

                    // Update weights and bias
                    for (let j = 0; j < numFeatures; j++) {
                        this.weights[j] -= (learningRate * weightGradients[j]) / numSamples;
                    }
                    this.bias -= (learningRate * biasGradient) / numSamples;
                }

                this.trainingTime = performance.now() - startTime;
                this.isTrained = true;

                console.log(`Model training completed in ${this.trainingTime.toFixed(4)}ms`);
                resolve(this);

            } catch (error) {
                reject(new ModelTrainingError(`Training failed: ${error.message}`));
            }
        });
    }

    @timing
    @validateInput
    predict(X: number[][]): Promise<number[]> {
        return new Promise((resolve, reject) => {
            if (!this.isTrained) {
                reject(new PredictionError("Model must be trained before making predictions"));
                return;
            }

            try {
                // Basic validation
                X.forEach(row => {
                    if (row.some(val => isNaN(val) || !isFinite(val))) {
                        throw new DataValidationError("Input contains NaN or infinite values");
                    }
                });

                const predictions = X.map(row => 
                    row.reduce((sum, feature, idx) => sum + feature * this.weights[idx], this.bias)
                );

                resolve(predictions);

            } catch (error) {
                reject(new PredictionError(`Prediction failed: ${error.message}`));
            }
        });
    }

    @timing
    async evaluate(X: number[][], y: number[]): Promise<ModelMetrics> {
        if (!this.isTrained) {
            throw new ModelTrainingError("Model must be trained before evaluation");
        }

        try {
            const startTime = performance.now();
            const predictions = await this.predict(X);
            const predictionTime = performance.now() - startTime;

            // Calculate metrics
            const errors = predictions.map((pred, i) => pred - y[i]);
            const mse = errors.reduce((sum, error) => sum + error * error, 0) / errors.length;
            const rmse = Math.sqrt(mse);
            
            // Calculate R²
            const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;
            const totalSumSquares = y.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
            const residualSumSquares = errors.reduce((sum, error) => sum + error * error, 0);
            const rSquared = 1 - (residualSumSquares / totalSumSquares);

            const metrics: ModelMetrics = {
                accuracy: rSquared, // Using R² as accuracy for regression
                trainingTime: this.trainingTime,
                predictionTime: predictionTime,
                modelSize: this.getModelSize()
            };

            console.log(`Model evaluation completed. R²: ${rSquared.toFixed(4)}, RMSE: ${rmse.toFixed(4)}`);

            return metrics;

        } catch (error) {
            console.error(`Model evaluation failed: ${error.message}`);
            throw error;
        }
    }

    getModelSize(): number {
        // Estimate model size in bytes
        return (this.weights.length + 1) * 8; // 8 bytes per float64
    }

    saveModel(): string {
        if (!this.isTrained) {
            throw new ModelTrainingError("Cannot save untrained model");
        }

        return JSON.stringify({
            weights: this.weights,
            bias: this.bias,
            isTrained: this.isTrained,
            trainingTime: this.trainingTime,
            saveTimestamp: Date.now()
        });
    }

    loadModel(modelData: string): SimpleLinearRegression {
        try {
            const data = JSON.parse(modelData);
            this.weights = data.weights;
            this.bias = data.bias;
            this.isTrained = data.isTrained;
            this.trainingTime = data.trainingTime || 0;

            console.log(`Model loaded successfully`);
            return this;

        } catch (error) {
            console.error(`Failed to load model: ${error.message}`);
            throw error;
        }
    }
}

/**
 * Feature engineering utilities for JavaScript ML
 */
class FeatureEngineer {
    private scalers: Map<string, { mean: number; std: number }> = new Map();
    private encoders: Map<string, Map<string, number>> = new Map();

    @timing
    async createPolynomialFeatures(
        data: number[][],
        degree: number = 2
    ): Promise<number[][]> {
        if (degree < 2 || degree > 5) {
            throw new Error("Polynomial degree must be between 2 and 5");
        }

        console.log(`Creating polynomial features of degree ${degree}...`);

        const result = data.map(row => {
            const polyFeatures = [...row]; // Start with original features
            
            // Add squared terms
            for (let i = 0; i < row.length; i++) {
                polyFeatures.push(row[i] * row[i]);
            }
            
            // Add interaction terms
            if (degree >= 2) {
                for (let i = 0; i < row.length; i++) {
                    for (let j = i + 1; j < row.length; j++) {
                        polyFeatures.push(row[i] * row[j]);
                    }
                }
            }
            
            return polyFeatures;
        });

        console.log(`Polynomial features created: ${data[0].length} -> ${result[0].length} features`);
        return result;
    }

    @timing
    async standardScaler(
        data: number[][],
        fit: boolean = true
    ): Promise<number[][]> {
        if (data.length === 0 || data[0].length === 0) {
            throw new Error("Cannot scale empty data");
        }

        const numFeatures = data[0].length;
        const result: number[][] = [];

        if (fit) {
            // Calculate means and standard deviations
            for (let j = 0; j < numFeatures; j++) {
                const column = data.map(row => row[j]);
                const mean = column.reduce((sum, val) => sum + val, 0) / column.length;
                const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / column.length;
                const std = Math.sqrt(variance);
                
                this.scalers.set(`feature_${j}`, { mean, std });
            }
        }

        // Apply scaling
        for (const row of data) {
            const scaledRow: number[] = [];
            for (let j = 0; j < numFeatures; j++) {
                const scaler = this.scalers.get(`feature_${j}`);
                if (!scaler) {
                    throw new Error(`Scaler not found for feature ${j}. Call fit first.`);
                }
                
                const scaledValue = scaler.std === 0 ? 0 : (row[j] - scaler.mean) / scaler.std;
                scaledRow.push(scaledValue);
            }
            result.push(scaledRow);
        }

        console.log(`Standard scaling applied to ${numFeatures} features`);
        return result;
    }

    @timing
    async handleMissingValues(
        data: DataPoint[],
        strategy: 'mean' | 'median' | 'mode' | 'drop' = 'mean'
    ): Promise<DataPoint[]> {
        console.log(`Handling missing values using ${strategy} strategy...`);

        if (strategy === 'drop') {
            // Remove rows with any missing values
            return data.filter(row => {
                return Object.values(row).every(val => 
                    val !== null && val !== undefined && val !== ''
                );
            });
        }

        // For imputation strategies, we need to calculate statistics
        const result = data.map(row => ({ ...row })); // Deep copy
        const columns = Object.keys(result[0] || {});

        for (const col of columns) {
            if (col === 'id') continue; // Skip ID columns

            const values = result
                .map(row => (row as any)[col])
                .filter(val => val !== null && val !== undefined && val !== '');

            if (values.length === 0) continue;

            let fillValue: any;
            
            if (strategy === 'mean' && values.every(val => typeof val === 'number')) {
                fillValue = values.reduce((sum, val) => sum + val, 0) / values.length;
            } else if (strategy === 'median' && values.every(val => typeof val === 'number')) {
                const sorted = values.sort((a, b) => a - b);
                const mid = Math.floor(sorted.length / 2);
                fillValue = sorted.length % 2 === 0 ? 
                    (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
            } else if (strategy === 'mode') {
                const counts = new Map();
                values.forEach(val => counts.set(val, (counts.get(val) || 0) + 1));
                fillValue = Array.from(counts.entries()).reduce((a, b) => a[1] > b[1] ? a : b)[0];
            }

            // Apply imputation
            result.forEach(row => {
                const value = (row as any)[col];
                if (value === null || value === undefined || value === '') {
                    (row as any)[col] = fillValue;
                }
            });
        }

        const originalCount = data.length;
        const resultCount = result.length;
        console.log(`Missing value handling completed. Rows: ${originalCount} -> ${resultCount}`);

        return result;
    }

    @timing
    async detectOutliers(
        data: number[][],
        method: 'iqr' | 'zscore' = 'iqr',
        threshold: number = 1.5
    ): Promise<{ outlierIndices: number[]; outlierInfo: any[] }> {
        console.log(`Detecting outliers using ${method} method...`);

        const outlierIndices: Set<number> = new Set();
        const outlierInfo: any[] = [];
        const numFeatures = data[0].length;

        for (let j = 0; j < numFeatures; j++) {
            const column = data.map(row => row[j]);
            const outliers: number[] = [];

            if (method === 'iqr') {
                // Calculate quartiles
                const sorted = [...column].sort((a, b) => a - b);
                const q1Index = Math.floor(sorted.length * 0.25);
                const q3Index = Math.floor(sorted.length * 0.75);
                const q1 = sorted[q1Index];
                const q3 = sorted[q3Index];
                const iqr = q3 - q1;
                
                const lowerBound = q1 - threshold * iqr;
                const upperBound = q3 + threshold * iqr;

                column.forEach((value, index) => {
                    if (value < lowerBound || value > upperBound) {
                        outliers.push(index);
                        outlierIndices.add(index);
                    }
                });

            } else if (method === 'zscore') {
                const mean = column.reduce((sum, val) => sum + val, 0) / column.length;
                const std = Math.sqrt(
                    column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / column.length
                );

                column.forEach((value, index) => {
                    const zscore = Math.abs((value - mean) / std);
                    if (zscore > threshold) {
                        outliers.push(index);
                        outlierIndices.add(index);
                    }
                });
            }

            if (outliers.length > 0) {
                outlierInfo.push({
                    feature: j,
                    outlierCount: outliers.length,
                    outlierPercentage: (outliers.length / column.length) * 100,
                    method,
                    threshold
                });
            }
        }

        console.log(`Outlier detection completed. Found ${outlierIndices.size} outlier rows`);

        return {
            outlierIndices: Array.from(outlierIndices),
            outlierInfo
        };
    }
}

/**
 * Async utilities for ML operations
 */
class MLUtilities {
    static async sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    static async generateSyntheticData(
        numSamples: number,
        numFeatures: number,
        noiseLevel: number = 0.1
    ): Promise<{ X: number[][]; y: number[] }> {
        console.log(`Generating synthetic dataset: ${numSamples} samples, ${numFeatures} features`);

        const X: number[][] = [];
        const y: number[] = [];
        
        // Generate random weights for the linear relationship
        const trueWeights = Array(numFeatures).fill(0).map(() => Math.random() * 2 - 1);
        const trueBias = Math.random() * 2 - 1;

        for (let i = 0; i < numSamples; i++) {
            const features = Array(numFeatures).fill(0).map(() => Math.random() * 10 - 5);
            const target = features.reduce((sum, feature, idx) => sum + feature * trueWeights[idx], trueBias) +
                          Math.random() * noiseLevel - noiseLevel / 2; // Add noise
            
            X.push(features);
            y.push(target);
        }

        return { X, y };
    }

    static async trainTestSplit<T>(
        X: T[],
        y: number[],
        testSize: number = 0.2,
        randomSeed: number = 42
    ): Promise<{
        XTrain: T[];
        XTest: T[];
        yTrain: number[];
        yTest: number[];
    }> {
        // Simple seeded random number generator for reproducibility
        let seed = randomSeed;
        const random = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };

        // Create indices and shuffle them
        const indices = Array.from({ length: X.length }, (_, i) => i);
        
        // Fisher-Yates shuffle with seeded random
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const testLength = Math.floor(X.length * testSize);
        const testIndices = indices.slice(0, testLength);
        const trainIndices = indices.slice(testLength);

        return {
            XTrain: trainIndices.map(i => X[i]),
            XTest: testIndices.map(i => X[i]),
            yTrain: trainIndices.map(i => y[i]),
            yTest: testIndices.map(i => y[i])
        };
    }
}

/**
 * Comprehensive demonstration of JavaScript/TypeScript ML patterns
 */
async function demonstrateJavaScriptMLPatterns(): Promise<void> {
    console.log("🚀 JavaScript/TypeScript ML Pipeline Demonstration");
    console.log("=".repeat(55));

    try {
        // Generate synthetic data
        const { X, y } = await MLUtilities.generateSyntheticData(1000, 5, 0.1);
        console.log(`✅ Generated dataset: ${X.length} samples, ${X[0].length} features`);

        // Split data
        const { XTrain, XTest, yTrain, yTest } = await MLUtilities.trainTestSplit(X, y, 0.2);
        console.log(`✅ Data split: ${XTrain.length} training, ${XTest.length} test samples`);

        // Initialize validator and validate data
        console.log("\n🔄 Validating data...");
        const validator = new DataValidator(minRows: 100, maxMissingRatio: 0.1);
        const validationResult = validator.validateFeaturesTarget(XTrain, yTrain);
        
        if (!validationResult.isValid) {
            console.error("❌ Data validation failed:", validationResult.errors);
            return;
        }
        console.log("✅ Data validation passed");

        // Train model
        console.log("\n🔄 Training model...");
        const model = new SimpleLinearRegression();
        await model.fit(XTrain, yTrain);
        console.log("✅ Model training completed");

        // Make predictions
        console.log("\n🔮 Making predictions...");
        const predictions = await model.predict(XTest.slice(0, 10));
        console.log(`Sample predictions: [${predictions.slice(0, 5).map(p => p.toFixed(4)).join(', ')}]`);

        // Evaluate model
        console.log("\n📊 Evaluating model...");
        const metrics = await model.evaluate(XTest, yTest);
        console.log(`R²: ${metrics.accuracy.toFixed(4)}`);
        console.log(`Training time: ${metrics.trainingTime.toFixed(4)}ms`);
        console.log(`Prediction time: ${metrics.predictionTime.toFixed(4)}ms`);
        console.log(`Model size: ${metrics.modelSize} bytes`);

        // Feature engineering demonstration
        console.log("\n🔧 Feature Engineering...");
        const engineer = new FeatureEngineer();

        // Create polynomial features
        const polyFeatures = await engineer.createPolynomialFeatures(XTrain.slice(0, 100), 2);
        console.log(`Features after polynomial engineering: ${XTrain[0].length} -> ${polyFeatures[0].length}`);

        // Standard scaling
        const scaledFeatures = await engineer.standardScaler(XTrain.slice(0, 100), true);
        console.log(`Standard scaling applied to ${scaledFeatures[0].length} features`);

        // Outlier detection
        const outlierResults = await engineer.detectOutliers(XTrain, 'iqr', 1.5);
        console.log(`Outlier detection: ${outlierResults.outlierIndices.length} outlier rows found`);

        // Model persistence
        console.log("\n💾 Testing model persistence...");
        const modelData = model.saveModel();
        const newModel = new SimpleLinearRegression().loadModel(modelData);
        console.log("✅ Model save/load completed");

        console.log("\n✅ JavaScript/TypeScript ML Pipeline demonstration completed successfully!");

    } catch (error) {
        console.error("❌ Demonstration failed:", error.message);
        if (error.stack) {
            console.error("Stack trace:", error.stack);
        }
    }
}

// Example of how to handle async operations in a web environment
class WebMLPipeline {
    private model: SimpleLinearRegression;
    private isTraining: boolean = false;
    private trainingProgress: number = 0;

    constructor() {
        this.model = new SimpleLinearRegression();
    }

    async trainWithProgress(
        X: number[][],
        y: number[],
        onProgress?: (progress: number) => void
    ): Promise<void> {
        this.isTraining = true;
        this.trainingProgress = 0;

        try {
            // Simulate training progress reporting
            const progressSteps = [10, 30, 60, 80, 100];
            
            for (const step of progressSteps) {
                this.trainingProgress = step;
                if (onProgress) onProgress(step);
                
                // Simulate training step
                await MLUtilities.sleep(200);
            }

            // Actual training
            await this.model.fit(X, y);
            
            this.trainingProgress = 100;
            if (onProgress) onProgress(100);

        } finally {
            this.isTraining = false;
        }
    }

    getTrainingStatus(): { isTraining: boolean; progress: number } {
        return {
            isTraining: this.isTraining,
            progress: this.trainingProgress
        };
    }

    async predict(X: number[][]): Promise<number[]> {
        if (this.isTraining) {
            throw new Error("Cannot make predictions while training is in progress");
        }
        return await this.model.predict(X);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DataValidator,
        SimpleLinearRegression,
        FeatureEngineer,
        MLUtilities,
        WebMLPipeline,
        demonstrateJavaScriptMLPatterns,
        DataValidationError,
        ModelTrainingError,
        PredictionError
    };
}

// Run demonstration if this is the main module
if (typeof window === 'undefined' && typeof process !== 'undefined') {
    // Node.js environment
    demonstrateJavaScriptMLPatterns().catch(console.error);
}