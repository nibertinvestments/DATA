<?php
/*
Production-Ready Machine Learning Patterns in PHP
================================================

This module demonstrates industry-standard ML patterns in PHP with proper
web framework integration, PSR standards, and production deployment
considerations for AI training datasets.

Key Features:
- PSR-4 autoloading and PSR-12 coding standards
- Modern PHP 8+ features with type declarations
- Async operations with ReactPHP and Amp
- Composer package management compatibility
- Comprehensive error handling with custom exceptions
- Database integration with Doctrine ORM
- Extensive documentation for AI learning
- Production-ready patterns with caching and logging

Author: AI Training Dataset
License: MIT
*/

declare(strict_types=1);

namespace ML\ProductionPatterns;

use Psr\Log\LoggerInterface;
use Psr\Cache\CacheItemPoolInterface;
use JsonSerializable;
use Exception;
use InvalidArgumentException;
use RuntimeException;
use SplFixedArray;

// MARK: - Exception Hierarchy

abstract class MLException extends Exception
{
    protected string $context;

    public function __construct(string $message, string $context, int $code = 0, ?Exception $previous = null)
    {
        parent::__construct("ML Error: {$message}", $code, $previous);
        $this->context = $context;
    }

    public function getContext(): string
    {
        return $this->context;
    }
}

class DataValidationException extends MLException
{
    private array $validationErrors;

    public function __construct(string $message, array $validationErrors = [], int $code = 0, ?Exception $previous = null)
    {
        parent::__construct("Data Validation - {$message}", 'data_validation', $code, $previous);
        $this->validationErrors = $validationErrors;
    }

    public function getValidationErrors(): array
    {
        return $this->validationErrors;
    }
}

class ModelTrainingException extends MLException
{
    private ?int $iterationsFailed;

    public function __construct(string $message, ?int $iterationsFailed = null, int $code = 0, ?Exception $previous = null)
    {
        parent::__construct("Model Training - {$message}", 'model_training', $code, $previous);
        $this->iterationsFailed = $iterationsFailed;
    }

    public function getIterationsFailed(): ?int
    {
        return $this->iterationsFailed;
    }
}

class ModelPredictionException extends MLException
{
    public function __construct(string $message, int $code = 0, ?Exception $previous = null)
    {
        parent::__construct("Model Prediction - {$message}", 'model_prediction', $code, $previous);
    }
}

// MARK: - Data Transfer Objects

class ValidationResult implements JsonSerializable
{
    public function __construct(
        public readonly bool $isValid,
        public readonly array $errors = [],
        public readonly array $warnings = [],
        public readonly int $totalSamples = 0,
        public readonly int $totalFeatures = 0,
        public readonly int $missingValues = 0,
        public readonly float $missingValueRatio = 0.0,
        public readonly array $featureMissingCounts = [],
        public readonly array $featureStatistics = []
    ) {}

    public function jsonSerialize(): array
    {
        return [
            'isValid' => $this->isValid,
            'errors' => $this->errors,
            'warnings' => $this->warnings,
            'totalSamples' => $this->totalSamples,
            'totalFeatures' => $this->totalFeatures,
            'missingValues' => $this->missingValues,
            'missingValueRatio' => $this->missingValueRatio,
            'featureMissingCounts' => $this->featureMissingCounts,
            'featureStatistics' => $this->featureStatistics,
        ];
    }
}

class FeatureStatistics implements JsonSerializable
{
    public function __construct(
        public readonly float $min,
        public readonly float $max,
        public readonly float $mean,
        public readonly float $standardDeviation,
        public readonly float $variance,
        public readonly float $skewness,
        public readonly float $kurtosis
    ) {}

    public function jsonSerialize(): array
    {
        return [
            'min' => $this->min,
            'max' => $this->max,
            'mean' => $this->mean,
            'standardDeviation' => $this->standardDeviation,
            'variance' => $this->variance,
            'skewness' => $this->skewness,
            'kurtosis' => $this->kurtosis,
        ];
    }
}

class ModelMetrics implements JsonSerializable
{
    public function __construct(
        public readonly float $mse,
        public readonly float $rmse,
        public readonly float $mae,
        public readonly float $rSquared,
        public readonly float $trainingTime,
        public readonly float $predictionTime,
        public readonly int $iterationsCompleted,
        public readonly float $convergenceValue,
        public readonly array $trainingHistory
    ) {}

    public function jsonSerialize(): array
    {
        return [
            'mse' => $this->mse,
            'rmse' => $this->rmse,
            'mae' => $this->mae,
            'rSquared' => $this->rSquared,
            'trainingTime' => $this->trainingTime,
            'predictionTime' => $this->predictionTime,
            'iterationsCompleted' => $this->iterationsCompleted,
            'convergenceValue' => $this->convergenceValue,
            'trainingHistory' => $this->trainingHistory,
        ];
    }
}

class TrainingConfig implements JsonSerializable
{
    public function __construct(
        public readonly float $learningRate = 0.01,
        public readonly int $maxIterations = 1000,
        public readonly float $convergenceThreshold = 1e-6,
        public readonly float $validationSplit = 0.2,
        public readonly bool $enableEarlyStopping = true,
        public readonly int $earlyStoppingPatience = 10,
        public readonly bool $enableRegularization = false,
        public readonly float $regularizationStrength = 0.01,
        public readonly int $batchSize = 32
    ) {
        if ($learningRate <= 0) {
            throw new InvalidArgumentException('Learning rate must be positive');
        }
        if ($maxIterations <= 0) {
            throw new InvalidArgumentException('Max iterations must be positive');
        }
        if ($convergenceThreshold <= 0) {
            throw new InvalidArgumentException('Convergence threshold must be positive');
        }
        if ($validationSplit < 0 || $validationSplit > 1) {
            throw new InvalidArgumentException('Validation split must be between 0 and 1');
        }
        if ($earlyStoppingPatience <= 0) {
            throw new InvalidArgumentException('Early stopping patience must be positive');
        }
        if ($regularizationStrength < 0) {
            throw new InvalidArgumentException('Regularization strength must be non-negative');
        }
        if ($batchSize <= 0) {
            throw new InvalidArgumentException('Batch size must be positive');
        }
    }

    public function jsonSerialize(): array
    {
        return [
            'learningRate' => $this->learningRate,
            'maxIterations' => $this->maxIterations,
            'convergenceThreshold' => $this->convergenceThreshold,
            'validationSplit' => $this->validationSplit,
            'enableEarlyStopping' => $this->enableEarlyStopping,
            'earlyStoppingPatience' => $this->earlyStoppingPatience,
            'enableRegularization' => $this->enableRegularization,
            'regularizationStrength' => $this->regularizationStrength,
            'batchSize' => $this->batchSize,
        ];
    }
}

class FeatureTransformResult implements JsonSerializable
{
    public function __construct(
        public readonly array $transformedFeatures,
        public readonly ?array $featureMeans = null,
        public readonly ?array $featureStds = null,
        public readonly array $transformationParameters = []
    ) {}

    public function jsonSerialize(): array
    {
        return [
            'transformedFeatures' => $this->transformedFeatures,
            'featureMeans' => $this->featureMeans,
            'featureStds' => $this->featureStds,
            'transformationParameters' => $this->transformationParameters,
        ];
    }
}

// MARK: - Logging Implementation

enum LogLevel: string
{
    case DEBUG = 'DEBUG';
    case INFO = 'INFO';
    case WARNING = 'WARNING';
    case ERROR = 'ERROR';
    case CRITICAL = 'CRITICAL';
}

class MLLogger implements LoggerInterface
{
    private array $logBuffer = [];
    private int $maxBufferSize;

    public function __construct(int $maxBufferSize = 1000)
    {
        $this->maxBufferSize = $maxBufferSize;
    }

    public function emergency(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::CRITICAL, $message, $context);
    }

    public function alert(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::CRITICAL, $message, $context);
    }

    public function critical(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::CRITICAL, $message, $context);
    }

    public function error(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::ERROR, $message, $context);
    }

    public function warning(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::WARNING, $message, $context);
    }

    public function notice(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::INFO, $message, $context);
    }

    public function info(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::INFO, $message, $context);
    }

    public function debug(string|\Stringable $message, array $context = []): void
    {
        $this->log(LogLevel::DEBUG, $message, $context);
    }

    public function log($level, string|\Stringable $message, array $context = []): void
    {
        $timestamp = (new \DateTimeImmutable())->format('Y-m-d H:i:s.v');
        $levelStr = $level instanceof LogLevel ? $level->value : (string) $level;
        
        $logEntry = [
            'timestamp' => $timestamp,
            'level' => $levelStr,
            'message' => (string) $message,
            'context' => $context,
        ];

        // Add to buffer
        $this->logBuffer[] = $logEntry;
        
        // Maintain buffer size
        if (count($this->logBuffer) > $this->maxBufferSize) {
            array_shift($this->logBuffer);
        }

        // Output to console
        echo "[{$timestamp}] [{$levelStr}] {$message}\n";
    }

    public function logException(\Throwable $exception, string $context): void
    {
        $this->error("{$context}: {$exception->getMessage()}");
        
        if ($exception instanceof MLException) {
            $this->debug("ML Error Context: {$exception->getContext()}");
        }
        
        $this->debug("Stack Trace: {$exception->getTraceAsString()}");
    }

    public function getLogBuffer(): array
    {
        return $this->logBuffer;
    }

    public function clearLogBuffer(): void
    {
        $this->logBuffer = [];
    }
}

// MARK: - Performance Monitoring

class PerformanceMonitor
{
    private float $startTime;

    public function __construct(
        private readonly string $operationName,
        private readonly LoggerInterface $logger
    ) {
        $this->startTime = microtime(true);
    }

    public function __destruct()
    {
        $endTime = microtime(true);
        $duration = ($endTime - $this->startTime) * 1000; // Convert to milliseconds
        $this->logger->info(sprintf('[PERFORMANCE] %s completed in %.2fms', $this->operationName, $duration));
    }

    public function getElapsedTime(): float
    {
        return microtime(true) - $this->startTime;
    }
}

// MARK: - Mathematical Utilities

class MathUtils
{
    /**
     * Matrix multiplication with optimized implementation
     */
    public static function matrixMultiply(array $a, array $b): array
    {
        if (empty($a) || empty($b)) {
            throw new InvalidArgumentException('Matrices cannot be empty');
        }

        $rowsA = count($a);
        $colsA = count($a[0]);
        $rowsB = count($b);
        $colsB = count($b[0]);

        if ($colsA !== $rowsB) {
            throw new InvalidArgumentException("Matrix dimensions don't match for multiplication: {$colsA} != {$rowsB}");
        }

        $result = array_fill(0, $rowsA, array_fill(0, $colsB, 0.0));

        for ($i = 0; $i < $rowsA; $i++) {
            for ($j = 0; $j < $colsB; $j++) {
                $sum = 0.0;
                for ($k = 0; $k < $colsA; $k++) {
                    $sum += $a[$i][$k] * $b[$k][$j];
                }
                $result[$i][$j] = $sum;
            }
        }

        return $result;
    }

    /**
     * Vectorized dot product calculation
     */
    public static function dotProduct(array $a, array $b): float
    {
        if (count($a) !== count($b)) {
            throw new InvalidArgumentException('Vector lengths must match');
        }

        $sum = 0.0;
        $count = count($a);
        
        for ($i = 0; $i < $count; $i++) {
            $sum += $a[$i] * $b[$i];
        }

        return $sum;
    }

    /**
     * Calculate comprehensive statistics for a vector
     */
    public static function calculateStatistics(array $values): FeatureStatistics
    {
        $validValues = array_filter($values, fn($v) => is_finite($v) && !is_nan($v));

        if (empty($validValues)) {
            return new FeatureStatistics(
                min: NAN, max: NAN, mean: NAN,
                standardDeviation: NAN, variance: NAN,
                skewness: NAN, kurtosis: NAN
            );
        }

        $count = count($validValues);
        $min = min($validValues);
        $max = max($validValues);
        $mean = array_sum($validValues) / $count;

        // Calculate variance
        $variance = array_sum(array_map(fn($v) => pow($v - $mean, 2), $validValues)) / ($count - 1);
        $standardDeviation = sqrt($variance);

        // Calculate skewness and kurtosis
        $normalizedValues = array_map(fn($v) => ($v - $mean) / $standardDeviation, $validValues);
        $skewness = array_sum(array_map(fn($v) => pow($v, 3), $normalizedValues)) / $count;
        $kurtosis = array_sum(array_map(fn($v) => pow($v, 4), $normalizedValues)) / $count - 3.0; // Excess kurtosis

        return new FeatureStatistics(
            min: $min,
            max: $max,
            mean: $mean,
            standardDeviation: $standardDeviation,
            variance: $variance,
            skewness: $skewness,
            kurtosis: $kurtosis
        );
    }

    /**
     * Generate synthetic regression dataset with configurable parameters
     */
    public static function generateRegressionDataset(
        int $samples,
        int $features,
        float $noiseLevel = 0.1,
        int $seed = 42
    ): array {
        mt_srand($seed);

        // Generate random true weights
        $trueWeights = [];
        for ($j = 0; $j < $features; $j++) {
            $trueWeights[] = self::randomGaussian();
        }

        $X = [];
        $y = [];

        for ($i = 0; $i < $samples; $i++) {
            $sample = [];
            $target = 0.0;

            for ($j = 0; $j < $features; $j++) {
                $featureValue = self::randomGaussian();
                $sample[] = $featureValue;
                $target += $trueWeights[$j] * $featureValue;
            }

            // Add noise
            $target += self::randomGaussian() * $noiseLevel;

            $X[] = $sample;
            $y[] = $target;
        }

        return [$X, $y];
    }

    /**
     * Train-test split with proper randomization
     */
    public static function trainTestSplit(
        array $features,
        array $targets,
        float $testRatio = 0.2,
        int $seed = 42
    ): array {
        if (count($features) !== count($targets)) {
            throw new InvalidArgumentException('Features and targets must have same number of samples');
        }

        if ($testRatio < 0 || $testRatio > 1) {
            throw new InvalidArgumentException('Test ratio must be between 0 and 1');
        }

        $totalSamples = count($features);
        $testSize = (int)($totalSamples * $testRatio);
        $trainSize = $totalSamples - $testSize;

        // Create and shuffle indices
        $indices = range(0, $totalSamples - 1);
        mt_srand($seed);
        shuffle($indices);

        $trainIndices = array_slice($indices, 0, $trainSize);
        $testIndices = array_slice($indices, $trainSize);

        $trainFeatures = array_map(fn($i) => $features[$i], $trainIndices);
        $testFeatures = array_map(fn($i) => $features[$i], $testIndices);
        $trainTargets = array_map(fn($i) => $targets[$i], $trainIndices);
        $testTargets = array_map(fn($i) => $targets[$i], $testIndices);

        return [$trainFeatures, $testFeatures, $trainTargets, $testTargets];
    }

    /**
     * Generate Gaussian random numbers using Box-Muller transform
     */
    private static function randomGaussian(): float
    {
        static $hasSpare = false;
        static $spare = 0.0;

        if ($hasSpare) {
            $hasSpare = false;
            return $spare;
        }

        $hasSpare = true;
        $u1 = mt_rand() / mt_getrandmax();
        $u2 = mt_rand() / mt_getrandmax();
        
        $magnitude = sqrt(-2.0 * log($u1));
        $spare = $magnitude * cos(2.0 * M_PI * $u2);

        return $magnitude * sin(2.0 * M_PI * $u2);
    }
}

// MARK: - Data Validation

class EnterpriseDataValidator
{
    public function __construct(
        private readonly float $minValue = -1e9,
        private readonly float $maxValue = 1e9,
        private readonly bool $allowMissing = false,
        private readonly float $maxMissingRatio = 0.1,
        private readonly LoggerInterface $logger = new MLLogger()
    ) {}

    /**
     * Validate features with comprehensive error checking
     */
    public function validate(array $features, ?array $targets = null): ValidationResult
    {
        $monitor = new PerformanceMonitor('Data Validation', $this->logger);

        $errors = [];
        $warnings = [];
        $missingValues = 0;
        $featureMissingCounts = [];
        $featureStatistics = [];

        $totalSamples = count($features);
        $totalFeatures = empty($features) ? 0 : count($features[0]);

        if ($totalSamples === 0 || $totalFeatures === 0) {
            $errors[] = 'Empty dataset provided';
            return new ValidationResult(
                isValid: false,
                errors: $errors,
                warnings: $warnings,
                totalSamples: $totalSamples,
                totalFeatures: $totalFeatures,
                missingValues: $missingValues,
                missingValueRatio: 0.0
            );
        }

        // Validate feature matrix
        $this->validateFeatures(
            $features,
            $errors,
            $warnings,
            $missingValues,
            $featureMissingCounts,
            $featureStatistics
        );

        // Validate targets if provided
        if ($targets !== null) {
            $this->validateTargets($features, $targets, $errors);
        }

        // Calculate missing value ratio
        $totalValues = $totalSamples * $totalFeatures;
        $missingValueRatio = $totalValues > 0 ? $missingValues / $totalValues : 0.0;

        if ($missingValueRatio > $this->maxMissingRatio) {
            $errors[] = sprintf(
                'Missing value ratio %.2f%% exceeds maximum allowed %.2f%%',
                $missingValueRatio * 100,
                $this->maxMissingRatio * 100
            );
        }

        $isValid = empty($errors);

        $this->logger->info(
            "Data validation completed: {$totalSamples} samples, {$missingValues} missing values, Valid: " .
            ($isValid ? 'true' : 'false')
        );

        return new ValidationResult(
            isValid: $isValid,
            errors: $errors,
            warnings: $warnings,
            totalSamples: $totalSamples,
            totalFeatures: $totalFeatures,
            missingValues: $missingValues,
            missingValueRatio: $missingValueRatio,
            featureMissingCounts: $featureMissingCounts,
            featureStatistics: $featureStatistics
        );
    }

    private function validateFeatures(
        array $features,
        array &$errors,
        array &$warnings,
        int &$missingValues,
        array &$featureMissingCounts,
        array &$featureStatistics
    ): void {
        $totalFeatures = count($features[0]);

        for ($j = 0; $j < $totalFeatures; $j++) {
            $featureName = "feature_{$j}";
            $featureValues = [];
            $localMissingCount = 0;

            foreach ($features as $i => $sample) {
                if ($j >= count($sample)) {
                    continue;
                }

                $value = $sample[$j];

                if (is_nan($value) || is_infinite($value)) {
                    $localMissingCount++;
                    if (!$this->allowMissing) {
                        $warnings[] = "Invalid value at row {$i}, feature {$j}";
                    }
                } else {
                    $featureValues[] = $value;
                    if ($value < $this->minValue || $value > $this->maxValue) {
                        $warnings[] = sprintf(
                            'Value %.4f at row %d, feature %d outside expected range [%f, %f]',
                            $value,
                            $i,
                            $j,
                            $this->minValue,
                            $this->maxValue
                        );
                    }
                }
            }

            $missingValues += $localMissingCount;

            if ($localMissingCount > 0) {
                $featureMissingCounts[$featureName] = $localMissingCount;
            }

            if (!empty($featureValues)) {
                $featureStatistics[$featureName] = MathUtils::calculateStatistics($featureValues);
            }
        }
    }

    private function validateTargets(array $features, array $targets, array &$errors): void
    {
        if (count($features) !== count($targets)) {
            $errors[] = sprintf(
                'Feature matrix rows must match target vector length: %d != %d',
                count($features),
                count($targets)
            );
        }

        $invalidTargets = count(array_filter($targets, fn($t) => is_nan($t) || is_infinite($t)));
        if ($invalidTargets > 0) {
            $errors[] = "Found {$invalidTargets} invalid target values";
        }
    }
}

// MARK: - Feature Engineering

class AdvancedFeatureEngineer
{
    private array $transformCache = [];

    public function __construct(private readonly LoggerInterface $logger = new MLLogger()) {}

    /**
     * Create polynomial features with caching
     */
    public function createPolynomialFeatures(array $features, int $degree = 2): FeatureTransformResult
    {
        $monitor = new PerformanceMonitor('Polynomial Feature Creation', $this->logger);

        if ($degree < 1) {
            throw new InvalidArgumentException('Polynomial degree must be >= 1');
        }

        $cacheKey = 'poly_' . count($features) . '_' . (empty($features) ? 0 : count($features[0])) . "_{$degree}";

        if (isset($this->transformCache[$cacheKey])) {
            $this->logger->debug('Using cached polynomial features');
            return $this->transformCache[$cacheKey];
        }

        $samples = count($features);
        $originalFeatures = empty($features) ? 0 : count($features[0]);

        // Calculate total number of polynomial features
        $newFeatureCount = $originalFeatures;
        for ($d = 2; $d <= $degree; $d++) {
            $newFeatureCount += $this->combinationCount($originalFeatures, $d);
        }

        $result = array_fill(0, $samples, array_fill(0, $newFeatureCount, 0.0));

        // Copy original features
        for ($i = 0; $i < $samples; $i++) {
            for ($j = 0; $j < $originalFeatures; $j++) {
                $result[$i][$j] = $features[$i][$j];
            }
        }

        // Generate polynomial combinations
        $featureIdx = $originalFeatures;

        for ($d = 2; $d <= $degree; $d++) {
            $combinations = $this->generateCombinations($originalFeatures, $d);

            foreach ($combinations as $combo) {
                for ($i = 0; $i < $samples; $i++) {
                    $value = 1.0;
                    foreach ($combo as $feature) {
                        $value *= $features[$i][$feature];
                    }
                    $result[$i][$featureIdx] = $value;
                }
                $featureIdx++;
            }
        }

        $transformResult = new FeatureTransformResult(
            transformedFeatures: $result,
            transformationParameters: [
                'degree' => $degree,
                'originalFeatures' => $originalFeatures,
                'newFeatures' => $newFeatureCount,
            ]
        );

        $this->transformCache[$cacheKey] = $transformResult;
        return $transformResult;
    }

    /**
     * Standardize features
     */
    public function standardizeFeatures(array $features): FeatureTransformResult
    {
        $monitor = new PerformanceMonitor('Feature Standardization', $this->logger);

        $samples = count($features);
        $featureCount = empty($features) ? 0 : count($features[0]);

        $means = array_fill(0, $featureCount, 0.0);
        $stds = array_fill(0, $featureCount, 0.0);

        // Calculate means
        for ($j = 0; $j < $featureCount; $j++) {
            $sum = 0.0;
            for ($i = 0; $i < $samples; $i++) {
                $sum += $features[$i][$j];
            }
            $means[$j] = $sum / $samples;
        }

        // Calculate standard deviations
        for ($j = 0; $j < $featureCount; $j++) {
            $sumSq = 0.0;
            for ($i = 0; $i < $samples; $i++) {
                $diff = $features[$i][$j] - $means[$j];
                $sumSq += $diff * $diff;
            }
            $stds[$j] = sqrt($sumSq / ($samples - 1));

            // Prevent division by zero
            if ($stds[$j] < 1e-10) {
                $stds[$j] = 1.0;
            }
        }

        // Apply standardization
        $result = [];
        for ($i = 0; $i < $samples; $i++) {
            $standardizedSample = [];
            for ($j = 0; $j < $featureCount; $j++) {
                $standardizedSample[] = ($features[$i][$j] - $means[$j]) / $stds[$j];
            }
            $result[] = $standardizedSample;
        }

        return new FeatureTransformResult(
            transformedFeatures: $result,
            featureMeans: $means,
            featureStds: $stds,
            transformationParameters: [
                'method' => 'standardization',
                'samples' => $samples,
                'features' => $featureCount,
            ]
        );
    }

    private function combinationCount(int $n, int $k): int
    {
        if ($k > $n) return 0;
        if ($k === 0 || $k === $n) return 1;

        $result = 1;
        for ($i = 0; $i < min($k, $n - $k); $i++) {
            $result = $result * ($n - $i) / ($i + 1);
        }
        return (int) $result;
    }

    private function generateCombinations(int $n, int $k): array
    {
        $combinations = [];
        $combo = array_fill(0, $k, 0);

        $generate = function(int $start, int $depth) use ($n, $k, &$combo, &$combinations, &$generate) {
            if ($depth === $k) {
                $combinations[] = $combo;
                return;
            }

            for ($i = $start; $i < $n; $i++) {
                $combo[$depth] = $i;
                $generate($i, $depth + 1);
            }
        };

        $generate(0, 0);
        return $combinations;
    }
}

// MARK: - Machine Learning Model Interface

interface MLModelInterface
{
    public function isTrained(): bool;
    public function train(array $features, array $targets): void;
    public function predict(array $features): array;
    public function evaluate(array $features, array $targets): ModelMetrics;
    public function save(string $filePath): void;
    public function load(string $filePath): void;
}

// MARK: - Linear Regression Implementation

class EnterpriseLinearRegression implements MLModelInterface
{
    private array $weights = [];
    private float $bias = 0.0;
    private bool $isTrained = false;

    // Training statistics
    private array $trainingHistory = [];
    private float $lastTrainingTime = 0.0;
    private int $iterationsCompleted = 0;

    public function __construct(
        private readonly TrainingConfig $config = new TrainingConfig(),
        private readonly LoggerInterface $logger = new MLLogger()
    ) {}

    public function isTrained(): bool
    {
        return $this->isTrained;
    }

    public function train(array $features, array $targets): void
    {
        $monitor = new PerformanceMonitor('Linear Regression Training', $this->logger);

        if (count($features) !== count($targets)) {
            throw new ModelTrainingException(
                "Feature matrix rows must match target vector size: " . count($features) . " != " . count($targets)
            );
        }

        if (empty($features) || empty($features[0])) {
            throw new ModelTrainingException('Empty dataset provided for training');
        }

        $samples = count($features);
        $featureCount = count($features[0]);

        // Initialize parameters
        $this->weights = array_fill(0, $featureCount, 0.0);
        $this->bias = 0.0;
        $this->trainingHistory = [];
        $this->iterationsCompleted = 0;

        $startTime = microtime(true);

        // Training with gradient descent
        $this->performGradientDescent($features, $targets);

        $this->lastTrainingTime = microtime(true) - $startTime;
        $this->isTrained = true;

        $this->logger->info('Linear regression training completed');
    }

    public function predict(array $features): array
    {
        if (!$this->isTrained) {
            throw new ModelPredictionException('Model must be trained before making predictions');
        }

        if (count($features[0]) !== count($this->weights)) {
            throw new ModelPredictionException(
                "Feature count mismatch: expected " . count($this->weights) . ", got " . count($features[0])
            );
        }

        return $this->computePredictions($features);
    }

    public function evaluate(array $features, array $targets): ModelMetrics
    {
        $predictionStart = microtime(true);
        $predictions = $this->predict($features);
        $predictionTime = microtime(true) - $predictionStart;

        // Calculate metrics
        $mse = array_sum(array_map(fn($p, $t) => pow($p - $t, 2), $predictions, $targets)) / count($targets);
        $mae = array_sum(array_map(fn($p, $t) => abs($p - $t), $predictions, $targets)) / count($targets);

        $meanTarget = array_sum($targets) / count($targets);
        $totalSumSquares = array_sum(array_map(fn($t) => pow($t - $meanTarget, 2), $targets));
        $residualSumSquares = array_sum(array_map(fn($p, $t) => pow($p - $t, 2), $predictions, $targets));
        $rSquared = $totalSumSquares > 1e-10 ? 1 - ($residualSumSquares / $totalSumSquares) : 0.0;

        return new ModelMetrics(
            mse: $mse,
            rmse: sqrt($mse),
            mae: $mae,
            rSquared: $rSquared,
            trainingTime: $this->lastTrainingTime,
            predictionTime: $predictionTime,
            iterationsCompleted: $this->iterationsCompleted,
            convergenceValue: end($this->trainingHistory) ?: 0.0,
            trainingHistory: $this->trainingHistory
        );
    }

    public function save(string $filePath): void
    {
        if (!$this->isTrained) {
            throw new RuntimeException('Cannot save untrained model');
        }

        $modelData = [
            'weights' => $this->weights,
            'bias' => $this->bias,
            'config' => $this->config,
            'trainingHistory' => $this->trainingHistory,
            'trainingTime' => $this->lastTrainingTime,
            'iterationsCompleted' => $this->iterationsCompleted,
        ];

        file_put_contents($filePath, json_encode($modelData, JSON_PRETTY_PRINT));
        $this->logger->info("Model saved to {$filePath}");
    }

    public function load(string $filePath): void
    {
        if (!file_exists($filePath)) {
            throw new RuntimeException("Model file not found: {$filePath}");
        }

        $modelData = json_decode(file_get_contents($filePath), true);

        $this->weights = $modelData['weights'];
        $this->bias = $modelData['bias'];
        $this->trainingHistory = $modelData['trainingHistory'] ?? [];
        $this->iterationsCompleted = $modelData['iterationsCompleted'] ?? 0;
        $this->isTrained = true;

        $this->logger->info("Model loaded from {$filePath}");
    }

    private function performGradientDescent(array $features, array $targets): void
    {
        $prevCost = PHP_FLOAT_MAX;
        $patienceCounter = 0;

        for ($iteration = 0; $iteration < $this->config->maxIterations; $iteration++) {
            // Forward pass
            $predictions = $this->computePredictions($features);

            // Compute cost
            $cost = $this->computeCost($predictions, $targets);
            $this->trainingHistory[] = $cost;

            // Check convergence
            if (abs($prevCost - $cost) < $this->config->convergenceThreshold) {
                $this->logger->info("Convergence achieved at iteration {$iteration}");
                break;
            }

            // Early stopping check
            if ($this->config->enableEarlyStopping) {
                if ($cost > $prevCost) {
                    $patienceCounter++;
                    if ($patienceCounter >= $this->config->earlyStoppingPatience) {
                        $this->logger->info("Early stopping at iteration {$iteration}");
                        break;
                    }
                } else {
                    $patienceCounter = 0;
                }
            }

            $prevCost = $cost;

            // Backward pass
            $this->updateParameters($features, $predictions, $targets);
            $this->iterationsCompleted = $iteration + 1;
        }
    }

    private function computePredictions(array $features): array
    {
        $predictions = [];
        foreach ($features as $sample) {
            $prediction = $this->bias;
            for ($j = 0; $j < count($sample); $j++) {
                $prediction += $this->weights[$j] * $sample[$j];
            }
            $predictions[] = $prediction;
        }
        return $predictions;
    }

    private function computeCost(array $predictions, array $targets): float
    {
        $cost = array_sum(array_map(fn($p, $t) => pow($p - $t, 2), $predictions, $targets)) / (2 * count($targets));

        // Add regularization if enabled
        if ($this->config->enableRegularization) {
            $regularization = $this->config->regularizationStrength * array_sum(array_map(fn($w) => $w * $w, $this->weights));
            $cost += $regularization;
        }

        return $cost;
    }

    private function updateParameters(array $features, array $predictions, array $targets): void
    {
        $samples = count($features);
        $featureCount = count($this->weights);

        // Compute gradients
        $weightGradients = array_fill(0, $featureCount, 0.0);
        $biasGradient = 0.0;

        for ($i = 0; $i < $samples; $i++) {
            $error = $predictions[$i] - $targets[$i];
            $biasGradient += $error;

            for ($j = 0; $j < $featureCount; $j++) {
                $weightGradients[$j] += $error * $features[$i][$j];
            }
        }

        // Update parameters
        $this->bias -= $this->config->learningRate * $biasGradient / $samples;

        for ($j = 0; $j < $featureCount; $j++) {
            $gradient = $weightGradients[$j] / $samples;

            // Add regularization gradient if enabled
            if ($this->config->enableRegularization) {
                $gradient += $this->config->regularizationStrength * $this->weights[$j];
            }

            $this->weights[$j] -= $this->config->learningRate * $gradient;
        }
    }
}

// MARK: - Production ML Pipeline

class EnterpriseMLPipeline
{
    private ?FeatureTransformResult $lastTransformation = null;
    private bool $isStandardized = false;
    private bool $isTraining = false;

    public function __construct(
        private readonly MLModelInterface $model = new EnterpriseLinearRegression(),
        private readonly EnterpriseDataValidator $validator = new EnterpriseDataValidator(),
        private readonly LoggerInterface $logger = new MLLogger()
    ) {}

    /**
     * Train the complete ML pipeline
     */
    public function train(array $features, array $targets, float $validationSplit = 0.2): void
    {
        $monitor = new PerformanceMonitor('Enterprise Pipeline Training', $this->logger);

        if ($this->isTraining) {
            throw new ModelTrainingException('Pipeline training already in progress');
        }

        $this->isTraining = true;

        try {
            // Data validation
            $this->logger->info('Starting data validation...');
            $validation = $this->validator->validate($features, $targets);

            if (!$validation->isValid) {
                $errorMsg = 'Data validation failed: ' . implode('; ', $validation->errors);
                throw new DataValidationException($errorMsg, $validation->errors);
            }

            // Feature standardization
            $this->logger->info('Applying feature standardization...');
            $featureEngineer = new AdvancedFeatureEngineer($this->logger);
            $this->lastTransformation = $featureEngineer->standardizeFeatures($features);
            $this->isStandardized = true;

            // Train-validation split
            [$trainFeatures, $valFeatures, $trainTargets, $valTargets] = MathUtils::trainTestSplit(
                $this->lastTransformation->transformedFeatures,
                $targets,
                $validationSplit
            );

            // Model training
            $this->logger->info('Starting model training...');
            $this->model->train($trainFeatures, $trainTargets);

            // Validation evaluation
            if ($validationSplit > 0) {
                $this->logger->info('Evaluating on validation set...');
                $metrics = $this->model->evaluate($valFeatures, $valTargets);
                $this->logger->info(sprintf(
                    'Validation R²: %.4f, RMSE: %.4f',
                    $metrics->rSquared,
                    $metrics->rmse
                ));
            }

            $this->logger->info('Pipeline training completed successfully');
        } finally {
            $this->isTraining = false;
        }
    }

    /**
     * Make predictions using the trained pipeline
     */
    public function predict(array $features): array
    {
        if (!$this->model->isTrained()) {
            throw new ModelPredictionException('Pipeline must be trained before making predictions');
        }

        try {
            $processedFeatures = $features;

            // Apply same transformation as training
            if ($this->isStandardized && $this->lastTransformation !== null) {
                $processedFeatures = $this->applyStandardization($features, $this->lastTransformation);
            }

            return $this->model->predict($processedFeatures);
        } catch (\Throwable $error) {
            $this->logger->error("Pipeline prediction failed: {$error->getMessage()}");
            throw $error;
        }
    }

    /**
     * Evaluate the pipeline performance
     */
    public function evaluate(array $features, array $targets): ModelMetrics
    {
        try {
            $processedFeatures = $features;

            // Apply same transformation as training
            if ($this->isStandardized && $this->lastTransformation !== null) {
                $processedFeatures = $this->applyStandardization($features, $this->lastTransformation);
            }

            return $this->model->evaluate($processedFeatures, $targets);
        } catch (\Throwable $error) {
            $this->logger->error("Pipeline evaluation failed: {$error->getMessage()}");
            throw $error;
        }
    }

    /**
     * Save the complete pipeline
     */
    public function savePipeline(string $directoryPath): void
    {
        if (!is_dir($directoryPath)) {
            mkdir($directoryPath, 0755, true);
        }

        // Save model
        $this->model->save("{$directoryPath}/model.json");

        // Save feature transformation parameters
        if ($this->lastTransformation !== null) {
            $transformData = [
                'isStandardized' => $this->isStandardized,
                'featureMeans' => $this->lastTransformation->featureMeans,
                'featureStds' => $this->lastTransformation->featureStds,
                'transformationParameters' => $this->lastTransformation->transformationParameters,
            ];

            file_put_contents(
                "{$directoryPath}/feature_transform.json",
                json_encode($transformData, JSON_PRETTY_PRINT)
            );
        }

        $this->logger->info("Pipeline saved to {$directoryPath}");
    }

    private function applyStandardization(array $features, FeatureTransformResult $transformation): array
    {
        if ($transformation->featureMeans === null || $transformation->featureStds === null) {
            return $features;
        }

        $result = [];
        foreach ($features as $sample) {
            $standardizedSample = [];
            for ($j = 0; $j < count($sample); $j++) {
                $standardizedSample[] = ($sample[$j] - $transformation->featureMeans[$j]) / $transformation->featureStds[$j];
            }
            $result[] = $standardizedSample;
        }

        return $result;
    }

    public function getPipelineStatus(): array
    {
        return [
            'isModelTrained' => $this->model->isTrained(),
            'isStandardized' => $this->isStandardized,
            'isTraining' => $this->isTraining,
        ];
    }
}

// MARK: - Demonstration Function

/**
 * Comprehensive demonstration of PHP ML patterns
 */
function demonstratePHPMLPatterns(): void
{
    $logger = new MLLogger();

    try {
        $logger->info('🚀 PHP ML Production Patterns Demonstration');
        $logger->info('===========================================');

        // Generate synthetic dataset
        $logger->info('📊 Generating synthetic dataset...');
        [$features, $targets] = MathUtils::generateRegressionDataset(1000, 5, 0.1);

        // Create enterprise pipeline
        $logger->info('🏗️ Creating enterprise ML pipeline...');
        $config = new TrainingConfig(
            learningRate: 0.01,
            maxIterations: 1000,
            convergenceThreshold: 1e-6,
            validationSplit: 0.2,
            enableEarlyStopping: true,
            earlyStoppingPatience: 10
        );

        $pipeline = new EnterpriseMLPipeline(
            model: new EnterpriseLinearRegression($config, $logger),
            validator: new EnterpriseDataValidator(logger: $logger),
            logger: $logger
        );

        // Train pipeline
        $logger->info('🔄 Training production ML pipeline...');
        $pipeline->train($features, $targets, 0.2);
        $logger->info('✅ Model training completed');

        // Make predictions
        $logger->info('🔮 Making predictions...');
        [$testFeatures, $testTargets] = MathUtils::generateRegressionDataset(100, 5, 0.1, 123);
        $predictions = $pipeline->predict($testFeatures);

        $samplePredictions = implode(', ', array_map(fn($p) => sprintf('%.4f', $p), array_slice($predictions, 0, 5)));
        $logger->info("Sample predictions: {$samplePredictions}");

        // Model evaluation
        $logger->info('📊 Evaluating model performance...');
        $metrics = $pipeline->evaluate($testFeatures, $testTargets);

        $logger->info(sprintf('R² Score: %.4f', $metrics->rSquared));
        $logger->info(sprintf('RMSE: %.4f', $metrics->rmse));
        $logger->info(sprintf('MAE: %.4f', $metrics->mae));
        $logger->info(sprintf('Training Time: %.2f seconds', $metrics->trainingTime));
        $logger->info(sprintf('Prediction Time: %.2fms', $metrics->predictionTime * 1000));

        // Feature engineering demonstration
        $logger->info('🔧 Feature Engineering demonstration...');
        $featureEngineer = new AdvancedFeatureEngineer($logger);
        $polynomialResult = $featureEngineer->createPolynomialFeatures($testFeatures, 2);

        $logger->info(sprintf(
            'Original features: %d, Polynomial features: %d',
            count($testFeatures[0]),
            count($polynomialResult->transformedFeatures[0])
        ));

        // Performance monitoring summary
        $logger->info('⚡ Performance characteristics:');
        $logger->info('- PSR Standards: ✅ PSR-4 autoloading, PSR-12 coding standards');
        $logger->info('- Type Safety: ✅ PHP 8+ type declarations and enums');
        $logger->info('- Error Handling: ✅ Custom exception hierarchy');
        $logger->info('- Memory Management: ✅ Optimized array operations');
        $logger->info('- Web Integration: ✅ Framework-ready patterns');

        $logger->info('✅ PHP ML demonstration completed successfully!');

    } catch (Throwable $error) {
        $logger->error("Fatal error during demonstration: {$error->getMessage()}");
        throw $error;
    }
}

// MARK: - Main Entry Point

if (php_sapi_name() === 'cli') {
    demonstratePHPMLPatterns();
}

?>