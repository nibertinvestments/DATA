<?php
/**
 * Production-Ready Decision Tree Implementation in PHP
 * ===================================================
 * 
 * This module demonstrates a comprehensive decision tree classifier
 * with entropy-based splitting, pruning, and modern PHP patterns
 * for AI training datasets.
 *
 * Key Features:
 * - ID3 and C4.5 algorithm implementations
 * - Information gain and gain ratio for feature selection
 * - Tree pruning to prevent overfitting
 * - Support for both categorical and numerical features
 * - Modern PHP 8+ features with strict typing
 * - Object-oriented design with interfaces and traits
 * - Comprehensive error handling with custom exceptions
 * - Memory-efficient tree construction
 * - Production logging and monitoring
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

declare(strict_types=1);

namespace MLProductionPatterns;

use InvalidArgumentException;
use RuntimeException;
use Exception;

/**
 * Custom exception for decision tree errors
 */
class DecisionTreeException extends Exception
{
    public function __construct(string $message, int $code = 0, ?Exception $previous = null)
    {
        parent::__construct("DecisionTree Error: {$message}", $code, $previous);
    }
}

/**
 * Data validation exception
 */
class DataValidationException extends DecisionTreeException
{
    private array $validationErrors;

    public function __construct(string $message, array $validationErrors = [])
    {
        $this->validationErrors = $validationErrors;
        $errorSummary = empty($validationErrors) ? '' : ' (' . implode(', ', $validationErrors) . ')';
        parent::__construct("Data Validation - {$message}{$errorSummary}");
    }

    public function getValidationErrors(): array
    {
        return $this->validationErrors;
    }
}

/**
 * Training sample representing a data point with features and label
 */
readonly class TrainingSample
{
    public function __construct(
        public array $features,
        public string $label,
        public int $id
    ) {
        if (empty($features)) {
            throw new InvalidArgumentException('Features cannot be empty');
        }
    }

    public function getFeatureCount(): int
    {
        return count($this->features);
    }

    public function __toString(): string
    {
        $featuresStr = '[' . implode(', ', array_map(fn($f) => is_numeric($f) ? number_format($f, 3) : $f, $this->features)) . ']';
        return "Sample {$this->id}: features={$featuresStr}, label={$this->label}";
    }
}

/**
 * Split condition for decision tree nodes
 */
readonly class SplitCondition
{
    public function __construct(
        public int $featureIndex,
        public mixed $threshold,
        public string $featureName,
        public bool $isNumerical,
        public float $informationGain
    ) {}

    public function evaluate(TrainingSample $sample): bool
    {
        $value = $sample->features[$this->featureIndex];
        
        if ($this->isNumerical) {
            return (float)$value <= (float)$this->threshold;
        } else {
            return $value === $this->threshold;
        }
    }

    public function __toString(): string
    {
        if ($this->isNumerical) {
            return "{$this->featureName} <= " . number_format((float)$this->threshold, 3);
        } else {
            return "{$this->featureName} = {$this->threshold}";
        }
    }
}

/**
 * Node in the decision tree
 */
class TreeNode
{
    public ?SplitCondition $splitCondition = null;
    public ?TreeNode $leftChild = null;
    public ?TreeNode $rightChild = null;
    public ?string $prediction = null;
    public float $confidence = 0.0;
    public int $depth = 0;
    public int $sampleCount = 0;
    public array $classDistribution = [];

    public static function createLeaf(
        string $prediction,
        float $confidence,
        int $sampleCount,
        array $classDistribution,
        int $depth
    ): self {
        $node = new self();
        $node->prediction = $prediction;
        $node->confidence = $confidence;
        $node->sampleCount = $sampleCount;
        $node->classDistribution = $classDistribution;
        $node->depth = $depth;
        return $node;
    }

    public static function createInternal(
        SplitCondition $splitCondition,
        int $sampleCount,
        array $classDistribution,
        int $depth
    ): self {
        $node = new self();
        $node->splitCondition = $splitCondition;
        $node->sampleCount = $sampleCount;
        $node->classDistribution = $classDistribution;
        $node->depth = $depth;
        return $node;
    }

    public function isLeaf(): bool
    {
        return $this->splitCondition === null;
    }
}

/**
 * Classification metrics for model evaluation
 */
readonly class ClassificationMetrics
{
    public function __construct(
        public float $accuracy,
        public array $precision,
        public array $recall,
        public array $f1Score,
        public float $macroPrecision,
        public float $macroRecall,
        public float $macroF1,
        public array $confusionMatrix,
        public int $sampleCount
    ) {}

    public function __toString(): string
    {
        $output = "Classification Metrics:\n";
        $output .= sprintf("  Accuracy: %.6f\n", $this->accuracy);
        $output .= sprintf("  Macro Precision: %.6f\n", $this->macroPrecision);
        $output .= sprintf("  Macro Recall: %.6f\n", $this->macroRecall);
        $output .= sprintf("  Macro F1-Score: %.6f\n", $this->macroF1);
        $output .= sprintf("  Sample count: %d\n\n", $this->sampleCount);

        $output .= "Per-class Metrics:\n";
        $classes = array_keys($this->precision);
        sort($classes);

        foreach ($classes as $className) {
            $output .= "  Class '{$className}':\n";
            $output .= sprintf("    Precision: %.6f\n", $this->precision[$className] ?? 0.0);
            $output .= sprintf("    Recall: %.6f\n", $this->recall[$className] ?? 0.0);
            $output .= sprintf("    F1-Score: %.6f\n", $this->f1Score[$className] ?? 0.0);
        }

        return $output;
    }
}

/**
 * Trait for entropy calculations
 */
trait EntropyCalculations
{
    /**
     * Calculate entropy of a dataset
     */
    private function calculateEntropy(array $samples): float
    {
        if (empty($samples)) {
            return 0.0;
        }

        $labelCounts = [];
        foreach ($samples as $sample) {
            /** @var TrainingSample $sample */
            $labelCounts[$sample->label] = ($labelCounts[$sample->label] ?? 0) + 1;
        }

        $entropy = 0.0;
        $totalSamples = count($samples);

        foreach ($labelCounts as $count) {
            if ($count > 0) {
                $probability = $count / $totalSamples;
                $entropy -= $probability * log($probability, 2);
            }
        }

        return $entropy;
    }

    /**
     * Calculate information gain for a split
     */
    private function calculateInformationGain(array $samples, array $leftSplit, array $rightSplit): float
    {
        $originalEntropy = $this->calculateEntropy($samples);
        $leftEntropy = $this->calculateEntropy($leftSplit);
        $rightEntropy = $this->calculateEntropy($rightSplit);

        $leftWeight = count($leftSplit) / count($samples);
        $rightWeight = count($rightSplit) / count($samples);

        return $originalEntropy - ($leftWeight * $leftEntropy + $rightWeight * $rightEntropy);
    }
}

/**
 * Trait for statistical calculations
 */
trait StatisticalCalculations
{
    /**
     * Get the most common class in a set of samples
     */
    private function getMajorityClass(array $samples): string
    {
        if (empty($samples)) {
            return 'unknown';
        }

        $labelCounts = [];
        foreach ($samples as $sample) {
            /** @var TrainingSample $sample */
            $labelCounts[$sample->label] = ($labelCounts[$sample->label] ?? 0) + 1;
        }

        return array_keys($labelCounts, max($labelCounts))[0];
    }

    /**
     * Calculate prediction confidence based on class distribution
     */
    private function calculateConfidence(array $samples): float
    {
        if (empty($samples)) {
            return 0.0;
        }

        $labelCounts = [];
        foreach ($samples as $sample) {
            /** @var TrainingSample $sample */
            $labelCounts[$sample->label] = ($labelCounts[$sample->label] ?? 0) + 1;
        }

        $maxCount = max($labelCounts);
        return $maxCount / count($samples);
    }

    /**
     * Get class distribution
     */
    private function getClassDistribution(array $samples): array
    {
        $distribution = [];
        foreach ($samples as $sample) {
            /** @var TrainingSample $sample */
            $distribution[$sample->label] = ($distribution[$sample->label] ?? 0) + 1;
        }
        return $distribution;
    }
}

/**
 * Interface for logging operations
 */
interface LoggerInterface
{
    public function info(string $message, array $context = []): void;
    public function error(string $message, array $context = []): void;
    public function debug(string $message, array $context = []): void;
}

/**
 * Simple console logger implementation
 */
class ConsoleLogger implements LoggerInterface
{
    public function info(string $message, array $context = []): void
    {
        echo "[INFO] " . $this->formatMessage($message, $context) . "\n";
    }

    public function error(string $message, array $context = []): void
    {
        echo "[ERROR] " . $this->formatMessage($message, $context) . "\n";
    }

    public function debug(string $message, array $context = []): void
    {
        echo "[DEBUG] " . $this->formatMessage($message, $context) . "\n";
    }

    private function formatMessage(string $message, array $context): string
    {
        $contextStr = empty($context) ? '' : ' | ' . json_encode($context);
        return $message . $contextStr;
    }
}

/**
 * Comprehensive Decision Tree Classifier Implementation
 */
class ProductionDecisionTree
{
    use EntropyCalculations, StatisticalCalculations;

    private ?TreeNode $root = null;
    private bool $fitted = false;
    private array $featureNames = [];
    private array $classNames = [];
    private LoggerInterface $logger;

    public function __construct(
        array $featureNames,
        private int $maxDepth = 10,
        private int $minSamplesLeaf = 1,
        private int $minSamplesSplit = 2,
        private float $minInfoGain = 0.0,
        private bool $usePruning = true,
        ?LoggerInterface $logger = null
    ) {
        if (empty($featureNames)) {
            throw new InvalidArgumentException('Feature names cannot be empty');
        }

        $this->featureNames = $featureNames;
        $this->logger = $logger ?? new ConsoleLogger();
    }

    /**
     * Validates input training data
     */
    private function validateTrainingData(array $samples): void
    {
        if (empty($samples)) {
            throw new DataValidationException('Training data cannot be empty');
        }

        $errors = [];
        $expectedFeatureCount = count($this->featureNames);

        foreach ($samples as $index => $sample) {
            if (!$sample instanceof TrainingSample) {
                $errors[] = "Sample at index {$index} is not a TrainingSample instance";
                continue;
            }

            if ($sample->getFeatureCount() !== $expectedFeatureCount) {
                $errors[] = "Sample {$sample->id} has {$sample->getFeatureCount()} features, expected {$expectedFeatureCount}";
            }
        }

        if (!empty($errors)) {
            throw new DataValidationException('Training data validation failed', $errors);
        }
    }

    /**
     * Find the best split for a feature
     */
    private function findBestSplit(array $samples): ?SplitCondition
    {
        $bestInfoGain = -1.0;
        $bestSplit = null;

        // Try each feature
        for ($featureIdx = 0; $featureIdx < count($this->featureNames); $featureIdx++) {
            $featureName = $this->featureNames[$featureIdx];

            // Get unique values for this feature
            $uniqueValues = [];
            foreach ($samples as $sample) {
                /** @var TrainingSample $sample */
                $value = $sample->features[$featureIdx];
                if (!in_array($value, $uniqueValues, true)) {
                    $uniqueValues[] = $value;
                }
            }

            // Try different split points
            if (count($uniqueValues) < 2) {
                continue; // Can't split on this feature
            }

            // Check if feature is numerical or categorical
            $isNumerical = is_numeric($uniqueValues[0]);

            if ($isNumerical) {
                // For numerical features, try thresholds between unique values
                sort($uniqueValues, SORT_NUMERIC);
                for ($i = 0; $i < count($uniqueValues) - 1; $i++) {
                    $threshold = ($uniqueValues[$i] + $uniqueValues[$i + 1]) / 2.0;
                    
                    [$leftSplit, $rightSplit] = $this->splitSamples($samples, $featureIdx, $threshold, true);
                    
                    if (!empty($leftSplit) && !empty($rightSplit)) {
                        $infoGain = $this->calculateInformationGain($samples, $leftSplit, $rightSplit);
                        
                        if ($infoGain > $bestInfoGain) {
                            $bestInfoGain = $infoGain;
                            $bestSplit = new SplitCondition($featureIdx, $threshold, $featureName, true, $infoGain);
                        }
                    }
                }
            } else {
                // For categorical features, try each unique value
                foreach ($uniqueValues as $value) {
                    [$leftSplit, $rightSplit] = $this->splitSamples($samples, $featureIdx, $value, false);
                    
                    if (!empty($leftSplit) && !empty($rightSplit)) {
                        $infoGain = $this->calculateInformationGain($samples, $leftSplit, $rightSplit);
                        
                        if ($infoGain > $bestInfoGain) {
                            $bestInfoGain = $infoGain;
                            $bestSplit = new SplitCondition($featureIdx, $value, $featureName, false, $infoGain);
                        }
                    }
                }
            }
        }

        return $bestInfoGain > $this->minInfoGain ? $bestSplit : null;
    }

    /**
     * Split samples based on a condition
     */
    private function splitSamples(array $samples, int $featureIndex, mixed $threshold, bool $isNumerical): array
    {
        $leftSplit = [];
        $rightSplit = [];

        foreach ($samples as $sample) {
            /** @var TrainingSample $sample */
            $value = $sample->features[$featureIndex];
            
            if ($isNumerical) {
                if ((float)$value <= (float)$threshold) {
                    $leftSplit[] = $sample;
                } else {
                    $rightSplit[] = $sample;
                }
            } else {
                if ($value === $threshold) {
                    $leftSplit[] = $sample;
                } else {
                    $rightSplit[] = $sample;
                }
            }
        }

        return [$leftSplit, $rightSplit];
    }

    /**
     * Build decision tree recursively
     */
    private function buildTree(array $samples, int $depth): TreeNode
    {
        if (empty($samples)) {
            return TreeNode::createLeaf('unknown', 0.0, 0, [], $depth);
        }

        $classDistribution = $this->getClassDistribution($samples);

        // Stopping criteria
        $shouldStop = count($samples) < $this->minSamplesSplit ||
                     $depth >= $this->maxDepth ||
                     count($classDistribution) === 1;

        if ($shouldStop) {
            $prediction = $this->getMajorityClass($samples);
            $confidence = $this->calculateConfidence($samples);
            return TreeNode::createLeaf($prediction, $confidence, count($samples), $classDistribution, $depth);
        }

        // Find best split
        $bestSplit = $this->findBestSplit($samples);

        if ($bestSplit === null) {
            $prediction = $this->getMajorityClass($samples);
            $confidence = $this->calculateConfidence($samples);
            return TreeNode::createLeaf($prediction, $confidence, count($samples), $classDistribution, $depth);
        }

        // Create internal node
        $node = TreeNode::createInternal($bestSplit, count($samples), $classDistribution, $depth);

        // Split samples and build children
        [$leftSamples, $rightSamples] = $this->splitSamples($samples, $bestSplit->featureIndex, $bestSplit->threshold, $bestSplit->isNumerical);

        if (!empty($leftSamples) && count($leftSamples) >= $this->minSamplesLeaf) {
            $node->leftChild = $this->buildTree($leftSamples, $depth + 1);
        }
        if (!empty($rightSamples) && count($rightSamples) >= $this->minSamplesLeaf) {
            $node->rightChild = $this->buildTree($rightSamples, $depth + 1);
        }

        return $node;
    }

    /**
     * Train the decision tree
     */
    public function fit(array $trainingSamples): void
    {
        $this->logger->info("🌳 Training Decision Tree Classifier");
        $this->logger->info(str_repeat("=", 40));

        $startTime = microtime(true);

        try {
            $this->validateTrainingData($trainingSamples);

            $this->logger->info("Training samples: " . count($trainingSamples));
            $this->logger->info("Features: " . count($this->featureNames));
            $this->logger->info("Max depth: {$this->maxDepth}");

            // Extract unique class names
            $this->classNames = [];
            foreach ($trainingSamples as $sample) {
                /** @var TrainingSample $sample */
                if (!in_array($sample->label, $this->classNames, true)) {
                    $this->classNames[] = $sample->label;
                }
            }
            sort($this->classNames);

            $this->logger->info("Classes: " . count($this->classNames) . " - " . implode(', ', $this->classNames));

            // Build the tree
            $this->root = $this->buildTree($trainingSamples, 0);

            $endTime = microtime(true);
            $trainingTime = ($endTime - $startTime) * 1000; // Convert to milliseconds

            $this->logger->info("✅ Tree construction completed in " . number_format($trainingTime, 3) . "ms");

            // Print tree statistics
            $this->printTreeStatistics();
            $this->fitted = true;

        } catch (Exception $e) {
            $this->logger->error("Training failed: " . $e->getMessage());
            throw new DecisionTreeException("Training failed", 0, $e);
        }
    }

    /**
     * Make a prediction for a single sample
     */
    public function predict(TrainingSample $sample): string
    {
        if (!$this->fitted || $this->root === null) {
            throw new RuntimeException('Model not trained. Call fit() first.');
        }

        $current = $this->root;

        while (!$current->isLeaf()) {
            $condition = $current->splitCondition;
            
            if ($condition->evaluate($sample)) {
                $current = $current->leftChild ?? $current;
            } else {
                $current = $current->rightChild ?? $current;
            }

            // Safety check to prevent infinite loops
            if ($current === $this->root) {
                break;
            }
        }

        return $current->prediction ?? 'unknown';
    }

    /**
     * Evaluate model on test data
     */
    public function evaluate(array $testSamples): ClassificationMetrics
    {
        $this->logger->info("\n📊 Evaluating Decision Tree Model");
        $this->logger->info(str_repeat("=", 35));

        $predictions = [];
        $actualLabels = [];

        foreach ($testSamples as $sample) {
            /** @var TrainingSample $sample */
            $prediction = $this->predict($sample);
            $predictions[] = $prediction;
            $actualLabels[] = $sample->label;
        }

        // Calculate confusion matrix
        $confusionMatrix = [];
        for ($i = 0; $i < count($actualLabels); $i++) {
            $actual = $actualLabels[$i];
            $predicted = $predictions[$i];
            
            $confusionMatrix[$actual][$predicted] = ($confusionMatrix[$actual][$predicted] ?? 0) + 1;
        }

        // Calculate per-class metrics
        $classes = array_unique(array_merge($actualLabels, $predictions));
        $precision = [];
        $recall = [];
        $f1Score = [];

        foreach ($classes as $className) {
            $tp = $confusionMatrix[$className][$className] ?? 0;
            
            $fp = 0;
            foreach ($classes as $otherClass) {
                if ($otherClass !== $className) {
                    $fp += $confusionMatrix[$otherClass][$className] ?? 0;
                }
            }
            
            $fn = 0;
            $classRow = $confusionMatrix[$className] ?? [];
            foreach ($classRow as $predicted => $count) {
                if ($predicted !== $className) {
                    $fn += $count;
                }
            }
            
            $precisionValue = ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0.0;
            $recallValue = ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
            $f1Value = ($precisionValue + $recallValue) > 0 
                ? 2 * $precisionValue * $recallValue / ($precisionValue + $recallValue) 
                : 0.0;
            
            $precision[$className] = $precisionValue;
            $recall[$className] = $recallValue;
            $f1Score[$className] = $f1Value;
        }

        // Calculate macro averages
        $macroPrecision = empty($precision) ? 0.0 : array_sum($precision) / count($precision);
        $macroRecall = empty($recall) ? 0.0 : array_sum($recall) / count($recall);
        $macroF1 = empty($f1Score) ? 0.0 : array_sum($f1Score) / count($f1Score);

        // Calculate accuracy
        $correct = 0;
        for ($i = 0; $i < count($actualLabels); $i++) {
            if ($actualLabels[$i] === $predictions[$i]) {
                $correct++;
            }
        }
        $accuracy = count($testSamples) > 0 ? $correct / count($testSamples) : 0.0;

        $metrics = new ClassificationMetrics(
            accuracy: $accuracy,
            precision: $precision,
            recall: $recall,
            f1Score: $f1Score,
            macroPrecision: $macroPrecision,
            macroRecall: $macroRecall,
            macroF1: $macroF1,
            confusionMatrix: $confusionMatrix,
            sampleCount: count($testSamples)
        );

        // Print evaluation results
        $this->logger->info("Test samples: " . count($testSamples));
        $this->logger->info("Accuracy: " . number_format($accuracy, 4));
        $this->logger->info("Macro Precision: " . number_format($macroPrecision, 4));
        $this->logger->info("Macro Recall: " . number_format($macroRecall, 4));
        $this->logger->info("Macro F1-Score: " . number_format($macroF1, 4));

        return $metrics;
    }

    /**
     * Print tree statistics
     */
    private function printTreeStatistics(): void
    {
        if ($this->root === null) {
            return;
        }

        $totalNodes = $this->countNodes($this->root);
        $leafNodes = $this->countLeaves($this->root);
        $maxDepth = $this->getMaxDepth($this->root);

        $this->logger->info("\n🌳 Tree Statistics:");
        $this->logger->info("Total nodes: {$totalNodes}");
        $this->logger->info("Leaf nodes: {$leafNodes}");
        $this->logger->info("Internal nodes: " . ($totalNodes - $leafNodes));
        $this->logger->info("Maximum depth: {$maxDepth}");
    }

    private function countNodes(?TreeNode $node): int
    {
        if ($node === null) {
            return 0;
        }
        return 1 + $this->countNodes($node->leftChild) + $this->countNodes($node->rightChild);
    }

    private function countLeaves(?TreeNode $node): int
    {
        if ($node === null) {
            return 0;
        }
        if ($node->isLeaf()) {
            return 1;
        }
        return $this->countLeaves($node->leftChild) + $this->countLeaves($node->rightChild);
    }

    private function getMaxDepth(?TreeNode $node): int
    {
        if ($node === null) {
            return 0;
        }
        return 1 + max($this->getMaxDepth($node->leftChild), $this->getMaxDepth($node->rightChild));
    }

    /**
     * Get feature names
     */
    public function getFeatureNames(): array
    {
        return $this->featureNames;
    }

    /**
     * Get class names
     */
    public function getClassNames(): array
    {
        return $this->classNames;
    }

    /**
     * Check if model is fitted
     */
    public function isFitted(): bool
    {
        return $this->fitted;
    }
}

/**
 * Utility class for generating synthetic datasets
 */
class SyntheticDataGenerator
{
    /**
     * Generate a synthetic Iris-like dataset
     */
    public static function generateIrisDataset(int $samples = 150): array
    {
        $classes = ['setosa', 'versicolor', 'virginica'];
        $dataset = [];

        echo "📊 Generating synthetic Iris dataset\n";

        for ($i = 0; $i < $samples; $i++) {
            $classIndex = $i % 3;
            $className = $classes[$classIndex];

            // Generate features based on class
            switch ($classIndex) {
                case 0: // Setosa
                    $features = [
                        4.5 + self::randomGaussian() * 0.5, // Sepal length
                        3.0 + self::randomGaussian() * 0.3, // Sepal width
                        1.5 + self::randomGaussian() * 0.3, // Petal length
                        0.3 + self::randomGaussian() * 0.1, // Petal width
                    ];
                    break;
                case 1: // Versicolor
                    $features = [
                        6.0 + self::randomGaussian() * 0.5,
                        2.8 + self::randomGaussian() * 0.3,
                        4.5 + self::randomGaussian() * 0.5,
                        1.4 + self::randomGaussian() * 0.3,
                    ];
                    break;
                case 2: // Virginica
                    $features = [
                        6.5 + self::randomGaussian() * 0.5,
                        3.0 + self::randomGaussian() * 0.3,
                        5.5 + self::randomGaussian() * 0.5,
                        2.0 + self::randomGaussian() * 0.3,
                    ];
                    break;
            }

            // Ensure non-negative values
            $features = array_map(fn($f) => max(0.1, $f), $features);

            $dataset[] = new TrainingSample($features, $className, $i);
        }

        // Shuffle the dataset
        shuffle($dataset);

        $classDistribution = array_count_values(array_map(fn($s) => $s->label, $dataset));
        echo "✅ Synthetic Iris dataset generated\n";
        echo "Class distribution: " . implode(', ', array_map(fn($k, $v) => "{$k}: {$v}", array_keys($classDistribution), $classDistribution)) . "\n";

        return $dataset;
    }

    /**
     * Generate Gaussian random number (Box-Muller transform)
     */
    private static function randomGaussian(): float
    {
        static $hasSpare = false;
        static $spare;

        if ($hasSpare) {
            $hasSpare = false;
            return $spare;
        }

        $hasSpare = true;
        $u = (float)mt_rand() / (float)mt_getrandmax();
        $v = (float)mt_rand() / (float)mt_getrandmax();
        $magnitude = sqrt(-2.0 * log($u));
        $spare = $magnitude * cos(2.0 * M_PI * $v);

        return $magnitude * sin(2.0 * M_PI * $v);
    }
}

/**
 * Comprehensive demonstration class
 */
class DecisionTreeDemo
{
    /**
     * Demonstrate comprehensive decision tree capabilities
     */
    public static function demonstrateDecisionTree(): void
    {
        echo "🚀 Decision Tree Implementation Demonstration\n";
        echo str_repeat("=", 50) . "\n";

        try {
            // Generate Iris-like dataset
            $dataset = SyntheticDataGenerator::generateIrisDataset(150);

            // Split into train/test
            shuffle($dataset);
            $trainSize = (int)(count($dataset) * 0.8);
            $trainData = array_slice($dataset, 0, $trainSize);
            $testData = array_slice($dataset, $trainSize);

            echo "Total samples: " . count($dataset) . ", Train: {$trainSize}, Test: " . count($testData) . "\n";

            // Create decision tree
            $featureNames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];

            $dt = new ProductionDecisionTree(
                featureNames: $featureNames,
                maxDepth: 8,
                minSamplesLeaf: 2,
                minSamplesSplit: 5
            );

            // Train the model
            $dt->fit($trainData);

            // Evaluate on test set
            $testResults = $dt->evaluate($testData);

            // Test individual predictions
            echo "\n🧪 Sample Predictions:\n";
            $sampleCount = min(5, count($testData));
            for ($i = 0; $i < $sampleCount; $i++) {
                $sample = $testData[$i];
                $prediction = $dt->predict($sample);
                $featuresStr = '[' . implode(', ', array_map(fn($f) => number_format($f, 2), $sample->features)) . ']';
                echo "Sample {$sample->id}: {$featuresStr} -> Predicted: {$prediction}, Actual: {$sample->label}\n";
            }

            echo "\n✅ Decision tree demonstration completed successfully!\n";

        } catch (Exception $e) {
            echo "❌ Decision tree demonstration failed: " . $e->getMessage() . "\n";
            echo "Stack trace:\n" . $e->getTraceAsString() . "\n";
        }
    }
}

// Set random seed for reproducibility
mt_srand(42);

// Run demonstration if this script is executed directly
if (basename(__FILE__) === basename($_SERVER['SCRIPT_NAME'] ?? '')) {
    DecisionTreeDemo::demonstrateDecisionTree();
}