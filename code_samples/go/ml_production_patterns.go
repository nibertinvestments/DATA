/*
Production-Ready Machine Learning Patterns in Go
===============================================

This module demonstrates industry-standard ML patterns in Go with proper
error handling, validation, concurrency, and production deployment
considerations for AI training datasets.

Key Features:
- Strong type safety with interfaces
- Concurrent processing with goroutines and channels
- Comprehensive error handling with custom error types
- Memory-efficient data processing
- Idiomatic Go patterns for ML operations
- Extensive documentation for AI learning
- Production-ready patterns with proper resource management

Author: AI Training Dataset
License: MIT
*/

package main

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"
)

// Custom error types for better error handling
type DataValidationError struct {
	Message string
	Field   string
}

func (e *DataValidationError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("data validation error in field '%s': %s", e.Field, e.Message)
	}
	return fmt.Sprintf("data validation error: %s", e.Message)
}

type ModelTrainingError struct {
	Message string
	Cause   error
}

func (e *ModelTrainingError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("model training error: %s (caused by: %v)", e.Message, e.Cause)
	}
	return fmt.Sprintf("model training error: %s", e.Message)
}

type PredictionError struct {
	Message string
	Index   int
}

func (e *PredictionError) Error() string {
	if e.Index >= 0 {
		return fmt.Sprintf("prediction error at index %d: %s", e.Index, e.Message)
	}
	return fmt.Sprintf("prediction error: %s", e.Message)
}

// DataPoint represents a single training example
type DataPoint struct {
	Features []float64 `json:"features"`
	Target   float64   `json:"target"`
	ID       string    `json:"id"`
}

// NewDataPoint creates a new DataPoint with validation
func NewDataPoint(features []float64, target float64, id string) (*DataPoint, error) {
	if len(features) == 0 {
		return nil, &DataValidationError{Message: "features cannot be empty"}
	}
	
	for i, f := range features {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, &DataValidationError{
				Message: fmt.Sprintf("invalid feature value at index %d: %f", i, f),
			}
		}
	}
	
	if math.IsNaN(target) || math.IsInf(target, 0) {
		return nil, &DataValidationError{Message: fmt.Sprintf("invalid target value: %f", target)}
	}
	
	if id == "" {
		id = generateID()
	}
	
	return &DataPoint{
		Features: make([]float64, len(features)),
		Target:   target,
		ID:       id,
	}, nil
}

// Copy creates a deep copy of the DataPoint
func (dp *DataPoint) Copy() *DataPoint {
	features := make([]float64, len(dp.Features))
	copy(features, dp.Features)
	return &DataPoint{
		Features: features,
		Target:   dp.Target,
		ID:       dp.ID,
	}
}

// ModelMetrics holds comprehensive model performance metrics
type ModelMetrics struct {
	Accuracy      float64       `json:"accuracy"`
	RMSE          float64       `json:"rmse"`
	MSE           float64       `json:"mse"`
	RSquared      float64       `json:"r_squared"`
	TrainingTime  time.Duration `json:"training_time"`
	PredictionTime time.Duration `json:"prediction_time"`
	ModelSize     int64         `json:"model_size_bytes"`
	Timestamp     time.Time     `json:"timestamp"`
}

// String returns a formatted string representation of metrics
func (m *ModelMetrics) String() string {
	return fmt.Sprintf("ModelMetrics{R²=%.4f, RMSE=%.4f, TrainingTime=%v, ModelSize=%d bytes}",
		m.RSquared, m.RMSE, m.TrainingTime, m.ModelSize)
}

// ValidationResult holds data validation results
type ValidationResult struct {
	IsValid       bool              `json:"is_valid"`
	Errors        []string          `json:"errors"`
	Warnings      []string          `json:"warnings"`
	RowCount      int               `json:"row_count"`
	FeatureCount  int               `json:"feature_count"`
	MissingValues map[string]int    `json:"missing_values"`
}

// MLModel interface defines the contract for ML models
type MLModel interface {
	Fit(ctx context.Context, X [][]float64, y []float64) error
	Predict(ctx context.Context, X [][]float64) ([]float64, error)
	Evaluate(ctx context.Context, X [][]float64, y []float64) (*ModelMetrics, error)
	SaveModel() ([]byte, error)
	LoadModel(data []byte) error
	IsTrained() bool
}

// DataValidator provides comprehensive data validation for ML pipelines
type DataValidator struct {
	MinRows          int
	MaxMissingRatio  float64
	RequiredColumns  []string
	SecurityChecks   bool
	mu               sync.RWMutex
}

// NewDataValidator creates a new DataValidator with default settings
func NewDataValidator() *DataValidator {
	return &DataValidator{
		MinRows:         10,
		MaxMissingRatio: 0.3,
		SecurityChecks:  true,
	}
}

// WithMinRows sets the minimum number of rows required
func (dv *DataValidator) WithMinRows(minRows int) *DataValidator {
	dv.mu.Lock()
	defer dv.mu.Unlock()
	dv.MinRows = minRows
	return dv
}

// WithMaxMissingRatio sets the maximum allowed missing value ratio
func (dv *DataValidator) WithMaxMissingRatio(ratio float64) *DataValidator {
	dv.mu.Lock()
	defer dv.mu.Unlock()
	dv.MaxMissingRatio = ratio
	return dv
}

// ValidateDataPoints validates a slice of DataPoints
func (dv *DataValidator) ValidateDataPoints(ctx context.Context, data []*DataPoint) (*ValidationResult, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	
	if data == nil {
		return nil, errors.New("data cannot be nil")
	}
	
	result := &ValidationResult{
		IsValid:       true,
		Errors:        []string{},
		Warnings:      []string{},
		RowCount:      len(data),
		MissingValues: make(map[string]int),
	}
	
	// Check minimum rows
	if len(data) < dv.MinRows {
		result.Errors = append(result.Errors,
			fmt.Sprintf("insufficient data: %d rows < %d", len(data), dv.MinRows))
		result.IsValid = false
	}
	
	// Check for empty dataset
	if len(data) == 0 {
		result.Errors = append(result.Errors, "dataset is empty")
		result.IsValid = false
		return result, nil
	}
	
	// Get expected feature count from first row
	expectedFeatures := len(data[0].Features)
	result.FeatureCount = expectedFeatures
	
	// Validate each data point
	duplicateCheck := make(map[string]int)
	
	for i, point := range data {
		// Check for context cancellation
		if i%100 == 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
		}
		
		if point == nil {
			result.Errors = append(result.Errors, fmt.Sprintf("nil data point at index %d", i))
			result.IsValid = false
			continue
		}
		
		// Check feature count consistency
		if len(point.Features) != expectedFeatures {
			result.Errors = append(result.Errors,
				fmt.Sprintf("inconsistent feature count at row %d: expected %d, got %d",
					i, expectedFeatures, len(point.Features)))
			result.IsValid = false
		}
		
		// Check for invalid values
		for j, feature := range point.Features {
			if math.IsNaN(feature) {
				result.Errors = append(result.Errors,
					fmt.Sprintf("NaN value found at row %d, feature %d", i, j))
				result.IsValid = false
			}
			if math.IsInf(feature, 0) {
				result.Errors = append(result.Errors,
					fmt.Sprintf("infinite value found at row %d, feature %d", i, j))
				result.IsValid = false
			}
		}
		
		// Check target value
		if math.IsNaN(point.Target) {
			result.Errors = append(result.Errors, fmt.Sprintf("NaN target at row %d", i))
			result.IsValid = false
		}
		if math.IsInf(point.Target, 0) {
			result.Errors = append(result.Errors, fmt.Sprintf("infinite target at row %d", i))
			result.IsValid = false
		}
		
		// Check for duplicates
		pointKey := generatePointKey(point)
		duplicateCheck[pointKey]++
	}
	
	// Report duplicates
	duplicateCount := 0
	for _, count := range duplicateCheck {
		if count > 1 {
			duplicateCount += count - 1
		}
	}
	
	if duplicateCount > 0 {
		ratio := float64(duplicateCount) / float64(len(data)) * 100
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("found %d duplicate rows (%.2f%%)", duplicateCount, ratio))
	}
	
	// Security validation if enabled
	if dv.SecurityChecks {
		dv.validateSecurity(data, result)
	}
	
	return result, nil
}

// validateSecurity performs basic security validation on string fields
func (dv *DataValidator) validateSecurity(data []*DataPoint, result *ValidationResult) {
	suspiciousPatterns := []string{
		"<script", "javascript:", "onload=", "onerror=",
		"'; DROP", "UNION SELECT", "../", "$(", "`",
	}
	
	for i, point := range data {
		if point == nil {
			continue
		}
		
		// Check ID field for suspicious patterns
		id := strings.ToLower(point.ID)
		for _, pattern := range suspiciousPatterns {
			if strings.Contains(id, strings.ToLower(pattern)) {
				result.Errors = append(result.Errors,
					fmt.Sprintf("suspicious pattern '%s' detected in ID at row %d", pattern, i))
				result.IsValid = false
			}
		}
	}
}

// SimpleLinearRegression implements a thread-safe linear regression model
type SimpleLinearRegression struct {
	weights     []float64
	bias        float64
	isTrained   bool
	trainingTime time.Duration
	mu          sync.RWMutex
	
	// Hyperparameters
	learningRate float64
	maxEpochs    int
	tolerance    float64
}

// NewSimpleLinearRegression creates a new linear regression model
func NewSimpleLinearRegression() *SimpleLinearRegression {
	return &SimpleLinearRegression{
		learningRate: 0.01,
		maxEpochs:    1000,
		tolerance:    1e-6,
	}
}

// WithHyperparameters sets model hyperparameters
func (slr *SimpleLinearRegression) WithHyperparameters(lr float64, epochs int, tolerance float64) *SimpleLinearRegression {
	slr.learningRate = lr
	slr.maxEpochs = epochs
	slr.tolerance = tolerance
	return slr
}

// Fit trains the linear regression model using gradient descent
func (slr *SimpleLinearRegression) Fit(ctx context.Context, X [][]float64, y []float64) error {
	slr.mu.Lock()
	defer slr.mu.Unlock()
	
	startTime := time.Now()
	defer func() {
		slr.trainingTime = time.Since(startTime)
	}()
	
	// Validate input
	if len(X) != len(y) {
		return &ModelTrainingError{
			Message: fmt.Sprintf("feature and target length mismatch: %d != %d", len(X), len(y)),
		}
	}
	
	if len(X) == 0 {
		return &ModelTrainingError{Message: "empty training data"}
	}
	
	numSamples := len(X)
	numFeatures := len(X[0])
	
	// Validate feature dimensions
	for i, row := range X {
		if len(row) != numFeatures {
			return &ModelTrainingError{
				Message: fmt.Sprintf("inconsistent feature count at row %d: expected %d, got %d",
					i, numFeatures, len(row)),
			}
		}
	}
	
	// Initialize weights and bias
	slr.weights = make([]float64, numFeatures)
	rand.Seed(42) // Fixed seed for reproducibility
	for i := range slr.weights {
		slr.weights[i] = rand.NormFloat64() * 0.01
	}
	slr.bias = 0.0
	
	// Gradient descent with early stopping
	var previousCost float64 = math.Inf(1)
	
	for epoch := 0; epoch < slr.maxEpochs; epoch++ {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		
		// Forward pass - calculate predictions and cost
		var cost float64
		predictions := make([]float64, numSamples)
		
		for i := 0; i < numSamples; i++ {
			prediction := slr.bias
			for j := 0; j < numFeatures; j++ {
				prediction += X[i][j] * slr.weights[j]
			}
			predictions[i] = prediction
			
			error := prediction - y[i]
			cost += error * error
		}
		cost /= (2 * float64(numSamples))
		
		// Check for convergence
		if math.Abs(previousCost-cost) < slr.tolerance {
			log.Printf("Model converged at epoch %d with cost %.6f", epoch, cost)
			break
		}
		previousCost = cost
		
		// Backward pass - calculate gradients
		weightGradients := make([]float64, numFeatures)
		var biasGradient float64
		
		for i := 0; i < numSamples; i++ {
			error := predictions[i] - y[i]
			biasGradient += error
			
			for j := 0; j < numFeatures; j++ {
				weightGradients[j] += error * X[i][j]
			}
		}
		
		// Update parameters
		for j := 0; j < numFeatures; j++ {
			slr.weights[j] -= (slr.learningRate * weightGradients[j]) / float64(numSamples)
		}
		slr.bias -= (slr.learningRate * biasGradient) / float64(numSamples)
	}
	
	slr.isTrained = true
	log.Printf("Model training completed in %v", slr.trainingTime)
	
	return nil
}

// Predict makes predictions using the trained model
func (slr *SimpleLinearRegression) Predict(ctx context.Context, X [][]float64) ([]float64, error) {
	slr.mu.RLock()
	defer slr.mu.RUnlock()
	
	if !slr.isTrained {
		return nil, &PredictionError{Message: "model must be trained before making predictions"}
	}
	
	if len(X) == 0 {
		return []float64{}, nil
	}
	
	predictions := make([]float64, len(X))
	
	// Use worker goroutines for concurrent prediction
	const maxWorkers = 4
	workers := maxWorkers
	if len(X) < maxWorkers {
		workers = len(X)
	}
	
	type job struct {
		index int
		row   []float64
	}
	
	type result struct {
		index      int
		prediction float64
		err        error
	}
	
	jobs := make(chan job, len(X))
	results := make(chan result, len(X))
	
	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				// Check context cancellation
				select {
				case <-ctx.Done():
					results <- result{index: j.index, err: ctx.Err()}
					return
				default:
				}
				
				// Validate input
				if len(j.row) != len(slr.weights) {
					results <- result{
						index: j.index,
						err: &PredictionError{
							Index:   j.index,
							Message: fmt.Sprintf("feature count mismatch: expected %d, got %d", len(slr.weights), len(j.row)),
						},
					}
					continue
				}
				
				// Calculate prediction
				prediction := slr.bias
				for k, feature := range j.row {
					if math.IsNaN(feature) || math.IsInf(feature, 0) {
						results <- result{
							index: j.index,
							err: &PredictionError{
								Index:   j.index,
								Message: fmt.Sprintf("invalid feature value at position %d: %f", k, feature),
							},
						}
						continue
					}
					prediction += feature * slr.weights[k]
				}
				
				results <- result{index: j.index, prediction: prediction}
			}
		}()
	}
	
	// Send jobs
	go func() {
		defer close(jobs)
		for i, row := range X {
			jobs <- job{index: i, row: row}
		}
	}()
	
	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(results)
	}()
	
	// Collect results
	for r := range results {
		if r.err != nil {
			return nil, r.err
		}
		predictions[r.index] = r.prediction
	}
	
	return predictions, nil
}

// Evaluate calculates comprehensive metrics for model performance
func (slr *SimpleLinearRegression) Evaluate(ctx context.Context, X [][]float64, y []float64) (*ModelMetrics, error) {
	if !slr.isTrained {
		return nil, &ModelTrainingError{Message: "model must be trained before evaluation"}
	}
	
	startTime := time.Now()
	predictions, err := slr.Predict(ctx, X)
	predictionTime := time.Since(startTime)
	
	if err != nil {
		return nil, fmt.Errorf("evaluation failed: %w", err)
	}
	
	// Calculate metrics
	var mse, sumSquaredTotal float64
	
	// Calculate mean of y
	var ySum float64
	for _, val := range y {
		ySum += val
	}
	yMean := ySum / float64(len(y))
	
	// Calculate MSE and total sum of squares
	for i := 0; i < len(y); i++ {
		error := predictions[i] - y[i]
		mse += error * error
		sumSquaredTotal += (y[i] - yMean) * (y[i] - yMean)
	}
	
	mse /= float64(len(y))
	rmse := math.Sqrt(mse)
	
	// Calculate R-squared
	var rSquared float64
	if sumSquaredTotal != 0 {
		rSquared = 1.0 - (mse*float64(len(y)))/sumSquaredTotal
	}
	
	// Estimate model size
	modelSize := int64((len(slr.weights)+1)*8 + 64) // 8 bytes per float64 + metadata
	
	metrics := &ModelMetrics{
		Accuracy:       rSquared,
		RMSE:          rmse,
		MSE:           mse,
		RSquared:      rSquared,
		TrainingTime:  slr.trainingTime,
		PredictionTime: predictionTime,
		ModelSize:     modelSize,
		Timestamp:     time.Now(),
	}
	
	log.Printf("Model evaluation completed. R²: %.4f, RMSE: %.4f", rSquared, rmse)
	
	return metrics, nil
}

// SaveModel serializes the model to bytes
func (slr *SimpleLinearRegression) SaveModel() ([]byte, error) {
	slr.mu.RLock()
	defer slr.mu.RUnlock()
	
	if !slr.isTrained {
		return nil, &ModelTrainingError{Message: "cannot save untrained model"}
	}
	
	modelData := map[string]interface{}{
		"type":          "SimpleLinearRegression",
		"weights":       slr.weights,
		"bias":          slr.bias,
		"learning_rate": slr.learningRate,
		"max_epochs":    slr.maxEpochs,
		"tolerance":     slr.tolerance,
		"training_time": slr.trainingTime.String(),
		"timestamp":     time.Now().Unix(),
	}
	
	return json.Marshal(modelData)
}

// LoadModel deserializes the model from bytes
func (slr *SimpleLinearRegression) LoadModel(data []byte) error {
	slr.mu.Lock()
	defer slr.mu.Unlock()
	
	var modelData map[string]interface{}
	if err := json.Unmarshal(data, &modelData); err != nil {
		return &ModelTrainingError{Message: "failed to unmarshal model data", Cause: err}
	}
	
	// Extract weights
	weightsInterface, ok := modelData["weights"].([]interface{})
	if !ok {
		return &ModelTrainingError{Message: "invalid weights format"}
	}
	
	slr.weights = make([]float64, len(weightsInterface))
	for i, w := range weightsInterface {
		if weight, ok := w.(float64); ok {
			slr.weights[i] = weight
		} else {
			return &ModelTrainingError{Message: fmt.Sprintf("invalid weight at index %d", i)}
		}
	}
	
	// Extract bias
	if bias, ok := modelData["bias"].(float64); ok {
		slr.bias = bias
	} else {
		return &ModelTrainingError{Message: "invalid bias format"}
	}
	
	// Extract hyperparameters
	if lr, ok := modelData["learning_rate"].(float64); ok {
		slr.learningRate = lr
	}
	if epochs, ok := modelData["max_epochs"].(float64); ok {
		slr.maxEpochs = int(epochs)
	}
	if tol, ok := modelData["tolerance"].(float64); ok {
		slr.tolerance = tol
	}
	
	slr.isTrained = true
	log.Println("Model loaded successfully")
	
	return nil
}

// IsTrained returns whether the model has been trained
func (slr *SimpleLinearRegression) IsTrained() bool {
	slr.mu.RLock()
	defer slr.mu.RUnlock()
	return slr.isTrained
}

// FeatureEngineer provides feature engineering capabilities
type FeatureEngineer struct {
	scalerMeans map[int]float64
	scalerStds  map[int]float64
	isFitted    bool
	mu          sync.RWMutex
}

// NewFeatureEngineer creates a new FeatureEngineer
func NewFeatureEngineer() *FeatureEngineer {
	return &FeatureEngineer{
		scalerMeans: make(map[int]float64),
		scalerStds:  make(map[int]float64),
	}
}

// CreatePolynomialFeatures creates polynomial features up to specified degree
func (fe *FeatureEngineer) CreatePolynomialFeatures(ctx context.Context, X [][]float64, degree int) ([][]float64, error) {
	if degree < 2 || degree > 5 {
		return nil, errors.New("polynomial degree must be between 2 and 5")
	}
	
	startTime := time.Now()
	log.Printf("Creating polynomial features of degree %d...", degree)
	
	if len(X) == 0 {
		return [][]float64{}, nil
	}
	
	numSamples := len(X)
	numOriginalFeatures := len(X[0])
	
	// Calculate number of polynomial features
	numPolyFeatures := numOriginalFeatures        // Original features
	numPolyFeatures += numOriginalFeatures        // Squared terms
	
	if degree >= 2 {
		// Interaction terms: C(n,2) = n*(n-1)/2
		numPolyFeatures += (numOriginalFeatures * (numOriginalFeatures - 1)) / 2
	}
	
	// Create polynomial features concurrently
	result := make([][]float64, numSamples)
	const maxWorkers = 4
	workers := maxWorkers
	if numSamples < maxWorkers {
		workers = numSamples
	}
	
	type job struct {
		index int
		row   []float64
	}
	
	type processResult struct {
		index    int
		features []float64
		err      error
	}
	
	jobs := make(chan job, numSamples)
	results := make(chan processResult, numSamples)
	
	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				// Check context cancellation
				select {
				case <-ctx.Done():
					results <- processResult{index: j.index, err: ctx.Err()}
					return
				default:
				}
				
				polyFeatures := make([]float64, 0, numPolyFeatures)
				
				// Original features
				polyFeatures = append(polyFeatures, j.row...)
				
				// Squared terms
				for _, feature := range j.row {
					polyFeatures = append(polyFeatures, feature*feature)
				}
				
				// Interaction terms
				if degree >= 2 {
					for i := 0; i < len(j.row); i++ {
						for k := i + 1; k < len(j.row); k++ {
							polyFeatures = append(polyFeatures, j.row[i]*j.row[k])
						}
					}
				}
				
				results <- processResult{index: j.index, features: polyFeatures}
			}
		}()
	}
	
	// Send jobs
	go func() {
		defer close(jobs)
		for i, row := range X {
			jobs <- job{index: i, row: row}
		}
	}()
	
	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(results)
	}()
	
	// Collect results
	for r := range results {
		if r.err != nil {
			return nil, r.err
		}
		result[r.index] = r.features
	}
	
	duration := time.Since(startTime)
	log.Printf("Polynomial features created in %v: %d -> %d features", 
		duration, numOriginalFeatures, numPolyFeatures)
	
	return result, nil
}

// StandardScaler applies standard scaling (z-score normalization)
func (fe *FeatureEngineer) StandardScaler(ctx context.Context, X [][]float64, fit bool) ([][]float64, error) {
	fe.mu.Lock()
	defer fe.mu.Unlock()
	
	if len(X) == 0 || len(X[0]) == 0 {
		return nil, errors.New("cannot scale empty data")
	}
	
	startTime := time.Now()
	numSamples := len(X)
	numFeatures := len(X[0])
	
	if fit {
		// Calculate means and standard deviations
		fe.scalerMeans = make(map[int]float64)
		fe.scalerStds = make(map[int]float64)
		
		for j := 0; j < numFeatures; j++ {
			// Check context cancellation
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
			
			// Calculate mean
			var sum float64
			for i := 0; i < numSamples; i++ {
				sum += X[i][j]
			}
			mean := sum / float64(numSamples)
			fe.scalerMeans[j] = mean
			
			// Calculate standard deviation
			var sumSquaredDiffs float64
			for i := 0; i < numSamples; i++ {
				diff := X[i][j] - mean
				sumSquaredDiffs += diff * diff
			}
			std := math.Sqrt(sumSquaredDiffs / float64(numSamples))
			fe.scalerStds[j] = std
		}
		
		fe.isFitted = true
	}
	
	if !fe.isFitted {
		return nil, errors.New("scaler must be fitted before transform")
	}
	
	// Apply scaling
	result := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		result[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			mean := fe.scalerMeans[j]
			std := fe.scalerStds[j]
			
			if std == 0 {
				result[i][j] = 0 // Constant feature
			} else {
				result[i][j] = (X[i][j] - mean) / std
			}
		}
	}
	
	duration := time.Since(startTime)
	log.Printf("Standard scaling applied in %v to %d features", duration, numFeatures)
	
	return result, nil
}

// OutlierDetectionResult holds outlier detection results
type OutlierDetectionResult struct {
	OutlierIndices []int                    `json:"outlier_indices"`
	OutlierInfo    []map[string]interface{} `json:"outlier_info"`
}

// DetectOutliers detects outliers using specified method
func (fe *FeatureEngineer) DetectOutliers(ctx context.Context, X [][]float64, method string, threshold float64) (*OutlierDetectionResult, error) {
	startTime := time.Now()
	log.Printf("Detecting outliers using %s method...", method)
	
	if len(X) == 0 {
		return &OutlierDetectionResult{}, nil
	}
	
	outlierIndicesSet := make(map[int]bool)
	outlierInfo := []map[string]interface{}{}
	numFeatures := len(X[0])
	
	for j := 0; j < numFeatures; j++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		// Extract column
		column := make([]float64, len(X))
		for i := 0; i < len(X); i++ {
			column[i] = X[i][j]
		}
		
		var featureOutliers []int
		
		switch method {
		case "iqr":
			featureOutliers = fe.detectOutliersIQR(column, threshold)
		case "zscore":
			featureOutliers = fe.detectOutliersZScore(column, threshold)
		default:
			return nil, fmt.Errorf("unsupported outlier detection method: %s", method)
		}
		
		// Add to global outlier set
		for _, idx := range featureOutliers {
			outlierIndicesSet[idx] = true
		}
		
		// Create feature info if outliers found
		if len(featureOutliers) > 0 {
			info := map[string]interface{}{
				"feature":            j,
				"outlier_count":      len(featureOutliers),
				"outlier_percentage": float64(len(featureOutliers)) / float64(len(column)) * 100,
				"method":             method,
				"threshold":          threshold,
			}
			outlierInfo = append(outlierInfo, info)
		}
	}
	
	// Convert set to slice
	outlierIndices := make([]int, 0, len(outlierIndicesSet))
	for idx := range outlierIndicesSet {
		outlierIndices = append(outlierIndices, idx)
	}
	sort.Ints(outlierIndices)
	
	duration := time.Since(startTime)
	log.Printf("Outlier detection completed in %v. Found %d outlier rows", 
		duration, len(outlierIndices))
	
	return &OutlierDetectionResult{
		OutlierIndices: outlierIndices,
		OutlierInfo:    outlierInfo,
	}, nil
}

func (fe *FeatureEngineer) detectOutliersIQR(column []float64, threshold float64) []int {
	// Sort column for quartile calculation
	sortedColumn := make([]float64, len(column))
	copy(sortedColumn, column)
	sort.Float64s(sortedColumn)
	
	n := len(sortedColumn)
	q1 := sortedColumn[n/4]
	q3 := sortedColumn[3*n/4]
	iqr := q3 - q1
	
	lowerBound := q1 - threshold*iqr
	upperBound := q3 + threshold*iqr
	
	var outliers []int
	for i, value := range column {
		if value < lowerBound || value > upperBound {
			outliers = append(outliers, i)
		}
	}
	
	return outliers
}

func (fe *FeatureEngineer) detectOutliersZScore(column []float64, threshold float64) []int {
	// Calculate mean
	var sum float64
	for _, value := range column {
		sum += value
	}
	mean := sum / float64(len(column))
	
	// Calculate standard deviation
	var sumSquaredDiffs float64
	for _, value := range column {
		diff := value - mean
		sumSquaredDiffs += diff * diff
	}
	std := math.Sqrt(sumSquaredDiffs / float64(len(column)))
	
	var outliers []int
	for i, value := range column {
		zscore := math.Abs((value - mean) / std)
		if zscore > threshold {
			outliers = append(outliers, i)
		}
	}
	
	return outliers
}

// MLUtilities provides utility functions for ML operations
type MLUtilities struct{}

// GenerateSyntheticData creates synthetic regression data
func (MLUtilities) GenerateSyntheticData(numSamples, numFeatures int, noiseLevel float64) ([][]float64, []float64, error) {
	log.Printf("Generating synthetic dataset: %d samples, %d features", numSamples, numFeatures)
	
	if numSamples <= 0 || numFeatures <= 0 {
		return nil, nil, errors.New("number of samples and features must be positive")
	}
	
	rand.Seed(42) // Fixed seed for reproducibility
	
	// Generate true weights for linear relationship
	trueWeights := make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		trueWeights[i] = rand.NormFloat64()
	}
	trueBias := rand.NormFloat64()
	
	// Generate data
	X := make([][]float64, numSamples)
	y := make([]float64, numSamples)
	
	for i := 0; i < numSamples; i++ {
		// Generate features
		X[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			X[i][j] = rand.NormFloat64() * 2
		}
		
		// Generate target with linear relationship + noise
		target := trueBias
		for j := 0; j < numFeatures; j++ {
			target += X[i][j] * trueWeights[j]
		}
		target += rand.NormFloat64() * noiseLevel
		y[i] = target
	}
	
	return X, y, nil
}

// TrainTestSplit splits data into training and testing sets
func (MLUtilities) TrainTestSplit(X [][]float64, y []float64, testSize float64, randomSeed int64) ([][]float64, [][]float64, []float64, []float64, error) {
	if len(X) != len(y) {
		return nil, nil, nil, nil, errors.New("X and y must have the same length")
	}
	
	if testSize <= 0 || testSize >= 1 {
		return nil, nil, nil, nil, errors.New("testSize must be between 0 and 1")
	}
	
	rand.Seed(randomSeed)
	
	// Create indices and shuffle
	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}
	
	// Fisher-Yates shuffle
	for i := len(indices) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}
	
	testLength := int(float64(len(X)) * testSize)
	trainLength := len(X) - testLength
	
	// Split data
	XTrain := make([][]float64, trainLength)
	yTrain := make([]float64, trainLength)
	XTest := make([][]float64, testLength)
	yTest := make([]float64, testLength)
	
	for i := 0; i < trainLength; i++ {
		idx := indices[i]
		XTrain[i] = make([]float64, len(X[idx]))
		copy(XTrain[i], X[idx])
		yTrain[i] = y[idx]
	}
	
	for i := 0; i < testLength; i++ {
		idx := indices[trainLength+i]
		XTest[i] = make([]float64, len(X[idx]))
		copy(XTest[i], X[idx])
		yTest[i] = y[idx]
	}
	
	return XTrain, XTest, yTrain, yTest, nil
}

// Helper functions

// generateID creates a simple unique ID
func generateID() string {
	return fmt.Sprintf("dp_%d_%d", time.Now().UnixNano(), rand.Int63())
}

// generatePointKey creates a key for duplicate detection
func generatePointKey(point *DataPoint) string {
	h := md5.New()
	for _, f := range point.Features {
		fmt.Fprintf(h, "%.6f", f)
	}
	fmt.Fprintf(h, "%.6f", point.Target)
	return fmt.Sprintf("%x", h.Sum(nil))
}

// DemonstrateGoMLPatterns shows a comprehensive example of Go ML patterns
func DemonstrateGoMLPatterns() {
	fmt.Println("🚀 Go ML Pipeline Demonstration")
	fmt.Println(strings.Repeat("=", 40))
	
	ctx := context.Background()
	
	// Generate synthetic data
	utils := MLUtilities{}
	X, y, err := utils.GenerateSyntheticData(1000, 5, 0.1)
	if err != nil {
		log.Fatalf("❌ Failed to generate data: %v", err)
	}
	fmt.Printf("✅ Generated dataset: %d samples, %d features\n", len(X), len(X[0]))
	
	// Split data
	XTrain, XTest, yTrain, yTest, err := utils.TrainTestSplit(X, y, 0.2, 42)
	if err != nil {
		log.Fatalf("❌ Failed to split data: %v", err)
	}
	fmt.Printf("✅ Data split: %d training, %d test samples\n", len(XTrain), len(XTest))
	
	// Validate data
	fmt.Println("\n🔄 Validating data...")
	validator := NewDataValidator().WithMinRows(100).WithMaxMissingRatio(0.1)
	
	// Convert to DataPoints for validation (simplified for demo)
	dataPoints := make([]*DataPoint, len(XTrain))
	for i := 0; i < len(XTrain); i++ {
		dp, err := NewDataPoint(XTrain[i], yTrain[i], "")
		if err != nil {
			log.Fatalf("❌ Failed to create data point: %v", err)
		}
		dataPoints[i] = dp
	}
	
	validation, err := validator.ValidateDataPoints(ctx, dataPoints)
	if err != nil {
		log.Fatalf("❌ Validation failed: %v", err)
	}
	
	if !validation.IsValid {
		fmt.Printf("❌ Data validation failed: %v\n", validation.Errors)
		return
	}
	fmt.Println("✅ Data validation passed")
	
	// Train model
	fmt.Println("\n🔄 Training model...")
	model := NewSimpleLinearRegression().WithHyperparameters(0.01, 1000, 1e-6)
	
	if err := model.Fit(ctx, XTrain, yTrain); err != nil {
		log.Fatalf("❌ Training failed: %v", err)
	}
	fmt.Println("✅ Model training completed")
	
	// Make predictions
	fmt.Println("\n🔮 Making predictions...")
	XSample := XTest[:min(10, len(XTest))]
	predictions, err := model.Predict(ctx, XSample)
	if err != nil {
		log.Fatalf("❌ Prediction failed: %v", err)
	}
	
	fmt.Printf("Sample predictions: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
		predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])
	
	// Evaluate model
	fmt.Println("\n📊 Evaluating model...")
	metrics, err := model.Evaluate(ctx, XTest, yTest)
	if err != nil {
		log.Fatalf("❌ Evaluation failed: %v", err)
	}
	
	fmt.Printf("R²: %.4f\n", metrics.RSquared)
	fmt.Printf("RMSE: %.4f\n", metrics.RMSE)
	fmt.Printf("Training time: %v\n", metrics.TrainingTime)
	fmt.Printf("Model size: %d bytes\n", metrics.ModelSize)
	
	// Feature engineering demonstration
	fmt.Println("\n🔧 Feature Engineering...")
	engineer := NewFeatureEngineer()
	
	// Create polynomial features
	XTrainSample := XTrain[:min(100, len(XTrain))]
	polyFeatures, err := engineer.CreatePolynomialFeatures(ctx, XTrainSample, 2)
	if err != nil {
		log.Fatalf("❌ Polynomial features failed: %v", err)
	}
	fmt.Printf("Features after polynomial engineering: %d -> %d\n", 
		len(XTrainSample[0]), len(polyFeatures[0]))
	
	// Standard scaling
	scaledFeatures, err := engineer.StandardScaler(ctx, XTrainSample, true)
	if err != nil {
		log.Fatalf("❌ Standard scaling failed: %v", err)
	}
	fmt.Printf("Standard scaling applied to %d features\n", len(scaledFeatures[0]))
	
	// Outlier detection
	outlierResults, err := engineer.DetectOutliers(ctx, XTrain, "iqr", 1.5)
	if err != nil {
		log.Fatalf("❌ Outlier detection failed: %v", err)
	}
	fmt.Printf("Outlier detection: %d outlier rows found\n", len(outlierResults.OutlierIndices))
	
	// Model persistence
	fmt.Println("\n💾 Testing model persistence...")
	modelData, err := model.SaveModel()
	if err != nil {
		log.Fatalf("❌ Save model failed: %v", err)
	}
	
	newModel := NewSimpleLinearRegression()
	if err := newModel.LoadModel(modelData); err != nil {
		log.Fatalf("❌ Load model failed: %v", err)
	}
	fmt.Println("✅ Model save/load completed")
	
	fmt.Println("\n✅ Go ML Pipeline demonstration completed successfully!")
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// main function to run the demonstration
func main() {
	DemonstrateGoMLPatterns()
}