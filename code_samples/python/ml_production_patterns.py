"""
Production-Ready Machine Learning Patterns for AI Training Dataset
==================================================================

This module demonstrates industry-standard ML patterns with proper error handling,
validation, logging, and production deployment considerations.

Key Features:
- Type hints throughout for better code clarity
- Comprehensive error handling and validation
- Proper logging and monitoring hooks
- Memory-efficient implementations
- Thread-safe operations where applicable
- Extensive documentation for AI learning

Author: AI Training Dataset
License: MIT
"""

import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, 
    Protocol, TypeVar, Generic, Callable
)
import pickle
import json
import time
from functools import wraps, lru_cache
from threading import Lock
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions for better code clarity
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=BaseEstimator)


@dataclass
class ModelMetrics:
    """Data class for storing model performance metrics."""
    accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class DataValidationResult:
    """Results from data validation checks."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)


class MLModelProtocol(Protocol):
    """Protocol defining the interface for ML models."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLModelProtocol':
        """Train the model."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model score."""
        ...


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    return wrapper


def validate_input_data(func: Callable) -> Callable:
    """Decorator to validate input data for ML functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Basic validation for common ML function patterns
        if args and hasattr(args[0], '__class__'):
            # Skip self parameter for methods
            data_args = args[1:] if hasattr(args[0], '__dict__') else args
        else:
            data_args = args
        
        for i, arg in enumerate(data_args):
            if isinstance(arg, (np.ndarray, pd.DataFrame)):
                if isinstance(arg, np.ndarray) and arg.size == 0:
                    raise DataValidationError(f"Empty array at position {i}")
                elif isinstance(arg, pd.DataFrame) and arg.empty:
                    raise DataValidationError(f"Empty DataFrame at position {i}")
        
        return func(*args, **kwargs)
    return wrapper


class DataValidator:
    """
    Comprehensive data validation for ML pipelines.
    
    Validates data quality, structure, and suitability for ML training.
    """
    
    def __init__(self, 
                 min_rows: int = 10, 
                 max_missing_ratio: float = 0.3,
                 required_columns: Optional[List[str]] = None):
        """
        Initialize the data validator.
        
        Args:
            min_rows: Minimum number of rows required
            max_missing_ratio: Maximum ratio of missing values per column
            required_columns: List of required column names
        """
        self.min_rows = min_rows
        self.max_missing_ratio = max_missing_ratio
        self.required_columns = required_columns or []
        
    def validate_dataframe(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Comprehensive validation of a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataValidationResult with validation details
            
        Raises:
            DataValidationError: If critical validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        result = DataValidationResult(
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )
        
        # Check minimum rows
        if len(df) < self.min_rows:
            result.errors.append(f"Insufficient data: {len(df)} rows < {self.min_rows}")
            result.is_valid = False
            
        # Check for empty DataFrame
        if df.empty:
            result.errors.append("DataFrame is empty")
            result.is_valid = False
            return result
            
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            result.errors.append(f"Missing required columns: {list(missing_cols)}")
            result.is_valid = False
            
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)
            result.missing_values[col] = missing_count
            
            if missing_ratio > self.max_missing_ratio:
                result.errors.append(
                    f"Column '{col}' has {missing_ratio:.2%} missing values "
                    f"(max allowed: {self.max_missing_ratio:.2%})"
                )
                result.is_valid = False
                
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            result.warnings.append(f"Found {duplicate_count} duplicate rows")
            
        # Check data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                unique_types = set(type(x).__name__ for x in df[col].dropna().iloc[:100])
                if len(unique_types) > 1:
                    result.warnings.append(
                        f"Column '{col}' has mixed data types: {unique_types}"
                    )
                    
        return result
        
    def validate_features_target(self, 
                                X: Union[np.ndarray, pd.DataFrame], 
                                y: Union[np.ndarray, pd.Series]) -> DataValidationResult:
        """
        Validate features and target arrays/DataFrames for ML training.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            DataValidationResult with validation details
        """
        result = DataValidationResult(is_valid=True)
        
        # Convert to numpy arrays for consistent handling
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            result.column_count = X.shape[1]
        else:
            X_array = np.asarray(X)
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.asarray(y)
            
        result.row_count = len(X_array)
        
        # Check shapes
        if len(X_array) != len(y_array):
            result.errors.append(
                f"Feature and target length mismatch: {len(X_array)} != {len(y_array)}"
            )
            result.is_valid = False
            
        # Check for NaN values
        if np.isnan(X_array).any():
            nan_count = np.isnan(X_array).sum()
            result.errors.append(f"Features contain {nan_count} NaN values")
            result.is_valid = False
            
        if np.isnan(y_array).any():
            nan_count = np.isnan(y_array).sum()
            result.errors.append(f"Target contains {nan_count} NaN values")
            result.is_valid = False
            
        # Check for infinite values
        if np.isinf(X_array).any():
            inf_count = np.isinf(X_array).sum()
            result.errors.append(f"Features contain {inf_count} infinite values")
            result.is_valid = False
            
        return result


class ProductionMLPipeline:
    """
    Production-ready ML pipeline with comprehensive error handling,
    monitoring, and validation.
    
    Features:
    - Thread-safe operations
    - Model versioning and persistence
    - Comprehensive logging
    - Data validation at each step
    - Performance monitoring
    - Graceful error handling
    """
    
    def __init__(self, 
                 model: Optional[BaseEstimator] = None,
                 validator: Optional[DataValidator] = None,
                 enable_monitoring: bool = True):
        """
        Initialize the production ML pipeline.
        
        Args:
            model: ML model instance (defaults to RandomForest)
            validator: Data validator instance
            enable_monitoring: Enable performance monitoring
        """
        self.model = model or RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.validator = validator or DataValidator()
        self.enable_monitoring = enable_monitoring
        self._lock = Lock()  # Thread safety
        self._is_trained = False
        self._feature_names: Optional[List[str]] = None
        self._model_hash: Optional[str] = None
        
    @timing_decorator
    @validate_input_data
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            validation_split: float = 0.2) -> 'ProductionMLPipeline':
        """
        Train the ML model with comprehensive validation and monitoring.
        
        Args:
            X: Feature data
            y: Target data  
            validation_split: Fraction of data for validation
            
        Returns:
            Self for method chaining
            
        Raises:
            ModelTrainingError: If training fails
            DataValidationError: If data validation fails
        """
        with self._lock:  # Thread safety
            try:
                logger.info("Starting model training")
                
                # Validate input data
                validation_result = self.validator.validate_features_target(X, y)
                if not validation_result.is_valid:
                    raise DataValidationError(
                        f"Data validation failed: {validation_result.errors}"
                    )
                
                # Store feature names if DataFrame
                if isinstance(X, pd.DataFrame):
                    self._feature_names = list(X.columns)
                    X_array = X.values
                else:
                    X_array = np.asarray(X)
                    
                y_array = np.asarray(y)
                
                # Train-validation split
                if validation_split > 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_array, y_array, 
                        test_size=validation_split,
                        random_state=42,
                        stratify=y_array if len(np.unique(y_array)) > 1 else None
                    )
                else:
                    X_train, y_train = X_array, y_array
                
                # Train the model
                start_time = time.time()
                self.model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Validate on held-out data
                if validation_split > 0:
                    val_accuracy = self.model.score(X_val, y_val)
                    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
                
                # Generate model hash for versioning
                self._model_hash = self._generate_model_hash()
                
                self._is_trained = True
                logger.info(f"Model training completed in {training_time:.4f} seconds")
                
                return self
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}")
                raise ModelTrainingError(f"Training failed: {str(e)}") from e
    
    @timing_decorator  
    @validate_input_data
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with comprehensive error handling.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Prediction array
            
        Raises:
            ModelTrainingError: If model is not trained
            DataValidationError: If input data is invalid
        """
        if not self._is_trained:
            raise ModelTrainingError("Model must be trained before making predictions")
            
        try:
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                # Validate feature names match training
                if self._feature_names and list(X.columns) != self._feature_names:
                    logger.warning("Feature names don't match training data")
                X_array = X.values
            else:
                X_array = np.asarray(X)
            
            # Basic validation
            if np.isnan(X_array).any() or np.isinf(X_array).any():
                raise DataValidationError("Input contains NaN or infinite values")
            
            # Make predictions
            predictions = self.model.predict(X_array)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    @timing_decorator
    def evaluate(self, 
                X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> ModelMetrics:
        """
        Comprehensive model evaluation with detailed metrics.
        
        Args:
            X: Feature data
            y: True target values
            
        Returns:
            ModelMetrics with comprehensive evaluation
        """
        if not self._is_trained:
            raise ModelTrainingError("Model must be trained before evaluation")
        
        try:
            # Make predictions
            start_time = time.time()
            y_pred = self.predict(X)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)
            
            # Create metrics object
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=report['weighted avg']['precision'],
                recall=report['weighted avg']['recall'],
                f1_score=report['weighted avg']['f1-score'],
                classification_report=classification_report(y, y_pred),
                prediction_time=prediction_time,
                model_size_mb=self._estimate_model_size()
            )
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model with metadata.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ModelTrainingError: If model is not trained
        """
        if not self._is_trained:
            raise ModelTrainingError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self._feature_names,
            'model_hash': self._model_hash,
            'is_trained': self._is_trained,
            'save_timestamp': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'ProductionMLPipeline':
        """
        Load a saved model with validation.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self._feature_names = model_data.get('feature_names')
            self._model_hash = model_data.get('model_hash')
            self._is_trained = model_data.get('is_trained', True)
            
            logger.info(f"Model loaded from {filepath}")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _generate_model_hash(self) -> str:
        """Generate a hash for model versioning."""
        model_str = str(self.model.get_params())
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB."""
        try:
            import sys
            return sys.getsizeof(pickle.dumps(self.model)) / (1024 * 1024)
        except Exception:
            return 0.0


class FeatureEngineer:
    """
    Production-ready feature engineering with validation and monitoring.
    
    Provides common feature engineering operations with proper error handling
    and data validation for ML pipelines.
    """
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self._fitted_columns: set = set()
    
    @timing_decorator
    def create_polynomial_features(self, 
                                 df: pd.DataFrame, 
                                 columns: List[str], 
                                 degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features with validation.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if degree < 2 or degree > 5:
            raise ValueError("Polynomial degree must be between 2 and 5")
            
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        result_df = df.copy()
        
        # Create polynomial combinations
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i:], i):
                if i == j:
                    # Square terms
                    feature_name = f"{col1}_squared"
                    result_df[feature_name] = df[col1] ** 2
                else:
                    # Interaction terms
                    feature_name = f"{col1}_{col2}_interaction"
                    result_df[feature_name] = df[col1] * df[col2]
        
        logger.info(f"Created polynomial features for {len(columns)} columns")
        return result_df
    
    @timing_decorator
    def handle_missing_values(self, 
                            df: pd.DataFrame, 
                            strategy: str = 'mean',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values with multiple strategies.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            columns: Specific columns to process (default: all numeric)
            
        Returns:
            DataFrame with missing values handled
        """
        result_df = df.copy()
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            if strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                fill_value = df[col].mean()
                result_df[col] = result_df[col].fillna(fill_value)
            elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                fill_value = df[col].median()
                result_df[col] = result_df[col].fillna(fill_value)
            elif strategy == 'mode':
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                result_df[col] = result_df[col].fillna(fill_value)
            elif strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
            
            logger.info(f"Handled {missing_count} missing values in '{col}' using {strategy}")
        
        return result_df
    
    @timing_decorator
    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using multiple methods.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier information
        """
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
            
            outlier_count = len(outliers)
            if outlier_count > 0:
                outlier_info.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_count / len(df) * 100,
                    'method': method,
                    'threshold': threshold
                })
        
        return pd.DataFrame(outlier_info)


# Example usage and demonstration
def demonstrate_production_ml_pipeline():
    """
    Demonstrate the production ML pipeline with comprehensive examples.
    """
    print("🚀 Production ML Pipeline Demonstration")
    print("=" * 50)
    
    try:
        # Generate sample data
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Convert to DataFrame for better handling
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        print(f"✅ Generated dataset: {df.shape}")
        
        # Initialize pipeline with validation
        validator = DataValidator(min_rows=100, max_missing_ratio=0.1)
        pipeline = ProductionMLPipeline(validator=validator)
        
        # Train model
        print("\n🔄 Training model...")
        pipeline.fit(df[feature_names], df['target'], validation_split=0.2)
        
        # Make predictions
        print("\n🔮 Making predictions...")
        predictions = pipeline.predict(df[feature_names].head(10))
        print(f"Sample predictions: {predictions[:5]}")
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        metrics = pipeline.evaluate(df[feature_names], df['target'])
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"F1-Score: {metrics.f1_score:.4f}")
        
        # Feature engineering demonstration
        print("\n🔧 Feature Engineering...")
        engineer = FeatureEngineer()
        
        # Create polynomial features
        poly_df = engineer.create_polynomial_features(
            df, ['feature_0', 'feature_1'], degree=2
        )
        print(f"Features after polynomial engineering: {poly_df.shape[1]}")
        
        # Handle missing values (simulate some missing data)
        df_with_missing = df.copy()
        df_with_missing.loc[0:10, 'feature_0'] = np.nan
        
        cleaned_df = engineer.handle_missing_values(
            df_with_missing, strategy='mean'
        )
        print(f"Missing values handled: {df_with_missing['feature_0'].isnull().sum()} -> {cleaned_df['feature_0'].isnull().sum()}")
        
        # Outlier detection
        outlier_info = engineer.detect_outliers(df, method='iqr')
        print(f"Outlier detection completed for {len(outlier_info)} columns")
        
        print("\n✅ Production ML Pipeline demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    demonstrate_production_ml_pipeline()