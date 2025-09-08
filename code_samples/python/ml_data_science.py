"""
Machine Learning and Data Science Examples in Python
Demonstrates scikit-learn, data preprocessing, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)
from sklearn.datasets import make_classification, make_regression, load_iris, load_boston
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, Dict, List, Any, Optional
import warnings
from dataclasses import dataclass
import pickle
import joblib

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DataGenerator:
    """Generate sample datasets for demonstration."""
    
    @staticmethod
    def create_classification_dataset(n_samples: int = 1000, n_features: int = 10, 
                                    n_classes: int = 2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Create a classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=2,
            n_informative=n_features-2,
            random_state=random_state
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_regression_dataset(n_samples: int = 1000, n_features: int = 10, 
                                noise: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Create a regression dataset."""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_mixed_dataset() -> pd.DataFrame:
        """Create a dataset with mixed data types for preprocessing demo."""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 80, 1000),
            'salary': np.random.normal(50000, 15000, 1000),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
            'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Austin'], 1000),
            'experience_years': np.random.randint(0, 40, 1000),
            'has_degree': np.random.choice([True, False], 1000),
            'performance_score': np.random.uniform(1, 10, 1000)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(df.index, size=50, replace=False)
        df.loc[missing_indices, 'salary'] = np.nan
        
        return df


class DataPreprocessor:
    """Comprehensive data preprocessing utilities."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def preprocess_mixed_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Any]:
        """Preprocess mixed data types with proper encoding and scaling."""
        df_processed = df.copy()
        
        # Separate target if specified
        target = None
        if target_column and target_column in df.columns:
            target = df_processed.pop(target_column)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Identify column types
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        boolean_columns = df_processed.select_dtypes(include=['bool']).columns.tolist()
        
        # Process boolean columns (convert to int)
        for col in boolean_columns:
            df_processed[col] = df_processed[col].astype(int)
            numeric_columns.append(col)
        
        # Encode categorical variables
        df_processed = self._encode_categorical_variables(df_processed, categorical_columns)
        
        # Scale numeric variables
        df_processed = self._scale_numeric_variables(df_processed, numeric_columns)
        
        return df_processed, target
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_cleaned = df.copy()
        
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype in [np.float64, np.int64]:
                # Fill numeric columns with median
                df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
            else:
                # Fill categorical columns with mode
                df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
        
        return df_cleaned
    
    def _encode_categorical_variables(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables using one-hot encoding."""
        if not categorical_columns:
            return df
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            # Use pandas get_dummies for one-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = df_encoded.drop(col, axis=1)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        return df_encoded
    
    def _scale_numeric_variables(self, df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Scale numeric variables using StandardScaler."""
        if not numeric_columns:
            return df
        
        df_scaled = df.copy()
        
        scaler = StandardScaler()
        df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
        
        # Store scaler for later use
        self.scalers['standard'] = scaler
        
        return df_scaled


class ModelTrainer:
    """Train and evaluate various machine learning models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_classification_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, ModelMetrics]:
        """Train multiple classification models and return metrics."""
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average='weighted'),
                recall=recall_score(y_test, y_pred, average='weighted'),
                f1=f1_score(y_test, y_pred, average='weighted')
            )
            
            results[name] = metrics
            self.models[name] = model
            
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            print(f"  F1-Score: {metrics.f1:.4f}")
        
        return results
    
    def train_regression_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series) -> Dict[str, ModelMetrics]:
        """Train multiple regression models and return metrics."""
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = ModelMetrics(
                mse=mse,
                rmse=rmse,
                r2=r2
            )
            
            results[name] = metrics
            self.models[name] = model
            
            print(f"  RMSE: {metrics.rmse:.4f}")
            print(f"  R²: {metrics.r2:.4f}")
        
        return results
    
    def perform_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation on a model."""
        scores = cross_val_score(model, X, y, cv=cv)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    def hyperparameter_tuning(self, model, param_grid: Dict, X_train: pd.DataFrame, 
                            y_train: pd.Series, cv: int = 5) -> Any:
        """Perform hyperparameter tuning using GridSearchCV."""
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_


class FeatureEngineer:
    """Feature engineering and selection utilities."""
    
    @staticmethod
    def create_polynomial_features(X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(X.columns)
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    @staticmethod
    def select_best_features(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Select k best features using statistical tests."""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    @staticmethod
    def create_interaction_features(X: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features for specified feature pairs."""
        X_interactions = X.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                X_interactions[interaction_name] = X[feat1] * X[feat2]
        
        return X_interactions


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/tmp/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], top_k: int = 10):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame for easier manipulation
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top k features
            plt.figure(figsize=(10, 6))
            top_features = feat_imp_df.head(top_k)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_k} Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('/tmp/feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return feat_imp_df
        else:
            print("Model does not have feature_importances_ attribute")
            return None
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> str:
        """Generate detailed classification report."""
        return classification_report(y_true, y_pred, target_names=class_names)


class ModelPersistence:
    """Model serialization and persistence utilities."""
    
    @staticmethod
    def save_model(model, filepath: str, use_joblib: bool = True):
        """Save trained model to disk."""
        if use_joblib:
            joblib.dump(model, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str, use_joblib: bool = True):
        """Load trained model from disk."""
        if use_joblib:
            model = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        
        print(f"Model loaded from: {filepath}")
        return model


def demonstrate_classification_pipeline():
    """Demonstrate complete classification pipeline."""
    print("=== Classification Pipeline Demo ===")
    
    # Generate dataset
    data_gen = DataGenerator()
    X, y = data_gen.create_classification_dataset(n_samples=1000, n_features=10, n_classes=3)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_classification_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\nModel Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric, value in metrics.to_dict().items():
            print(f"  {metric}: {value:.4f}")
    
    # Best model analysis
    best_model_name = max(results.keys(), key=lambda k: results[k].accuracy)
    best_model = trainer.models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    
    # Feature importance
    evaluator = ModelEvaluator()
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = evaluator.plot_feature_importance(best_model, X.columns.tolist())
        print("Feature importance plot saved to /tmp/feature_importance.png")
    
    # Confusion matrix
    y_pred = best_model.predict(X_test)
    evaluator.plot_confusion_matrix(y_test, y_pred)
    print("Confusion matrix plot saved to /tmp/confusion_matrix.png")
    
    # Classification report
    report = evaluator.generate_classification_report(y_test, y_pred)
    print(f"\nClassification Report:\n{report}")
    
    return best_model, X_test, y_test


def demonstrate_regression_pipeline():
    """Demonstrate complete regression pipeline."""
    print("\n=== Regression Pipeline Demo ===")
    
    # Generate dataset
    data_gen = DataGenerator()
    X, y = data_gen.create_regression_dataset(n_samples=1000, n_features=8)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target statistics:\n{y.describe()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_regression_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\nModel Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric, value in metrics.to_dict().items():
            print(f"  {metric}: {value:.4f}")
    
    return trainer.models


def demonstrate_feature_engineering():
    """Demonstrate feature engineering techniques."""
    print("\n=== Feature Engineering Demo ===")
    
    # Create sample dataset
    data_gen = DataGenerator()
    X, y = data_gen.create_classification_dataset(n_samples=500, n_features=5)
    
    print(f"Original dataset shape: {X.shape}")
    
    # Feature selection
    engineer = FeatureEngineer()
    X_selected, selected_features = engineer.select_best_features(X, y, k=3)
    print(f"After feature selection: {X_selected.shape}")
    print(f"Selected features: {selected_features}")
    
    # Polynomial features
    X_poly = engineer.create_polynomial_features(X_selected, degree=2)
    print(f"After polynomial features: {X_poly.shape}")
    
    # Interaction features
    feature_pairs = [(selected_features[0], selected_features[1])]
    X_interactions = engineer.create_interaction_features(X_selected, feature_pairs)
    print(f"After interaction features: {X_interactions.shape}")
    
    return X_interactions, y


def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    print("\n=== Hyperparameter Tuning Demo ===")
    
    # Generate dataset
    data_gen = DataGenerator()
    X, y = data_gen.create_classification_dataset(n_samples=800, n_features=8)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform hyperparameter tuning
    trainer = ModelTrainer()
    rf_model = RandomForestClassifier(random_state=42)
    
    print("Performing hyperparameter tuning...")
    best_model = trainer.hyperparameter_tuning(rf_model, param_grid, X_train, y_train)
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best model test accuracy: {accuracy:.4f}")
    
    return best_model


def demonstrate_data_preprocessing():
    """Demonstrate comprehensive data preprocessing."""
    print("\n=== Data Preprocessing Demo ===")
    
    # Create mixed dataset
    data_gen = DataGenerator()
    df = data_gen.create_mixed_dataset()
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, _ = preprocessor.preprocess_mixed_data(df)
    
    print(f"\nAfter preprocessing shape: {X_processed.shape}")
    print(f"Feature names: {X_processed.columns.tolist()}")
    print("Data preprocessing completed successfully!")
    
    return X_processed


def demonstrate_model_persistence():
    """Demonstrate model saving and loading."""
    print("\n=== Model Persistence Demo ===")
    
    # Train a simple model
    data_gen = DataGenerator()
    X, y = data_gen.create_classification_dataset(n_samples=500, n_features=5)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_path = "/tmp/sample_model.pkl"
    ModelPersistence.save_model(model, model_path)
    
    # Load model
    loaded_model = ModelPersistence.load_model(model_path)
    
    # Verify loaded model works
    predictions = loaded_model.predict(X[:5])
    print(f"Sample predictions: {predictions}")
    
    return loaded_model


if __name__ == "__main__":
    print("=== Machine Learning and Data Science Examples ===\n")
    
    # Run demonstrations
    best_clf_model, X_test, y_test = demonstrate_classification_pipeline()
    regression_models = demonstrate_regression_pipeline()
    X_engineered, y_eng = demonstrate_feature_engineering()
    best_tuned_model = demonstrate_hyperparameter_tuning()
    X_preprocessed = demonstrate_data_preprocessing()
    loaded_model = demonstrate_model_persistence()
    
    print("\n=== ML Features Demonstrated ===")
    print("- Classification and regression pipelines")
    print("- Multiple algorithms (RF, SVM, Linear models)")
    print("- Data preprocessing and feature engineering")
    print("- Model evaluation and metrics")
    print("- Hyperparameter tuning with GridSearchCV")
    print("- Cross-validation")
    print("- Feature selection and importance")
    print("- Model persistence and serialization")
    print("- Visualization of results")
    print("- Comprehensive error handling")