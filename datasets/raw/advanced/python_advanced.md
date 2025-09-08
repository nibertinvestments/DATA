# Advanced Python Dataset - Machine Learning and AI Implementation

## Dataset 1: Deep Learning with PyTorch and TensorFlow
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import wandb  # For experiment tracking
from dataclasses import dataclass
import pickle
import json

# Configuration class for model hyperparameters
@dataclass
class ModelConfig:
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Custom Dataset class for structured data
class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets) if targets.dtype == int else torch.FloatTensor(targets)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, target

# Neural Network Architecture with advanced features
class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super(AdvancedNeuralNetwork, self).__init__()
        self.config = config
        
        # Build dynamic architecture
        layers = []
        input_size = config.input_size
        
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, config.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Advanced training class with callbacks and monitoring
class ModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, dataloader: DataLoader, criterion) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy (for classification)
            if len(targets.shape) == 1:  # Classification
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if len(targets.shape) == 1:  # Classification
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == targets).sum().item()
                    total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              criterion, patience: int = 20) -> Dict:
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, criterion)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                print(f'Epoch [{epoch+1}/{self.config.epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 50)
            
            # Early stopping check
            if self.patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss
        }
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Feature engineering and preprocessing pipeline
class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit transformers and transform the data"""
        X = df.drop(columns=[target_col] if target_col else [])
        y = df[target_col].values if target_col else None
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Process numerical features
        for col in numerical_cols:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler
        
        # Process categorical features
        for col in categorical_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            self.encoders[col] = encoder
        
        # Feature engineering
        X = self._create_features(X)
        
        return X.values, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted transformers"""
        X = df.copy()
        
        # Apply fitted scalers
        for col, scaler in self.scalers.items():
            if col in X.columns:
                X[col] = scaler.transform(X[col].values.reshape(-1, 1)).flatten()
        
        # Apply fitted encoders
        for col, encoder in self.encoders.items():
            if col in X.columns:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                mask = X[col].isin(encoder.classes_)
                X.loc[~mask, col] = 'unknown'
                
                # Add 'unknown' to encoder if not present
                if 'unknown' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'unknown')
                
                X[col] = encoder.transform(X[col])
        
        # Feature engineering
        X = self._create_features(X)
        
        return X.values
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        # Example feature engineering
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) >= 2:
            # Interaction features
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
        
        # Polynomial features for numerical columns
        for col in numerical_cols:
            X[f'{col}_squared'] = X[col] ** 2
            X[f'{col}_log'] = np.log(np.abs(X[col]) + 1)
        
        return X

# Model evaluation and interpretation
class ModelEvaluator:
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader: DataLoader, criterion) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Store predictions and targets
                if len(targets.shape) == 1:  # Classification
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                else:  # Regression
                    all_predictions.extend(outputs.cpu().numpy())
                
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions)
        metrics['test_loss'] = avg_loss
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {}
        
        # Determine if classification or regression
        if len(set(y_true)) <= 10 and all(isinstance(x, (int, np.integer)) for x in y_true):
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def get_feature_importance(self, test_loader: DataLoader, feature_names: List[str]) -> Dict:
        """Calculate feature importance using permutation importance"""
        self.model.eval()
        
        # Get baseline performance
        baseline_loss = self._get_model_loss(test_loader)
        
        importances = {}
        
        for i, feature_name in enumerate(feature_names):
            # Create copy of test data with feature permuted
            permuted_loader = self._create_permuted_loader(test_loader, i)
            permuted_loss = self._get_model_loss(permuted_loader)
            
            # Importance is the increase in loss when feature is permuted
            importance = permuted_loss - baseline_loss
            importances[feature_name] = importance
        
        return importances
    
    def _get_model_loss(self, dataloader: DataLoader) -> float:
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _create_permuted_loader(self, dataloader: DataLoader, feature_idx: int) -> DataLoader:
        # This is a simplified version - in practice, you'd need to handle this more carefully
        all_features = []
        all_targets = []
        
        for features, targets in dataloader:
            # Permute the specified feature
            permuted_features = features.clone()
            permuted_features[:, feature_idx] = permuted_features[torch.randperm(len(features)), feature_idx]
            
            all_features.append(permuted_features)
            all_targets.append(targets)
        
        combined_features = torch.cat(all_features)
        combined_targets = torch.cat(all_targets)
        
        dataset = torch.utils.data.TensorDataset(combined_features, combined_targets)
        return DataLoader(dataset, batch_size=dataloader.batch_size)

# Complete ML Pipeline
class MLPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.trainer = None
        self.evaluator = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2):
        """Prepare and split data"""
        # Feature engineering
        X, y = self.feature_engineer.fit_transform(df, target_col)
        
        # Update config with actual input size
        self.config.input_size = X.shape[1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create datasets and dataloaders
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        test_dataset = TabularDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        print(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"Input features: {self.config.input_size}")
        
    def train_model(self):
        """Train the neural network"""
        # Initialize model
        self.model = AdvancedNeuralNetwork(self.config)
        self.trainer = ModelTrainer(self.model, self.config)
        
        # Define loss function
        if self.config.output_size > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Train model
        training_history = self.trainer.train(self.train_loader, self.val_loader, criterion)
        
        # Plot training history
        self.trainer.plot_training_history()
        
        return training_history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        self.evaluator = ModelEvaluator(self.model, self.config.device)
        
        # Evaluate on test set
        if self.config.output_size > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        metrics = self.evaluator.evaluate(self.test_loader, criterion)
        
        print("\nTest Set Evaluation:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the complete pipeline"""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        # Save model state and configuration
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_engineer': self.feature_engineer
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved pipeline"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create pipeline
        pipeline = cls(checkpoint['config'])
        pipeline.feature_engineer = checkpoint['feature_engineer']
        
        # Load model
        pipeline.model = AdvancedNeuralNetwork(checkpoint['config'])
        pipeline.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return pipeline

# Example usage
def example_usage():
    """Example of using the ML pipeline"""
    # Generate sample data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_clusters_per_class=1, random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Configure model
    config = ModelConfig(
        input_size=20,  # Will be updated after feature engineering
        hidden_sizes=[128, 64, 32],
        output_size=2,  # Binary classification
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=50
    )
    
    # Create and run pipeline
    pipeline = MLPipeline(config)
    pipeline.prepare_data(df, 'target')
    training_history = pipeline.train_model()
    metrics = pipeline.evaluate_model()
    
    # Save model
    pipeline.save_model('advanced_model.pth')
    
    return pipeline, metrics

if __name__ == "__main__":
    pipeline, metrics = example_usage()
```

## Dataset 2: Advanced Web Scraping and Data Processing
```python
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import random
import json
import csv
from typing import List, Dict, Optional, Callable
import logging
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import hashlib
import sqlite3
from datetime import datetime, timedelta
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration for scraping
@dataclass
class ScrapingConfig:
    max_workers: int = 10
    delay_min: float = 1.0
    delay_max: float = 3.0
    timeout: int = 30
    retries: int = 3
    user_agent_rotation: bool = True
    use_proxy: bool = False
    proxy_list: Optional[List[str]] = None
    rate_limit_calls: int = 100
    rate_limit_period: int = 60  # seconds
    cache_enabled: bool = True
    cache_expiry_hours: int = 24

# Rate limiter for API calls
class RateLimiter:
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                # Wait until the oldest call is outside the time window
                sleep_time = self.time_window - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            self.calls.append(now)

# Cache system for scraped data
class ScrapingCache:
    def __init__(self, db_path: str = "scraping_cache.db", expiry_hours: int = 24):
        self.db_path = db_path
        self.expiry_hours = expiry_hours
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                url_hash TEXT PRIMARY KEY,
                url TEXT,
                content TEXT,
                timestamp DATETIME,
                content_type TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[str]:
        url_hash = self._get_url_hash(url)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, timestamp FROM cache 
            WHERE url_hash = ? AND datetime(timestamp, '+{} hours') > datetime('now')
        """.format(self.expiry_hours), (url_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def set(self, url: str, content: str, content_type: str = "html"):
        url_hash = self._get_url_hash(url)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache (url_hash, url, content, timestamp, content_type)
            VALUES (?, ?, ?, datetime('now'), ?)
        """, (url_hash, url, content, content_type))
        
        conn.commit()
        conn.close()
    
    def clear_expired(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM cache 
            WHERE datetime(timestamp, '+{} hours') <= datetime('now')
        """.format(self.expiry_hours))
        conn.commit()
        conn.close()

# Advanced HTTP session with retry logic
class AdvancedSession:
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session = requests.Session()
        self.user_agent = UserAgent() if config.user_agent_rotation else None
        self.cache = ScrapingCache() if config.cache_enabled else None
        self.rate_limiter = RateLimiter(config.rate_limit_calls, config.rate_limit_period)
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=config.retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        # Check cache first
        if self.cache:
            cached_content = self.cache.get(url)
            if cached_content:
                # Create a mock response object
                response = requests.Response()
                response.status_code = 200
                response._content = cached_content.encode()
                return response
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Set headers
        headers = kwargs.get('headers', {})
        if self.user_agent:
            headers['User-Agent'] = self.user_agent.random
        
        # Add random delay
        if self.config.delay_max > 0:
            delay = random.uniform(self.config.delay_min, self.config.delay_max)
            time.sleep(delay)
        
        # Make request
        kwargs['headers'] = headers
        kwargs['timeout'] = kwargs.get('timeout', self.config.timeout)
        
        response = self.session.get(url, **kwargs)
        
        # Cache successful responses
        if response.status_code == 200 and self.cache:
            self.cache.set(url, response.text)
        
        return response

# Selenium-based scraper for JavaScript-heavy sites
class SeleniumScraper:
    def __init__(self, config: ScrapingConfig, headless: bool = True):
        self.config = config
        self.headless = headless
        self.driver = None
        self._setup_driver()
    
    def _setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Random user agent
        if self.config.user_agent_rotation:
            ua = UserAgent()
            chrome_options.add_argument(f"--user-agent={ua.random}")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
    
    def get_page(self, url: str, wait_element: Optional[str] = None) -> str:
        self.driver.get(url)
        
        # Wait for specific element if provided
        if wait_element:
            WebDriverWait(self.driver, self.config.timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_element))
            )
        
        # Random delay
        time.sleep(random.uniform(self.config.delay_min, self.config.delay_max))
        
        return self.driver.page_source
    
    def scroll_to_bottom(self, pause_time: float = 1.0):
        """Scroll to bottom of page to load dynamic content"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def click_element(self, selector: str):
        element = WebDriverWait(self.driver, self.config.timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
        )
        element.click()
        time.sleep(random.uniform(0.5, 1.5))
    
    def close(self):
        if self.driver:
            self.driver.quit()

# Data extractor with multiple strategies
class DataExtractor:
    def __init__(self):
        self.extractors = {
            'text': self._extract_text,
            'links': self._extract_links,
            'images': self._extract_images,
            'tables': self._extract_tables,
            'metadata': self._extract_metadata,
            'json_ld': self._extract_json_ld,
            'custom': self._extract_custom
        }
    
    def extract(self, html: str, rules: Dict) -> Dict:
        soup = BeautifulSoup(html, 'html.parser')
        results = {}
        
        for field, rule in rules.items():
            if isinstance(rule, dict):
                extractor_type = rule.get('type', 'text')
                if extractor_type in self.extractors:
                    results[field] = self.extractors[extractor_type](soup, rule)
                else:
                    results[field] = None
            else:
                # Simple CSS selector
                element = soup.select_one(rule)
                results[field] = element.get_text(strip=True) if element else None
        
        return results
    
    def _extract_text(self, soup: BeautifulSoup, rule: Dict) -> Optional[str]:
        selector = rule.get('selector')
        if not selector:
            return None
        
        element = soup.select_one(selector)
        if not element:
            return None
        
        text = element.get_text(strip=True)
        
        # Apply transformations
        if rule.get('clean'):
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        if rule.get('regex'):
            match = re.search(rule['regex'], text)
            return match.group(1) if match else None
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, rule: Dict) -> List[str]:
        selector = rule.get('selector', 'a')
        elements = soup.select(selector)
        
        links = []
        for element in elements:
            href = element.get('href')
            if href:
                # Convert relative URLs to absolute
                base_url = rule.get('base_url', '')
                if base_url and not href.startswith('http'):
                    href = urljoin(base_url, href)
                links.append(href)
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, rule: Dict) -> List[Dict]:
        selector = rule.get('selector', 'img')
        elements = soup.select(selector)
        
        images = []
        for element in elements:
            img_data = {
                'src': element.get('src'),
                'alt': element.get('alt'),
                'title': element.get('title')
            }
            images.append(img_data)
        
        return images
    
    def _extract_tables(self, soup: BeautifulSoup, rule: Dict) -> List[Dict]:
        selector = rule.get('selector', 'table')
        tables = soup.select(selector)
        
        all_tables = []
        for table in tables:
            rows = table.find_all('tr')
            table_data = []
            
            headers = []
            if rows:
                header_row = rows[0]
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    row_data = {}
                    for i, cell in enumerate(cells):
                        header = headers[i] if i < len(headers) else f'column_{i}'
                        row_data[header] = cell.get_text(strip=True)
                    table_data.append(row_data)
            
            all_tables.append(table_data)
        
        return all_tables[0] if len(all_tables) == 1 else all_tables
    
    def _extract_metadata(self, soup: BeautifulSoup, rule: Dict) -> Dict:
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text(strip=True)
        
        return metadata
    
    def _extract_json_ld(self, soup: BeautifulSoup, rule: Dict) -> List[Dict]:
        scripts = soup.find_all('script', type='application/ld+json')
        json_data = []
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                json_data.append(data)
            except json.JSONDecodeError:
                continue
        
        return json_data
    
    def _extract_custom(self, soup: BeautifulSoup, rule: Dict) -> any:
        # Custom extraction function
        custom_func = rule.get('function')
        if custom_func and callable(custom_func):
            return custom_func(soup)
        return None

# Advanced web scraper orchestrator
class AdvancedWebScraper:
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session = AdvancedSession(config)
        self.selenium_scraper = None
        self.extractor = DataExtractor()
        self.results = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def scrape_urls(self, urls: List[str], extraction_rules: Dict, 
                   use_selenium: bool = False) -> List[Dict]:
        """Scrape multiple URLs with concurrent processing"""
        
        if use_selenium:
            return self._scrape_with_selenium(urls, extraction_rules)
        else:
            return self._scrape_with_requests(urls, extraction_rules)
    
    def _scrape_with_requests(self, urls: List[str], extraction_rules: Dict) -> List[Dict]:
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_url = {
                executor.submit(self._scrape_single_url, url, extraction_rules): url 
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        result['url'] = url
                        result['scraped_at'] = datetime.now().isoformat()
                        results.append(result)
                        self.logger.info(f"Successfully scraped: {url}")
                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {str(e)}")
        
        return results
    
    def _scrape_single_url(self, url: str, extraction_rules: Dict) -> Optional[Dict]:
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Extract data
            extracted_data = self.extractor.extract(response.text, extraction_rules)
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return None
    
    def _scrape_with_selenium(self, urls: List[str], extraction_rules: Dict) -> List[Dict]:
        if not self.selenium_scraper:
            self.selenium_scraper = SeleniumScraper(self.config)
        
        results = []
        
        for url in urls:
            try:
                html = self.selenium_scraper.get_page(url)
                extracted_data = self.extractor.extract(html, extraction_rules)
                
                if extracted_data:
                    extracted_data['url'] = url
                    extracted_data['scraped_at'] = datetime.now().isoformat()
                    results.append(extracted_data)
                    self.logger.info(f"Successfully scraped with Selenium: {url}")
                
            except Exception as e:
                self.logger.error(f"Error scraping {url} with Selenium: {str(e)}")
        
        return results
    
    def scrape_paginated_site(self, base_url: str, extraction_rules: Dict, 
                             pagination_rules: Dict, max_pages: int = 100) -> List[Dict]:
        """Scrape a paginated website"""
        all_results = []
        current_page = 1
        
        while current_page <= max_pages:
            page_url = pagination_rules['url_pattern'].format(page=current_page)
            
            try:
                response = self.session.get(page_url)
                response.raise_for_status()
                
                # Extract data from current page
                extracted_data = self.extractor.extract(response.text, extraction_rules)
                
                if extracted_data:
                    # Handle multiple items per page
                    if isinstance(extracted_data.get('items'), list):
                        for item in extracted_data['items']:
                            item['page'] = current_page
                            item['url'] = page_url
                            item['scraped_at'] = datetime.now().isoformat()
                            all_results.append(item)
                    else:
                        extracted_data['page'] = current_page
                        extracted_data['url'] = page_url
                        extracted_data['scraped_at'] = datetime.now().isoformat()
                        all_results.append(extracted_data)
                
                # Check if there's a next page
                soup = BeautifulSoup(response.text, 'html.parser')
                next_page_element = soup.select_one(pagination_rules.get('next_page_selector'))
                
                if not next_page_element:
                    self.logger.info("No more pages found")
                    break
                
                current_page += 1
                self.logger.info(f"Scraped page {current_page - 1}")
                
            except Exception as e:
                self.logger.error(f"Error scraping page {current_page}: {str(e)}")
                break
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str, format: str = 'json'):
        """Save scraping results to file"""
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            if results:
                df = pd.json_normalize(results)
                df.to_csv(filename, index=False)
        
        elif format == 'excel':
            if results:
                df = pd.json_normalize(results)
                df.to_excel(filename, index=False)
        
        self.logger.info(f"Results saved to {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.selenium_scraper:
            self.selenium_scraper.close()

# Example usage and demo
def example_news_scraper():
    """Example: Scraping news articles"""
    
    config = ScrapingConfig(
        max_workers=5,
        delay_min=1.0,
        delay_max=2.0,
        timeout=30,
        retries=3,
        user_agent_rotation=True,
        cache_enabled=True
    )
    
    scraper = AdvancedWebScraper(config)
    
    # Define extraction rules for news articles
    extraction_rules = {
        'title': {
            'type': 'text',
            'selector': 'h1, .article-title, .entry-title',
            'clean': True
        },
        'content': {
            'type': 'text',
            'selector': '.article-content, .entry-content, .post-content',
            'clean': True
        },
        'author': {
            'type': 'text',
            'selector': '.author, .byline, [rel="author"]',
            'clean': True
        },
        'publish_date': {
            'type': 'text',
            'selector': 'time, .publish-date, .date',
            'clean': True
        },
        'images': {
            'type': 'images',
            'selector': '.article-content img'
        },
        'links': {
            'type': 'links',
            'selector': '.article-content a',
            'base_url': 'https://example-news.com'
        }
    }
    
    # Example URLs (replace with real news URLs)
    urls = [
        'https://example-news.com/article1',
        'https://example-news.com/article2',
        'https://example-news.com/article3'
    ]
    
    # Scrape the URLs
    results = scraper.scrape_urls(urls, extraction_rules)
    
    # Save results
    scraper.save_results(results, 'news_articles.json', 'json')
    scraper.save_results(results, 'news_articles.csv', 'csv')
    
    # Cleanup
    scraper.cleanup()
    
    return results

def example_ecommerce_scraper():
    """Example: Scraping product information"""
    
    config = ScrapingConfig(
        max_workers=3,
        delay_min=2.0,
        delay_max=4.0,
        timeout=30,
        cache_enabled=True
    )
    
    scraper = AdvancedWebScraper(config)
    
    # Product extraction rules
    extraction_rules = {
        'name': {
            'type': 'text',
            'selector': 'h1.product-title, .product-name',
            'clean': True
        },
        'price': {
            'type': 'text',
            'selector': '.price, .product-price',
            'regex': r'[\d,]+\.?\d*',
            'clean': True
        },
        'description': {
            'type': 'text',
            'selector': '.product-description',
            'clean': True
        },
        'rating': {
            'type': 'text',
            'selector': '.rating, .stars',
            'regex': r'(\d+\.?\d*)',
            'clean': True
        },
        'availability': {
            'type': 'text',
            'selector': '.availability, .stock-status',
            'clean': True
        },
        'specifications': {
            'type': 'tables',
            'selector': '.specifications table'
        }
    }
    
    # For paginated product listings
    pagination_rules = {
        'url_pattern': 'https://example-shop.com/products?page={page}',
        'next_page_selector': '.pagination .next:not(.disabled)'
    }
    
    # Scrape paginated site
    results = scraper.scrape_paginated_site(
        'https://example-shop.com',
        extraction_rules,
        pagination_rules,
        max_pages=10
    )
    
    # Save results
    scraper.save_results(results, 'products.json', 'json')
    
    scraper.cleanup()
    return results

if __name__ == "__main__":
    # Run examples
    print("Running news scraper example...")
    # news_results = example_news_scraper()
    
    print("Running e-commerce scraper example...")
    # ecommerce_results = example_ecommerce_scraper()
    
    print("Scraping examples completed!")
```