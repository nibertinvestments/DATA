"""
Test validation for newly added AI/ML training datasets.
"""

import json
import os
import pytest


class TestNewDatasets:
    """Test suite for validating new datasets."""
    
    DATASETS_DIR = "datasets/processed"
    
    NEW_DATASETS = [
        "nlp_training_comprehensive.json",
        "computer_vision_training_comprehensive.json",
        "time_series_forecasting_comprehensive.json",
        "reinforcement_learning_comprehensive.json",
        "data_preprocessing_feature_engineering.json",
        "model_evaluation_metrics_comprehensive.json",
        "api_design_patterns_comprehensive.json",
        "database_design_optimization.json",
        "programming_patterns_idioms.json",
        "cloud_infrastructure_devops_patterns.json"
    ]
    
    def test_all_datasets_exist(self):
        """Test that all new datasets exist."""
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            assert os.path.exists(filepath), f"Dataset not found: {dataset_name}"
    
    def test_datasets_valid_json(self):
        """Test that all datasets are valid JSON."""
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    assert isinstance(data, dict), f"{dataset_name} is not a JSON object"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {dataset_name}: {e}")
    
    def test_datasets_have_metadata(self):
        """Test that all datasets have required metadata."""
        required_fields = ['dataset_name', 'version', 'description', 'created_at', 'sample_count']
        
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                assert 'metadata' in data, f"{dataset_name} missing metadata"
                metadata = data['metadata']
                
                for field in required_fields:
                    assert field in metadata, f"{dataset_name} missing metadata field: {field}"
    
    def test_datasets_have_training_samples(self):
        """Test that datasets have training samples."""
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                assert 'training_samples' in data, f"{dataset_name} missing training_samples"
                samples = data['training_samples']
                
                assert isinstance(samples, list), f"{dataset_name} training_samples not a list"
                assert len(samples) > 0, f"{dataset_name} has no training samples"
    
    def test_sample_structure(self):
        """Test that training samples have correct structure."""
        required_sample_fields = ['id', 'category', 'title']
        
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                samples = data.get('training_samples', [])
                
                if samples:  # If there are samples, validate structure
                    sample = samples[0]  # Check first sample
                    for field in required_sample_fields:
                        assert field in sample, f"{dataset_name} sample missing field: {field}"
    
    def test_total_sample_count(self):
        """Test that total sample count meets expectations."""
        total_samples = 0
        
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                sample_count = data.get('metadata', {}).get('sample_count', 0)
                total_samples += sample_count
        
        # We expect at least 2000 samples across all new datasets
        assert total_samples >= 2000, f"Total samples ({total_samples}) less than expected"
        print(f"✅ Total samples across new datasets: {total_samples}")
    
    def test_dataset_categories(self):
        """Test that datasets have defined categories."""
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                
                assert 'categories' in metadata, f"{dataset_name} missing categories"
                categories = metadata['categories']
                
                assert isinstance(categories, list), f"{dataset_name} categories not a list"
                assert len(categories) > 0, f"{dataset_name} has no categories"
    
    def test_complexity_levels(self):
        """Test that datasets define complexity levels."""
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                
                if 'complexity_levels' in metadata:
                    levels = metadata['complexity_levels']
                    assert isinstance(levels, list), f"{dataset_name} complexity_levels not a list"
    
    def test_use_cases(self):
        """Test that datasets define practical use cases."""
        for dataset_name in self.NEW_DATASETS:
            filepath = os.path.join(self.DATASETS_DIR, dataset_name)
            with open(filepath, 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                
                if 'use_cases' in metadata:
                    use_cases = metadata['use_cases']
                    assert isinstance(use_cases, list), f"{dataset_name} use_cases not a list"
                    assert len(use_cases) > 0, f"{dataset_name} has no use cases"


def test_dataset_summary():
    """Print summary of new datasets."""
    datasets_dir = "datasets/processed"
    new_datasets = [
        "nlp_training_comprehensive.json",
        "computer_vision_training_comprehensive.json",
        "time_series_forecasting_comprehensive.json",
        "reinforcement_learning_comprehensive.json",
        "data_preprocessing_feature_engineering.json",
        "model_evaluation_metrics_comprehensive.json",
        "api_design_patterns_comprehensive.json",
        "database_design_optimization.json",
        "programming_patterns_idioms.json",
        "cloud_infrastructure_devops_patterns.json"
    ]
    
    print("\n" + "=" * 70)
    print("NEW AI/ML TRAINING DATASETS SUMMARY")
    print("=" * 70)
    
    total_samples = 0
    for dataset_name in new_datasets:
        filepath = os.path.join(datasets_dir, dataset_name)
        with open(filepath, 'r') as f:
            data = json.load(f)
            metadata = data.get('metadata', {})
            sample_count = metadata.get('sample_count', 0)
            total_samples += sample_count
            
            print(f"\n{dataset_name}:")
            print(f"  Name: {metadata.get('dataset_name', 'N/A')}")
            print(f"  Samples: {sample_count}")
            print(f"  Categories: {len(metadata.get('categories', []))}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL NEW SAMPLES: {total_samples}")
    print(f"TOTAL NEW DATASETS: {len(new_datasets)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
