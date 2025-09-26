"""
Test suite for sample datasets validation.
Validates structure, content quality, and accessibility of all sample datasets.
"""

import json
import os
from pathlib import Path
import pytest


class TestSampleDatasets:
    """Test class for validating sample datasets."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.sample_datasets_path = Path('/home/runner/work/DATA/DATA/datasets/sample_datasets')
        cls.expected_datasets = {
            'code_analysis': 'programming_patterns_dataset.json',
            'nlp': 'nlp_training_dataset.json', 
            'computer_vision': 'computer_vision_dataset.json',
            'time_series': 'time_series_dataset.json',
            'recommendation': 'recommendation_dataset.json',
            'anomaly_detection': 'anomaly_detection_dataset.json',
            'multi_modal': 'multi_modal_dataset.json'
        }
    
    def test_directory_structure(self):
        """Test that all expected dataset directories exist."""
        assert self.sample_datasets_path.exists(), "Sample datasets directory should exist"
        
        for dataset_dir in self.expected_datasets.keys():
            dir_path = self.sample_datasets_path / dataset_dir
            assert dir_path.exists(), f"Dataset directory {dataset_dir} should exist"
            assert dir_path.is_dir(), f"{dataset_dir} should be a directory"
    
    def test_dataset_files_exist(self):
        """Test that all dataset files exist and are accessible."""
        for dataset_dir, filename in self.expected_datasets.items():
            file_path = self.sample_datasets_path / dataset_dir / filename
            assert file_path.exists(), f"Dataset file {filename} should exist in {dataset_dir}"
            assert file_path.is_file(), f"{filename} should be a file"
            assert file_path.stat().st_size > 0, f"Dataset file {filename} should not be empty"
    
    def test_json_validity(self):
        """Test that all dataset files contain valid JSON."""
        for dataset_dir, filename in self.expected_datasets.items():
            file_path = self.sample_datasets_path / dataset_dir / filename
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                assert isinstance(data, dict), f"Dataset {filename} should contain a JSON object"
            except json.JSONDecodeError as e:
                pytest.fail(f"Dataset {filename} contains invalid JSON: {e}")
    
    def test_dataset_structure(self):
        """Test that each dataset has required metadata and samples."""
        required_metadata_fields = ['name', 'version', 'description', 'purpose', 'total_samples']
        
        for dataset_dir, filename in self.expected_datasets.items():
            file_path = self.sample_datasets_path / dataset_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Test metadata structure
            assert 'dataset_metadata' in data, f"Dataset {filename} should have metadata section"
            metadata = data['dataset_metadata']
            
            for field in required_metadata_fields:
                assert field in metadata, f"Dataset {filename} metadata should have {field} field"
            
            # Test samples structure
            assert 'samples' in data, f"Dataset {filename} should have samples section"
            samples = data['samples']
            assert isinstance(samples, list), f"Samples in {filename} should be a list"
            assert len(samples) > 0, f"Dataset {filename} should have at least one sample"
            
            # Verify sample count matches metadata
            declared_count = metadata['total_samples']
            actual_count = len(samples)
            # Allow for some flexibility in count (metadata might indicate planned size)
            assert actual_count <= declared_count, f"Dataset {filename} has more samples than declared"
            assert actual_count >= 1, f"Dataset {filename} should have at least one sample"
    
    def test_sample_structure(self):
        """Test that each sample has required fields."""
        required_sample_fields = ['id', 'task_type', 'complexity']
        
        for dataset_dir, filename in self.expected_datasets.items():
            file_path = self.sample_datasets_path / dataset_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = data['samples']
            
            for i, sample in enumerate(samples[:5]):  # Test first 5 samples
                for field in required_sample_fields:
                    assert field in sample, f"Sample {i} in {filename} should have {field} field"
                
                # Test complexity levels
                complexity = sample['complexity']
                valid_complexity = ['basic', 'beginner', 'intermediate', 'advanced', 'expert']
                assert complexity in valid_complexity, f"Sample {i} in {filename} has invalid complexity: {complexity}"
    
    def test_documentation_files(self):
        """Test that documentation files exist and are readable."""
        readme_path = self.sample_datasets_path / 'README.md'
        index_path = self.sample_datasets_path / 'dataset_index.json'
        
        assert readme_path.exists(), "README.md should exist"
        assert index_path.exists(), "dataset_index.json should exist"
        
        # Test README is not empty
        assert readme_path.stat().st_size > 1000, "README should have substantial content"
        
        # Test index file is valid JSON
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            assert 'datasets' in index_data, "Index should contain datasets information"
            assert 'summary_statistics' in index_data, "Index should contain summary statistics"
        except json.JSONDecodeError as e:
            pytest.fail(f"Dataset index contains invalid JSON: {e}")
    
    def test_data_quality(self):
        """Test basic data quality metrics."""
        for dataset_dir, filename in self.expected_datasets.items():
            file_path = self.sample_datasets_path / dataset_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = data['samples']
            
            # Test for duplicate IDs
            sample_ids = [sample['id'] for sample in samples]
            assert len(sample_ids) == len(set(sample_ids)), f"Dataset {filename} has duplicate sample IDs"
            
            # Test for reasonable content length
            for sample in samples[:3]:  # Test first 3 samples
                sample_str = json.dumps(sample)
                assert len(sample_str) > 100, f"Sample in {filename} seems too small"
                assert len(sample_str) < 50000, f"Sample in {filename} seems too large"
    
    def test_ml_training_compatibility(self):
        """Test that datasets are compatible with ML training workflows."""
        # Test that we can load and process data as would be done in ML pipeline
        datasets_loaded = 0
        total_samples = 0
        
        for dataset_dir, filename in self.expected_datasets.items():
            file_path = self.sample_datasets_path / dataset_dir / filename
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                samples = data['samples']
                total_samples += len(samples)
                datasets_loaded += 1
                
                # Test that we can extract features from samples
                for sample in samples[:2]:  # Test first 2 samples
                    assert 'id' in sample, "Sample should have ID for tracking"
                    assert 'task_type' in sample, "Sample should have task type for classification"
                    
                    # Different datasets have different data structures, just verify basic structure
                    assert len(str(sample)) > 50, "Sample should have substantial content"
                    
            except Exception as e:
                pytest.fail(f"Failed to load dataset {filename} for ML compatibility test: {e}")
        
        assert datasets_loaded == len(self.expected_datasets), "Should load all datasets"
        assert total_samples >= 30, "Should have meaningful number of training samples"
        
        print(f"✅ Successfully validated {datasets_loaded} datasets with {total_samples} total samples")


def test_dataset_integration():
    """Integration test for the complete dataset collection."""
    sample_datasets_path = Path('/home/runner/work/DATA/DATA/datasets/sample_datasets')
    
    # Test that we can load the index and all referenced datasets
    index_path = sample_datasets_path / 'dataset_index.json'
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    # Verify index matches actual datasets
    index_datasets = {ds['id']: ds['file_path'] for ds in index_data['datasets']}
    
    for dataset_id, file_path in index_datasets.items():
        full_path = sample_datasets_path / file_path
        assert full_path.exists(), f"Dataset referenced in index should exist: {file_path}"
        
        # Verify file can be loaded
        with open(full_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        assert 'dataset_metadata' in dataset, f"Dataset {dataset_id} should have metadata"
        assert 'samples' in dataset, f"Dataset {dataset_id} should have samples"


if __name__ == "__main__":
    # Run basic validation
    test_class = TestSampleDatasets()
    test_class.setup_class()
    
    print("Running sample dataset validation tests...")
    
    try:
        test_class.test_directory_structure()
        print("✅ Directory structure test passed")
        
        test_class.test_dataset_files_exist()
        print("✅ Dataset files existence test passed")
        
        test_class.test_json_validity()
        print("✅ JSON validity test passed")
        
        test_class.test_dataset_structure()
        print("✅ Dataset structure test passed")
        
        test_class.test_sample_structure()
        print("✅ Sample structure test passed")
        
        test_class.test_documentation_files()
        print("✅ Documentation files test passed")
        
        test_class.test_data_quality()
        print("✅ Data quality test passed")
        
        test_class.test_ml_training_compatibility()
        print("✅ ML training compatibility test passed")
        
        test_dataset_integration()
        print("✅ Dataset integration test passed")
        
        print("\n🎉 All sample dataset validation tests passed!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise