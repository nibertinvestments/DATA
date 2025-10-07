#!/usr/bin/env python3
"""
Dataset Index Generator
Creates a comprehensive index of all available datasets in the repository.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class DatasetIndexer:
    """Creates and maintains dataset index."""
    
    def __init__(self):
        """Initialize indexer."""
        self.base_dir = Path("datasets")
        self.raw_dir = self.base_dir / "raw" / "external"
        
    def generate_index(self) -> Dict[str, Any]:
        """Generate comprehensive dataset index."""
        print("📊 Generating dataset index...\n")
        
        index = {
            "generated_at": datetime.now().isoformat(),
            "repository": "nibertinvestments/DATA",
            "description": "Comprehensive dataset index for LLM/ML/AI training",
            "total_datasets": 0,
            "categories": {},
            "datasets": []
        }
        
        # Index external datasets
        if self.raw_dir.exists():
            external_datasets = self._index_directory(self.raw_dir)
            index["datasets"].extend(external_datasets)
            print(f"✅ Indexed {len(external_datasets)} external datasets")
        
        # Calculate totals
        index["total_datasets"] = len(index["datasets"])
        
        # Group by category
        for dataset in index["datasets"]:
            category = dataset.get("category", "uncategorized")
            if category not in index["categories"]:
                index["categories"][category] = []
            index["categories"][category].append(dataset["name"])
        
        return index
    
    def _index_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Index all datasets in a directory."""
        datasets = []
        
        for file_path in directory.glob("*_dataset.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract metadata
                metadata = data.get("metadata", {})
                
                dataset_info = {
                    "name": file_path.stem.replace("_dataset", ""),
                    "filename": file_path.name,
                    "path": str(file_path.relative_to(Path("datasets"))),
                    "size_kb": round(file_path.stat().st_size / 1024, 2),
                    "category": self._categorize_dataset(file_path.stem),
                    "description": metadata.get("description", ""),
                    "languages": metadata.get("languages", []),
                    "total_samples": self._count_samples(data),
                    "created_at": metadata.get("created_at", "")
                }
                
                datasets.append(dataset_info)
                
            except Exception as e:
                print(f"⚠️  Error indexing {file_path.name}: {e}")
        
        return datasets
    
    def _categorize_dataset(self, name: str) -> str:
        """Categorize dataset based on name."""
        # Check each category keyword
        if "error" in name:
            return "error_handling"
        elif "api" in name or "documentation" in name:
            return "api_patterns"
        elif "translation" in name:
            return "code_translation"
        elif "design" in name or "pattern" in name:
            return "software_design"
        elif "security" in name or "vulnerabilit" in name:
            return "security"
        elif "performance" in name or "optimization" in name:
            return "performance"
        elif "test" in name:
            return "testing"
        elif "database" in name or "db" in name:
            return "database"
        elif "async" in name or "concurrent" in name:
            return "async_concurrent"
        elif "github" in name or "search" in name:
            return "code_samples"
        else:
            return "general"
    
    def _count_samples(self, data: Dict[str, Any]) -> int:
        """Count total samples in dataset."""
        # Try different keys that might contain samples
        for key in ["error_patterns", "api_patterns", "translation_examples", 
                    "patterns", "vulnerabilities", "optimizations", 
                    "testing_examples", "samples"]:
            if key in data:
                return len(data[key])
        
        return 0
    
    def save_index(self, index: Dict[str, Any], output_file: str = "datasets/DATASET_INDEX.json") -> None:
        """Save index to file."""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Index saved to: {output_path}")
    
    def print_summary(self, index: Dict[str, Any]) -> None:
        """Print index summary."""
        print("\n" + "=" * 70)
        print("Dataset Index Summary")
        print("=" * 70)
        print(f"Total Datasets: {index['total_datasets']}")
        print(f"\nCategories ({len(index['categories'])}):")
        
        for category, datasets in sorted(index['categories'].items()):
            print(f"  • {category}: {len(datasets)} dataset(s)")
        
        print(f"\nTotal Size: {sum(d['size_kb'] for d in index['datasets']):.2f} KB")
        print(f"Total Samples: {sum(d['total_samples'] for d in index['datasets'])}")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    indexer = DatasetIndexer()
    index = indexer.generate_index()
    indexer.save_index(index)
    indexer.print_summary(index)


if __name__ == "__main__":
    main()
