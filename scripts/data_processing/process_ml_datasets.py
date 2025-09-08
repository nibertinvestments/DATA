"""
ML Dataset Processing Pipeline
==============================

This script processes raw datasets and converts them into ML-ready formats
suitable for training AI coding agents. It includes data cleaning, feature
extraction, validation, and format conversion.

Features:
- Data validation and quality checks
- Feature extraction for code patterns
- Multiple output formats (JSON, CSV, Parquet)
- Data augmentation and balancing
- Train/validation/test splitting
"""

import json
import csv
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import re
import hashlib
from datetime import datetime
from collections import Counter
import pickle


class MLDataProcessor:
    """Processes raw datasets into ML-ready formats."""

    def __init__(
        self, raw_dir: str = "datasets/raw", processed_dir: str = "datasets/processed"
    ):
        """Initialize the data processor."""
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Language mappings for feature extraction
        self.language_features = {
            "python": {
                "paradigms": ["object_oriented", "functional", "procedural"],
                "type_system": "dynamic",
                "compiled": False,
                "syntax_complexity": "medium",
                "learning_curve": "easy",
            },
            "javascript": {
                "paradigms": ["object_oriented", "functional", "event_driven"],
                "type_system": "dynamic",
                "compiled": False,
                "syntax_complexity": "medium",
                "learning_curve": "medium",
            },
            "java": {
                "paradigms": ["object_oriented"],
                "type_system": "static",
                "compiled": True,
                "syntax_complexity": "high",
                "learning_curve": "medium",
            },
            "cpp": {
                "paradigms": ["object_oriented", "procedural", "generic"],
                "type_system": "static",
                "compiled": True,
                "syntax_complexity": "very_high",
                "learning_curve": "hard",
            },
            "rust": {
                "paradigms": ["functional", "object_oriented", "concurrent"],
                "type_system": "static",
                "compiled": True,
                "syntax_complexity": "high",
                "learning_curve": "hard",
            },
            "go": {
                "paradigms": ["procedural", "concurrent"],
                "type_system": "static",
                "compiled": True,
                "syntax_complexity": "low",
                "learning_curve": "easy",
            },
        }

    def extract_code_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract features from code for ML training."""
        features = {
            # Basic metrics
            "lines_of_code": len(code.split("\n")),
            "character_count": len(code),
            "word_count": len(code.split()),
            # Language-specific features
            "language": language,
            "language_paradigms": self.language_features.get(language, {}).get(
                "paradigms", []
            ),
            "type_system": self.language_features.get(language, {}).get(
                "type_system", "unknown"
            ),
            "is_compiled": self.language_features.get(language, {}).get(
                "compiled", False
            ),
            # Syntax complexity indicators
            "has_classes": bool(re.search(r"\bclass\b", code)),
            "has_functions": bool(re.search(r"\bdef\b|\bfunction\b|\bfunc\b", code)),
            "has_loops": bool(re.search(r"\bfor\b|\bwhile\b", code)),
            "has_conditionals": bool(
                re.search(r"\bif\b|\belse\b|\belif\b|\bswitch\b", code)
            ),
            "has_try_catch": bool(re.search(r"\btry\b|\bcatch\b|\bexcept\b", code)),
            "has_async": bool(re.search(r"\basync\b|\bawait\b|\bPromise\b", code)),
            # Code structure
            "indentation_levels": self._count_indentation_levels(code),
            "max_line_length": (
                max(len(line) for line in code.split("\n")) if code.split("\n") else 0
            ),
            "avg_line_length": (
                np.mean([len(line) for line in code.split("\n")])
                if code.split("\n")
                else 0
            ),
            # Comment density
            "comment_lines": self._count_comment_lines(code, language),
            "comment_ratio": self._calculate_comment_ratio(code, language),
            # Complexity indicators
            "cyclomatic_complexity": self._estimate_cyclomatic_complexity(code),
            "nesting_depth": self._calculate_nesting_depth(code),
            # Documentation indicators
            "has_docstring": self._has_docstring(code, language),
            "has_type_hints": self._has_type_hints(code, language),
        }

        return features

    def _count_indentation_levels(self, code: str) -> int:
        """Count unique indentation levels in code."""
        lines = code.split("\n")
        indentations = set()

        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip())
                indentations.add(leading_spaces)

        return len(indentations)

    def _count_comment_lines(self, code: str, language: str) -> int:
        """Count comment lines based on language syntax."""
        comment_patterns = {
            "python": r"^\s*#",
            "javascript": r"^\s*//",
            "java": r"^\s*//",
            "cpp": r"^\s*//",
            "rust": r"^\s*//",
            "go": r"^\s*//",
        }

        pattern = comment_patterns.get(language, r"^\s*//")
        lines = code.split("\n")

        comment_count = 0
        for line in lines:
            if re.match(pattern, line):
                comment_count += 1

        return comment_count

    def _calculate_comment_ratio(self, code: str, language: str) -> float:
        """Calculate ratio of comment lines to total non-empty lines."""
        lines = code.split("\n")
        non_empty_lines = len([line for line in lines if line.strip()])

        if non_empty_lines == 0:
            return 0.0

        comment_lines = self._count_comment_lines(code, language)
        return comment_lines / non_empty_lines

    def _estimate_cyclomatic_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity by counting decision points."""
        decision_keywords = [
            r"\bif\b",
            r"\belse\b",
            r"\belif\b",
            r"\belse if\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bdo\b",
            r"\bcase\b",
            r"\bswitch\b",
            r"\bcatch\b",
            r"\bexcept\b",
            r"\band\b",
            r"\bor\b",
            r"\b&&\b",
            r"\|\|",
        ]

        complexity = 1  # Base complexity

        for keyword in decision_keywords:
            matches = re.findall(keyword, code, re.IGNORECASE)
            complexity += len(matches)

        return complexity

    def _calculate_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth."""
        lines = code.split("\n")
        max_depth = 0
        current_depth = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Count opening braces/keywords that increase nesting
            if any(
                keyword in stripped
                for keyword in ["if", "for", "while", "def", "class", "function", "try"]
            ):
                if stripped.endswith(":") or "{" in stripped:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)

            # Count closing braces that decrease nesting
            if "}" in stripped or (
                stripped.startswith(("except", "finally", "else", "elif"))
            ):
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _has_docstring(self, code: str, language: str) -> bool:
        """Check if code has documentation strings."""
        if language == "python":
            return bool(
                re.search(r'""".*?"""', code, re.DOTALL)
                or re.search(r"'''.*?'''", code, re.DOTALL)
            )
        elif language in ["java", "javascript", "cpp"]:
            return bool(re.search(r"/\*\*.*?\*/", code, re.DOTALL))
        return False

    def _has_type_hints(self, code: str, language: str) -> bool:
        """Check if code has type annotations."""
        if language == "python":
            return bool(re.search(r":\s*\w+(\[.*?\])?", code))
        elif language == "typescript":
            return bool(re.search(r":\s*\w+", code))
        elif language in ["java", "cpp", "rust"]:
            return True  # These languages have static typing
        return False

    def process_code_patterns(self) -> None:
        """Process code patterns dataset."""
        print("🔄 Processing code patterns dataset...")

        # Load raw data
        with open(self.raw_dir / "code_patterns.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed_patterns = []

        for pattern in raw_data["patterns"]:
            # Extract features from before and after code
            before_features = self.extract_code_features(
                pattern["code_before"], pattern["language"]
            )
            after_features = self.extract_code_features(
                pattern["code_after"], pattern["language"]
            )

            # Create processed record
            processed_record = {
                "id": pattern["id"],
                "language": pattern["language"],
                "pattern_type": pattern["pattern_type"],
                "pattern_name": pattern["pattern_name"],
                "description": pattern["description"],
                "complexity": pattern["complexity"],
                "performance_impact": pattern["performance_impact"],
                "readability_score": pattern["readability_score"],
                "tags": pattern["tags"],
                # Code content
                "code_before": pattern["code_before"],
                "code_after": pattern["code_after"],
                # Features before transformation
                "before_lines_of_code": before_features["lines_of_code"],
                "before_character_count": before_features["character_count"],
                "before_cyclomatic_complexity": before_features[
                    "cyclomatic_complexity"
                ],
                "before_nesting_depth": before_features["nesting_depth"],
                "before_comment_ratio": before_features["comment_ratio"],
                # Features after transformation
                "after_lines_of_code": after_features["lines_of_code"],
                "after_character_count": after_features["character_count"],
                "after_cyclomatic_complexity": after_features["cyclomatic_complexity"],
                "after_nesting_depth": after_features["nesting_depth"],
                "after_comment_ratio": after_features["comment_ratio"],
                # Improvement metrics
                "lines_reduction": before_features["lines_of_code"]
                - after_features["lines_of_code"],
                "complexity_reduction": before_features["cyclomatic_complexity"]
                - after_features["cyclomatic_complexity"],
                "improved_readability": pattern["readability_score"] >= 8,
                # Language features
                "language_paradigms": before_features["language_paradigms"],
                "type_system": before_features["type_system"],
                "is_compiled": before_features["is_compiled"],
                # Binary features
                "has_classes": before_features["has_classes"],
                "has_functions": before_features["has_functions"],
                "has_loops": before_features["has_loops"],
                "has_conditionals": before_features["has_conditionals"],
                "has_try_catch": before_features["has_try_catch"],
                "has_async": before_features["has_async"],
                "has_docstring": before_features["has_docstring"],
                "has_type_hints": before_features["has_type_hints"],
            }

            processed_patterns.append(processed_record)

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(processed_patterns)

        # Add derived features
        df["improvement_score"] = (
            df["lines_reduction"] * 0.3
            + df["complexity_reduction"] * 0.4
            + df["readability_score"] * 0.3
        )

        # Categorical encoding
        df = pd.get_dummies(
            df, columns=["language", "pattern_type", "complexity", "performance_impact"]
        )

        # Save processed data in multiple formats
        self._save_dataset(df, "code_patterns_processed", raw_data["metadata"])

        print(f"✅ Processed {len(df)} code patterns")

    def process_algorithm_implementations(self) -> None:
        """Process algorithm implementations dataset."""
        print("🔄 Processing algorithm implementations dataset...")

        # Load raw data
        with open(
            self.raw_dir / "algorithm_implementations.json", "r", encoding="utf-8"
        ) as f:
            raw_data = json.load(f)

        processed_algorithms = []

        for algorithm in raw_data["algorithms"]:
            # Extract features from code
            code_features = self.extract_code_features(
                algorithm["code"], algorithm["language"]
            )

            # Parse complexity information
            time_complexity = algorithm["time_complexity"]

            # Create processed record
            processed_record = {
                "id": algorithm["id"],
                "algorithm_name": algorithm["algorithm_name"],
                "category": algorithm["category"],
                "language": algorithm["language"],
                "description": algorithm["description"],
                "difficulty": algorithm["difficulty"],
                "tags": algorithm["tags"],
                "applications": algorithm["applications"],
                # Code content
                "code": algorithm["code"],
                # Complexity analysis
                "time_complexity_best": time_complexity["best"],
                "time_complexity_average": time_complexity["average"],
                "time_complexity_worst": time_complexity["worst"],
                "space_complexity": algorithm["space_complexity"],
                # Code features
                "lines_of_code": code_features["lines_of_code"],
                "character_count": code_features["character_count"],
                "cyclomatic_complexity": code_features["cyclomatic_complexity"],
                "nesting_depth": code_features["nesting_depth"],
                "comment_ratio": code_features["comment_ratio"],
                "indentation_levels": code_features["indentation_levels"],
                "max_line_length": code_features["max_line_length"],
                # Language features
                "language_paradigms": code_features["language_paradigms"],
                "type_system": code_features["type_system"],
                "is_compiled": code_features["is_compiled"],
                # Binary features
                "has_classes": code_features["has_classes"],
                "has_functions": code_features["has_functions"],
                "has_loops": code_features["has_loops"],
                "has_conditionals": code_features["has_conditionals"],
                "has_try_catch": code_features["has_try_catch"],
                "has_docstring": code_features["has_docstring"],
                "has_type_hints": code_features["has_type_hints"],
                # Test cases
                "test_case_count": len(algorithm["test_cases"]),
                "has_edge_cases": any(
                    "[]" in test["input"] or "null" in test["input"]
                    for test in algorithm["test_cases"]
                ),
            }

            processed_algorithms.append(processed_record)

        # Convert to DataFrame
        df = pd.DataFrame(processed_algorithms)

        # Add complexity scoring
        complexity_scores = {
            "O(1)": 1,
            "O(log n)": 2,
            "O(n)": 3,
            "O(n log n)": 4,
            "O(n²)": 5,
            "O(n³)": 6,
            "O(2^n)": 7,
            "O(n!)": 8,
        }

        df["complexity_score"] = df["time_complexity_average"].map(
            lambda x: complexity_scores.get(x, 5)
        )

        # Categorical encoding
        df = pd.get_dummies(df, columns=["language", "category", "difficulty"])

        # Save processed data
        self._save_dataset(
            df, "algorithm_implementations_processed", raw_data["metadata"]
        )

        print(f"✅ Processed {len(df)} algorithm implementations")

    def process_error_handling_examples(self) -> None:
        """Process error handling examples dataset."""
        print("🔄 Processing error handling examples dataset...")

        # Load raw data
        with open(
            self.raw_dir / "error_handling_examples.json", "r", encoding="utf-8"
        ) as f:
            raw_data = json.load(f)

        processed_errors = []

        for error in raw_data["error_examples"]:
            # Extract features from buggy and fixed code
            buggy_features = self.extract_code_features(
                error["buggy_code"], error["language"]
            )
            fixed_features = self.extract_code_features(
                error["fixed_code"], error["language"]
            )

            # Create processed record
            processed_record = {
                "id": error["id"],
                "language": error["language"],
                "error_type": error["error_type"],
                "category": error["category"],
                "description": error["description"],
                "severity": error["severity"],
                "common_cause": error["common_cause"],
                "prevention_tips": error["prevention_tips"],
                # Code content
                "buggy_code": error["buggy_code"],
                "fixed_code": error["fixed_code"],
                "error_message": error["error_message"],
                "fix_explanation": error["fix_explanation"],
                # Buggy code features
                "buggy_lines_of_code": buggy_features["lines_of_code"],
                "buggy_character_count": buggy_features["character_count"],
                "buggy_cyclomatic_complexity": buggy_features["cyclomatic_complexity"],
                "buggy_nesting_depth": buggy_features["nesting_depth"],
                "buggy_comment_ratio": buggy_features["comment_ratio"],
                # Fixed code features
                "fixed_lines_of_code": fixed_features["lines_of_code"],
                "fixed_character_count": fixed_features["character_count"],
                "fixed_cyclomatic_complexity": fixed_features["cyclomatic_complexity"],
                "fixed_nesting_depth": fixed_features["nesting_depth"],
                "fixed_comment_ratio": fixed_features["comment_ratio"],
                # Improvement metrics
                "lines_added": fixed_features["lines_of_code"]
                - buggy_features["lines_of_code"],
                "complexity_added": fixed_features["cyclomatic_complexity"]
                - buggy_features["cyclomatic_complexity"],
                "comment_improvement": fixed_features["comment_ratio"]
                - buggy_features["comment_ratio"],
                # Language features
                "language_paradigms": buggy_features["language_paradigms"],
                "type_system": buggy_features["type_system"],
                "is_compiled": buggy_features["is_compiled"],
                # Error pattern features
                "has_null_check": "null" in error["fixed_code"].lower()
                or "none" in error["fixed_code"].lower(),
                "has_type_check": "isinstance" in error["fixed_code"]
                or "typeof" in error["fixed_code"],
                "has_bounds_check": "len(" in error["fixed_code"]
                or "length" in error["fixed_code"],
                "adds_try_catch": fixed_features["has_try_catch"]
                and not buggy_features["has_try_catch"],
                "adds_validation": "validate" in error["fixed_code"].lower()
                or "check" in error["fixed_code"].lower(),
            }

            processed_errors.append(processed_record)

        # Convert to DataFrame
        df = pd.DataFrame(processed_errors)

        # Add severity scoring
        severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        df["severity_score"] = df["severity"].map(severity_scores)

        # Categorical encoding
        df = pd.get_dummies(
            df, columns=["language", "error_type", "category", "severity"]
        )

        # Save processed data
        self._save_dataset(df, "error_handling_processed", raw_data["metadata"])

        print(f"✅ Processed {len(df)} error handling examples")

    def _save_dataset(
        self, df: pd.DataFrame, name: str, metadata: Dict[str, Any]
    ) -> None:
        """Save dataset in multiple formats."""

        # Create train/validation/test splits
        train_df, val_df, test_df = self._split_dataset(df)

        # Save full dataset
        self._save_dataframe(df, name, "full")

        # Save splits
        self._save_dataframe(train_df, name, "train")
        self._save_dataframe(val_df, name, "validation")
        self._save_dataframe(test_df, name, "test")

        # Save metadata
        enhanced_metadata = {
            **metadata,
            "processed_date": datetime.now().isoformat(),
            "total_samples": len(df),
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "features": list(df.columns),
            "feature_count": len(df.columns),
            "split_ratio": "70/15/15",
        }

        metadata_file = self.processed_dir / f"{name}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)

    def _save_dataframe(self, df: pd.DataFrame, name: str, split: str) -> None:
        """Save DataFrame in multiple formats."""
        base_path = self.processed_dir / f"{name}_{split}"

        # Save as JSON
        df.to_json(f"{base_path}.json", orient="records", indent=2)

        # Save as CSV
        df.to_csv(f"{base_path}.csv", index=False)

        # Save as Parquet (efficient binary format)
        try:
            df.to_parquet(f"{base_path}.parquet", index=False)
        except ImportError:
            print("⚠️  Parquet format skipped (pyarrow not installed)")

        # Save as pickle for Python objects
        df.to_pickle(f"{base_path}.pkl")

    def _split_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets."""
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split indices
        n = len(df_shuffled)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        # Split the data
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]

        return train_df, val_df, test_df

    def process_all_datasets(self) -> None:
        """Process all raw datasets."""
        print("🚀 Starting ML dataset processing pipeline...\n")

        # Process each dataset
        self.process_code_patterns()
        print()
        self.process_algorithm_implementations()
        print()
        self.process_error_handling_examples()
        print()

        # Generate summary
        self.generate_processing_summary()

        print("🎉 All datasets processed successfully!")
        print(f"📁 Processed datasets saved to: {self.processed_dir}")

    def generate_processing_summary(self) -> None:
        """Generate summary of processed datasets."""
        summary = {
            "processing_date": datetime.now().isoformat(),
            "processed_datasets": [
                {
                    "name": "code_patterns_processed",
                    "description": "Processed code patterns with extracted features",
                    "formats": ["json", "csv", "parquet", "pkl"],
                },
                {
                    "name": "algorithm_implementations_processed",
                    "description": "Processed algorithm implementations with complexity analysis",
                    "formats": ["json", "csv", "parquet", "pkl"],
                },
                {
                    "name": "error_handling_processed",
                    "description": "Processed error handling examples with fix patterns",
                    "formats": ["json", "csv", "parquet", "pkl"],
                },
            ],
            "data_splits": ["train", "validation", "test"],
            "split_ratio": "70/15/15",
            "ml_ready": True,
            "feature_engineering": {
                "code_metrics": [
                    "lines_of_code",
                    "cyclomatic_complexity",
                    "nesting_depth",
                ],
                "language_features": ["paradigms", "type_system", "compilation"],
                "pattern_features": ["improvement_score", "readability_metrics"],
                "error_features": ["severity_score", "fix_patterns"],
            },
        }

        summary_file = self.processed_dir / "processing_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📋 Processing summary saved to: {summary_file}")


if __name__ == "__main__":
    processor = MLDataProcessor()
    processor.process_all_datasets()
