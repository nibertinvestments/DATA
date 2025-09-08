"""
ML Dataset Validation and Testing Suite
=======================================

This script validates the quality and integrity of processed ML datasets,
ensuring they are ready for training AI coding agents.

Features:
- Data integrity checks
- Schema validation
- Feature quality assessment
- Train/validation/test split verification
- Performance benchmarks
- ML readiness assessment
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class DatasetValidator:
    """Validates ML datasets for quality and readiness."""

    def __init__(self, processed_dir: str = "datasets/processed"):
        """Initialize the dataset validator."""
        self.processed_dir = Path(processed_dir)
        self.validation_results = {}

    def validate_dataset_integrity(self, dataset_name: str) -> Dict[str, Any]:
        """Validate basic dataset integrity."""
        print(f"🔍 Validating dataset integrity: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "files_exist": {},
            "file_sizes": {},
            "row_counts": {},
            "split_consistency": {},
            "errors": [],
        }

        # Expected files
        splits = ["full", "train", "validation", "test"]
        formats = ["json", "csv", "pkl"]

        # Check if all files exist
        for split in splits:
            for format_type in formats:
                filename = f"{dataset_name}_{split}.{format_type}"
                filepath = self.processed_dir / filename

                exists = filepath.exists()
                results["files_exist"][filename] = exists

                if exists:
                    try:
                        size = filepath.stat().st_size
                        results["file_sizes"][filename] = size

                        # Load and check row count
                        if format_type == "csv":
                            df = pd.read_csv(filepath)
                            results["row_counts"][filename] = len(df)
                        elif format_type == "json":
                            with open(filepath, "r") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                results["row_counts"][filename] = len(data)
                            else:
                                results["row_counts"][filename] = 1
                        elif format_type == "pkl":
                            df = pd.read_pickle(filepath)
                            results["row_counts"][filename] = len(df)

                    except Exception as e:
                        results["errors"].append(f"Error reading {filename}: {str(e)}")
                else:
                    results["errors"].append(f"Missing file: {filename}")

        # Check split consistency
        for format_type in formats:
            try:
                full_count = results["row_counts"].get(
                    f"{dataset_name}_full.{format_type}", 0
                )
                train_count = results["row_counts"].get(
                    f"{dataset_name}_train.{format_type}", 0
                )
                val_count = results["row_counts"].get(
                    f"{dataset_name}_validation.{format_type}", 0
                )
                test_count = results["row_counts"].get(
                    f"{dataset_name}_test.{format_type}", 0
                )

                split_sum = train_count + val_count + test_count

                results["split_consistency"][format_type] = {
                    "full_count": full_count,
                    "split_sum": split_sum,
                    "consistent": abs(full_count - split_sum)
                    <= 1,  # Allow for rounding
                }

                if abs(full_count - split_sum) > 1:
                    results["errors"].append(
                        f"Split inconsistency in {format_type}: full={full_count}, sum={split_sum}"
                    )

            except Exception as e:
                results["errors"].append(
                    f"Error checking split consistency for {format_type}: {str(e)}"
                )

        # Overall integrity score
        total_files = len(splits) * len(formats)
        existing_files = sum(1 for exists in results["files_exist"].values() if exists)
        results["integrity_score"] = existing_files / total_files
        results["is_valid"] = (
            results["integrity_score"] >= 0.8 and len(results["errors"]) == 0
        )

        return results

    def validate_data_quality(self, dataset_name: str) -> Dict[str, Any]:
        """Validate data quality metrics."""
        print(f"📊 Validating data quality: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "missing_values": {},
            "duplicate_rows": {},
            "feature_statistics": {},
            "outliers": {},
            "data_types": {},
            "quality_score": 0,
            "issues": [],
        }

        try:
            # Load full dataset
            df = pd.read_csv(self.processed_dir / f"{dataset_name}_full.csv")

            # Check missing values
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100

            results["missing_values"] = {
                "counts": missing_counts.to_dict(),
                "percentages": missing_percentages.to_dict(),
                "columns_with_missing": missing_counts[
                    missing_counts > 0
                ].index.tolist(),
            }

            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            results["duplicate_rows"] = {
                "count": int(duplicate_count),
                "percentage": float((duplicate_count / len(df)) * 100),
            }

            # Feature statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=["object"]).columns

            results["feature_statistics"] = {
                "total_features": len(df.columns),
                "numeric_features": len(numeric_columns),
                "categorical_features": len(categorical_columns),
                "boolean_features": len(
                    [col for col in df.columns if df[col].dtype == "bool"]
                ),
            }

            # Check for outliers in numeric columns
            outlier_info = {}
            for col in numeric_columns:
                if df[col].dtype in ["int64", "float64"]:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_info[col] = {
                        "count": int(outliers),
                        "percentage": float((outliers / len(df)) * 100),
                    }

            results["outliers"] = outlier_info

            # Data type consistency
            results["data_types"] = df.dtypes.astype(str).to_dict()

            # Calculate quality score
            quality_factors = []

            # Missing values penalty
            avg_missing_pct = missing_percentages.mean()
            missing_score = max(
                0, 100 - avg_missing_pct * 2
            )  # Penalty for missing values
            quality_factors.append(missing_score)

            # Duplicate penalty
            duplicate_penalty = min(50, duplicate_count)  # Cap penalty at 50%
            duplicate_score = max(0, 100 - duplicate_penalty)
            quality_factors.append(duplicate_score)

            # Feature diversity score
            feature_diversity = min(
                100, (len(df.columns) / 10) * 100
            )  # Reward for having features
            quality_factors.append(feature_diversity)

            results["quality_score"] = np.mean(quality_factors)

            # Identify issues
            if avg_missing_pct > 10:
                results["issues"].append(
                    f"High missing values: {avg_missing_pct:.1f}% average"
                )

            if duplicate_count > len(df) * 0.05:
                results["issues"].append(
                    f"High duplicate rate: {(duplicate_count/len(df)*100):.1f}%"
                )

            for col, outlier_data in outlier_info.items():
                if outlier_data["percentage"] > 20:
                    results["issues"].append(
                        f"High outliers in {col}: {outlier_data['percentage']:.1f}%"
                    )

        except Exception as e:
            results["issues"].append(f"Error during quality validation: {str(e)}")
            results["quality_score"] = 0

        return results

    def validate_ml_readiness(self, dataset_name: str) -> Dict[str, Any]:
        """Validate ML readiness of the dataset."""
        print(f"🤖 Validating ML readiness: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "feature_engineering": {},
            "target_variables": {},
            "data_splits": {},
            "scalability": {},
            "ml_readiness_score": 0,
            "recommendations": [],
        }

        try:
            # Load datasets
            train_df = pd.read_csv(self.processed_dir / f"{dataset_name}_train.csv")
            val_df = pd.read_csv(self.processed_dir / f"{dataset_name}_validation.csv")
            test_df = pd.read_csv(self.processed_dir / f"{dataset_name}_test.csv")

            # Feature engineering assessment
            numeric_features = train_df.select_dtypes(include=[np.number]).columns
            categorical_features = train_df.select_dtypes(include=["object"]).columns
            boolean_features = [
                col for col in train_df.columns if train_df[col].dtype == "bool"
            ]

            results["feature_engineering"] = {
                "numeric_features": len(numeric_features),
                "categorical_features": len(categorical_features),
                "boolean_features": len(boolean_features),
                "total_features": len(train_df.columns),
                "has_encoded_categoricals": any("_" in col for col in train_df.columns),
                "feature_density": len(train_df.columns) / len(train_df),
            }

            # Identify potential target variables
            potential_targets = []
            score_columns = [col for col in train_df.columns if "score" in col.lower()]
            label_columns = [
                col
                for col in train_df.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["label", "class", "target", "prediction", "result"]
                )
            ]

            potential_targets.extend(score_columns)
            potential_targets.extend(label_columns)

            results["target_variables"] = {
                "potential_targets": potential_targets,
                "score_based_targets": score_columns,
                "classification_targets": label_columns,
            }

            # Data splits assessment
            total_samples = len(train_df) + len(val_df) + len(test_df)

            results["data_splits"] = {
                "total_samples": total_samples,
                "train_samples": len(train_df),
                "validation_samples": len(val_df),
                "test_samples": len(test_df),
                "train_ratio": len(train_df) / total_samples,
                "validation_ratio": len(val_df) / total_samples,
                "test_ratio": len(test_df) / total_samples,
                "minimum_samples_met": len(train_df) >= 50,  # Minimum for ML
                "balanced_splits": 0.6 <= len(train_df) / total_samples <= 0.8,
            }

            # Scalability assessment
            results["scalability"] = {
                "memory_efficient": len(train_df.columns)
                < 1000,  # Manageable feature count
                "computation_efficient": len(train_df)
                < 100000,  # Reasonable sample count
                "suitable_for_training": len(train_df)
                >= 10,  # Minimum viable training set
            }

            # Calculate ML readiness score
            readiness_factors = []

            # Feature engineering score
            if results["feature_engineering"]["has_encoded_categoricals"]:
                readiness_factors.append(85)
            else:
                readiness_factors.append(60)

            # Target variable score
            if len(potential_targets) > 0:
                readiness_factors.append(90)
            else:
                readiness_factors.append(40)

            # Data split score
            if (
                results["data_splits"]["balanced_splits"]
                and results["data_splits"]["minimum_samples_met"]
            ):
                readiness_factors.append(95)
            else:
                readiness_factors.append(50)

            # Scalability score
            scalability_score = (
                sum(results["scalability"].values()) / len(results["scalability"]) * 100
            )
            readiness_factors.append(scalability_score)

            results["ml_readiness_score"] = np.mean(readiness_factors)

            # Generate recommendations
            if not results["feature_engineering"]["has_encoded_categoricals"]:
                results["recommendations"].append(
                    "Consider encoding categorical variables"
                )

            if len(potential_targets) == 0:
                results["recommendations"].append(
                    "Identify or create target variables for supervised learning"
                )

            if not results["data_splits"]["minimum_samples_met"]:
                results["recommendations"].append(
                    "Increase training set size (minimum 50 samples recommended)"
                )

            if not results["data_splits"]["balanced_splits"]:
                results["recommendations"].append(
                    "Rebalance data splits (recommend 70/15/15 or 80/10/10)"
                )

            if results["feature_engineering"]["feature_density"] > 1:
                results["recommendations"].append(
                    "Consider feature selection to reduce dimensionality"
                )

        except Exception as e:
            results["recommendations"].append(
                f"Error during ML readiness validation: {str(e)}"
            )
            results["ml_readiness_score"] = 0

        return results

    def generate_validation_report(self, dataset_names: List[str]) -> Dict[str, Any]:
        """Generate comprehensive validation report for all datasets."""
        print("📋 Generating comprehensive validation report...\n")

        report = {
            "validation_date": datetime.now().isoformat(),
            "datasets_validated": len(dataset_names),
            "overall_summary": {},
            "dataset_results": {},
            "recommendations": [],
            "passed_validation": [],
        }

        integrity_scores = []
        quality_scores = []
        ml_readiness_scores = []

        for dataset_name in dataset_names:
            print(f"📊 Validating dataset: {dataset_name}")

            # Run all validations
            integrity_results = self.validate_dataset_integrity(dataset_name)
            quality_results = self.validate_data_quality(dataset_name)
            ml_readiness_results = self.validate_ml_readiness(dataset_name)

            # Combine results
            dataset_results = {
                "integrity": integrity_results,
                "quality": quality_results,
                "ml_readiness": ml_readiness_results,
                "overall_score": np.mean(
                    [
                        integrity_results["integrity_score"] * 100,
                        quality_results["quality_score"],
                        ml_readiness_results["ml_readiness_score"],
                    ]
                ),
                "validation_passed": (
                    integrity_results["is_valid"]
                    and quality_results["quality_score"] >= 70
                    and ml_readiness_results["ml_readiness_score"] >= 70
                ),
            }

            report["dataset_results"][dataset_name] = dataset_results

            # Collect scores
            integrity_scores.append(integrity_results["integrity_score"] * 100)
            quality_scores.append(quality_results["quality_score"])
            ml_readiness_scores.append(ml_readiness_results["ml_readiness_score"])

            if dataset_results["validation_passed"]:
                report["passed_validation"].append(dataset_name)

            print(f"  ✅ Integrity: {integrity_results['integrity_score']*100:.1f}%")
            print(f"  ✅ Quality: {quality_results['quality_score']:.1f}%")
            print(
                f"  ✅ ML Readiness: {ml_readiness_results['ml_readiness_score']:.1f}%"
            )
            print(f"  📊 Overall: {dataset_results['overall_score']:.1f}%")
            print()

        # Calculate overall summary
        report["overall_summary"] = {
            "average_integrity_score": np.mean(integrity_scores),
            "average_quality_score": np.mean(quality_scores),
            "average_ml_readiness_score": np.mean(ml_readiness_scores),
            "overall_average_score": np.mean(
                [
                    np.mean(integrity_scores),
                    np.mean(quality_scores),
                    np.mean(ml_readiness_scores),
                ]
            ),
            "datasets_passed": len(report["passed_validation"]),
            "pass_rate": len(report["passed_validation"]) / len(dataset_names) * 100,
        }

        # Generate overall recommendations
        if report["overall_summary"]["average_quality_score"] < 80:
            report["recommendations"].append(
                "Focus on improving data quality across datasets"
            )

        if report["overall_summary"]["average_ml_readiness_score"] < 75:
            report["recommendations"].append(
                "Enhance feature engineering and target variable identification"
            )

        if report["overall_summary"]["pass_rate"] < 80:
            report["recommendations"].append(
                "Address validation issues before using datasets for ML training"
            )

        return report

    def save_validation_report(self, report: Dict[str, Any]) -> None:
        """Save validation report to file."""
        output_file = self.processed_dir / "validation_report.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"📋 Validation report saved to: {output_file}")

    def run_full_validation(self) -> None:
        """Run full validation suite on all processed datasets."""
        print("🔍 Starting comprehensive dataset validation...\n")

        # Identify datasets to validate
        dataset_names = []
        for file_path in self.processed_dir.glob("*_metadata.json"):
            dataset_name = file_path.stem.replace("_metadata", "")
            dataset_names.append(dataset_name)

        if not dataset_names:
            print("❌ No processed datasets found for validation")
            return

        print(f"📊 Found {len(dataset_names)} datasets to validate:")
        for name in dataset_names:
            print(f"  - {name}")
        print()

        # Generate validation report
        report = self.generate_validation_report(dataset_names)

        # Save report
        self.save_validation_report(report)

        # Print summary
        print("🎯 Validation Summary:")
        print(f"  📊 Datasets validated: {report['datasets_validated']}")
        print(f"  ✅ Datasets passed: {report['overall_summary']['datasets_passed']}")
        print(f"  📈 Pass rate: {report['overall_summary']['pass_rate']:.1f}%")
        print(
            f"  🏆 Overall score: {report['overall_summary']['overall_average_score']:.1f}%"
        )
        print()

        if report["recommendations"]:
            print("💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        else:
            print("🎉 All datasets are in excellent condition!")

        print(
            f"\n📁 Detailed report saved to: {self.processed_dir}/validation_report.json"
        )


if __name__ == "__main__":
    validator = DatasetValidator()
    validator.run_full_validation()
