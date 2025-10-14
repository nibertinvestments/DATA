#!/usr/bin/env python3
"""
Synthetic Code Pattern Generator
Generates large volumes of diverse code patterns for AI training.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class SyntheticCodeGenerator:
    """Generate synthetic code patterns for AI training."""
    
    def __init__(self, output_dir: str = "datasets/processed"):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_algorithm_variants_dataset(self) -> Dict[str, Any]:
        """Generate various algorithm implementations."""
        print("🔢 Generating algorithm variants dataset...")
        
        samples = []
        
        # Sorting algorithms
        sorting_algos = [
            "bubble_sort", "insertion_sort", "selection_sort",
            "merge_sort", "quick_sort", "heap_sort", "radix_sort",
            "counting_sort", "bucket_sort", "shell_sort"
        ]
        
        for algo in sorting_algos:
            for lang in ["python", "javascript", "java", "cpp", "go"]:
                samples.append({
                    "id": f"algo_{len(samples)+1:05d}",
                    "algorithm": algo,
                    "language": lang,
                    "category": "sorting",
                    "time_complexity": "varies by algorithm",
                    "space_complexity": "varies by algorithm"
                })
        
        # Search algorithms
        search_algos = [
            "linear_search", "binary_search", "jump_search",
            "interpolation_search", "exponential_search",
            "fibonacci_search", "ternary_search"
        ]
        
        for algo in search_algos:
            for lang in ["python", "javascript", "java", "cpp", "rust"]:
                samples.append({
                    "id": f"algo_{len(samples)+1:05d}",
                    "algorithm": algo,
                    "language": lang,
                    "category": "searching",
                    "time_complexity": "varies by algorithm",
                    "space_complexity": "O(1) to O(n)"
                })
        
        # Graph algorithms
        graph_algos = [
            "bfs", "dfs", "dijkstra", "bellman_ford", "floyd_warshall",
            "prims_mst", "kruskals_mst", "topological_sort",
            "strongly_connected_components", "articulation_points"
        ]
        
        for algo in graph_algos:
            for lang in ["python", "java", "cpp", "go"]:
                samples.append({
                    "id": f"algo_{len(samples)+1:05d}",
                    "algorithm": algo,
                    "language": lang,
                    "category": "graph",
                    "time_complexity": "varies by algorithm",
                    "space_complexity": "O(V) to O(V+E)"
                })
        
        # Dynamic programming
        dp_algos = [
            "fibonacci", "knapsack", "longest_common_subsequence",
            "edit_distance", "coin_change", "matrix_chain_multiplication",
            "rod_cutting", "subset_sum", "partition_problem"
        ]
        
        for algo in dp_algos:
            for lang in ["python", "java", "cpp"]:
                samples.append({
                    "id": f"algo_{len(samples)+1:05d}",
                    "algorithm": algo,
                    "language": lang,
                    "category": "dynamic_programming",
                    "time_complexity": "O(n²) to O(n³)",
                    "space_complexity": "O(n) to O(n²)"
                })
        
        return {
            "metadata": {
                "name": "algorithm_variants_synthetic_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "categories": ["sorting", "searching", "graph", "dynamic_programming"],
                "purpose": "AI algorithm understanding training"
            },
            "samples": samples
        }
    
    def generate_data_structure_patterns_dataset(self) -> Dict[str, Any]:
        """Generate data structure implementation patterns."""
        print("📦 Generating data structure patterns dataset...")
        
        samples = []
        
        # Basic data structures
        basic_ds = [
            "array", "linked_list", "doubly_linked_list", "circular_linked_list",
            "stack", "queue", "deque", "priority_queue"
        ]
        
        for ds in basic_ds:
            for lang in ["python", "javascript", "java", "cpp", "go", "rust"]:
                for operation in ["insert", "delete", "search", "traverse", "update"]:
                    samples.append({
                        "id": f"ds_{len(samples)+1:05d}",
                        "data_structure": ds,
                        "language": lang,
                        "operation": operation,
                        "category": "linear",
                        "complexity": "varies by operation"
                    })
        
        # Tree structures
        tree_ds = [
            "binary_tree", "binary_search_tree", "avl_tree", "red_black_tree",
            "b_tree", "trie", "segment_tree", "fenwick_tree", "heap"
        ]
        
        for ds in tree_ds:
            for lang in ["python", "java", "cpp", "go"]:
                for operation in ["insert", "delete", "search", "traverse"]:
                    samples.append({
                        "id": f"ds_{len(samples)+1:05d}",
                        "data_structure": ds,
                        "language": lang,
                        "operation": operation,
                        "category": "tree",
                        "complexity": "O(log n) to O(n)"
                    })
        
        # Hash-based structures
        hash_ds = ["hash_table", "hash_set", "hash_map", "bloom_filter"]
        
        for ds in hash_ds:
            for lang in ["python", "javascript", "java", "cpp", "go"]:
                for operation in ["insert", "delete", "lookup", "contains"]:
                    samples.append({
                        "id": f"ds_{len(samples)+1:05d}",
                        "data_structure": ds,
                        "language": lang,
                        "operation": operation,
                        "category": "hashing",
                        "complexity": "O(1) average"
                    })
        
        # Graph structures
        graph_ds = ["adjacency_matrix", "adjacency_list", "edge_list"]
        
        for ds in graph_ds:
            for lang in ["python", "java", "cpp"]:
                for operation in ["add_vertex", "add_edge", "remove_edge", "get_neighbors"]:
                    samples.append({
                        "id": f"ds_{len(samples)+1:05d}",
                        "data_structure": ds,
                        "language": lang,
                        "operation": operation,
                        "category": "graph",
                        "complexity": "varies by representation"
                    })
        
        return {
            "metadata": {
                "name": "data_structure_patterns_synthetic_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "categories": ["linear", "tree", "hashing", "graph"],
                "purpose": "AI data structure understanding training"
            },
            "samples": samples
        }
    
    def generate_design_patterns_dataset(self) -> Dict[str, Any]:
        """Generate design pattern implementations."""
        print("🎨 Generating design patterns dataset...")
        
        samples = []
        
        # Creational patterns
        creational = [
            "singleton", "factory_method", "abstract_factory", "builder",
            "prototype", "object_pool"
        ]
        
        for pattern in creational:
            for lang in ["python", "javascript", "java", "cpp", "csharp", "go"]:
                samples.append({
                    "id": f"pattern_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "category": "creational",
                    "use_case": f"{pattern} implementation in {lang}"
                })
        
        # Structural patterns
        structural = [
            "adapter", "bridge", "composite", "decorator", "facade",
            "flyweight", "proxy"
        ]
        
        for pattern in structural:
            for lang in ["python", "javascript", "java", "cpp", "csharp"]:
                samples.append({
                    "id": f"pattern_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "category": "structural",
                    "use_case": f"{pattern} implementation in {lang}"
                })
        
        # Behavioral patterns
        behavioral = [
            "chain_of_responsibility", "command", "iterator", "mediator",
            "memento", "observer", "state", "strategy", "template_method",
            "visitor"
        ]
        
        for pattern in behavioral:
            for lang in ["python", "javascript", "java", "csharp"]:
                samples.append({
                    "id": f"pattern_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "category": "behavioral",
                    "use_case": f"{pattern} implementation in {lang}"
                })
        
        return {
            "metadata": {
                "name": "design_patterns_synthetic_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "categories": ["creational", "structural", "behavioral"],
                "purpose": "AI design pattern recognition training"
            },
            "samples": samples
        }
    
    def generate_api_design_patterns_dataset(self) -> Dict[str, Any]:
        """Generate API design patterns."""
        print("🌐 Generating API design patterns dataset...")
        
        samples = []
        
        # REST API patterns
        rest_patterns = [
            "resource_based_urls", "http_verbs", "status_codes", "pagination",
            "filtering", "sorting", "versioning", "hateoas", "rate_limiting",
            "authentication", "caching", "error_handling"
        ]
        
        for pattern in rest_patterns:
            for lang in ["python", "javascript", "java", "go", "php"]:
                for framework in ["express", "django", "spring", "gin", "laravel"]:
                    samples.append({
                        "id": f"api_{len(samples)+1:05d}",
                        "pattern": pattern,
                        "language": lang,
                        "framework": framework,
                        "api_type": "REST",
                        "category": "api_design"
                    })
        
        # GraphQL patterns
        graphql_patterns = [
            "schema_definition", "queries", "mutations", "subscriptions",
            "resolvers", "data_loaders", "pagination", "error_handling"
        ]
        
        for pattern in graphql_patterns:
            for lang in ["javascript", "python", "java", "go"]:
                samples.append({
                    "id": f"api_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "api_type": "GraphQL",
                    "category": "api_design"
                })
        
        # WebSocket patterns
        websocket_patterns = [
            "connection_handling", "message_broadcasting", "rooms",
            "authentication", "heartbeat", "reconnection"
        ]
        
        for pattern in websocket_patterns:
            for lang in ["javascript", "python", "java"]:
                samples.append({
                    "id": f"api_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "api_type": "WebSocket",
                    "category": "api_design"
                })
        
        return {
            "metadata": {
                "name": "api_design_patterns_synthetic_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "api_types": ["REST", "GraphQL", "WebSocket"],
                "purpose": "AI API design pattern training"
            },
            "samples": samples
        }
    
    def generate_testing_patterns_dataset(self) -> Dict[str, Any]:
        """Generate testing patterns and strategies."""
        print("🧪 Generating testing patterns dataset...")
        
        samples = []
        
        # Unit testing patterns
        unit_patterns = [
            "arrange_act_assert", "test_doubles", "mocking", "stubbing",
            "test_fixtures", "parameterized_tests", "test_data_builders",
            "test_naming_conventions"
        ]
        
        for pattern in unit_patterns:
            for lang in ["python", "javascript", "java", "csharp", "go"]:
                for framework in ["pytest", "jest", "junit", "nunit", "testing"]:
                    samples.append({
                        "id": f"test_{len(samples)+1:05d}",
                        "pattern": pattern,
                        "language": lang,
                        "framework": framework,
                        "test_type": "unit",
                        "category": "testing"
                    })
        
        # Integration testing patterns
        integration_patterns = [
            "database_testing", "api_testing", "service_testing",
            "test_containers", "test_databases", "mock_servers"
        ]
        
        for pattern in integration_patterns:
            for lang in ["python", "javascript", "java", "go"]:
                samples.append({
                    "id": f"test_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "test_type": "integration",
                    "category": "testing"
                })
        
        # E2E testing patterns
        e2e_patterns = [
            "page_object_model", "test_scenarios", "test_data_management",
            "screenshot_comparison", "performance_testing"
        ]
        
        for pattern in e2e_patterns:
            for tool in ["selenium", "playwright", "cypress", "puppeteer"]:
                samples.append({
                    "id": f"test_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "tool": tool,
                    "test_type": "e2e",
                    "category": "testing"
                })
        
        return {
            "metadata": {
                "name": "testing_patterns_synthetic_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "test_types": ["unit", "integration", "e2e"],
                "purpose": "AI testing strategy training"
            },
            "samples": samples
        }
    
    def generate_concurrency_patterns_dataset(self) -> Dict[str, Any]:
        """Generate concurrency and parallelism patterns."""
        print("⚡ Generating concurrency patterns dataset...")
        
        samples = []
        
        # Threading patterns
        threading_patterns = [
            "thread_pool", "producer_consumer", "reader_writer",
            "mutex", "semaphore", "condition_variable", "barrier",
            "thread_local_storage"
        ]
        
        for pattern in threading_patterns:
            for lang in ["python", "java", "cpp", "go", "rust"]:
                samples.append({
                    "id": f"concurrency_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "category": "threading",
                    "complexity": "medium to high"
                })
        
        # Async patterns
        async_patterns = [
            "async_await", "promises", "futures", "coroutines",
            "event_loop", "callback_hell_solution", "async_generators"
        ]
        
        for pattern in async_patterns:
            for lang in ["python", "javascript", "csharp", "rust"]:
                samples.append({
                    "id": f"concurrency_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "category": "async",
                    "complexity": "medium"
                })
        
        # Parallel processing patterns
        parallel_patterns = [
            "map_reduce", "fork_join", "worker_pool", "pipeline",
            "scatter_gather", "data_parallelism", "task_parallelism"
        ]
        
        for pattern in parallel_patterns:
            for lang in ["python", "java", "go", "rust"]:
                samples.append({
                    "id": f"concurrency_{len(samples)+1:05d}",
                    "pattern": pattern,
                    "language": lang,
                    "category": "parallel",
                    "complexity": "high"
                })
        
        return {
            "metadata": {
                "name": "concurrency_patterns_synthetic_dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "categories": ["threading", "async", "parallel"],
                "purpose": "AI concurrency pattern training"
            },
            "samples": samples
        }
    
    def generate_all(self):
        """Generate all synthetic datasets."""
        print("=" * 70)
        print("Synthetic Code Pattern Generator")
        print("=" * 70)
        print()
        
        datasets = [
            ("synthetic_algorithm_variants.json", self.generate_algorithm_variants_dataset()),
            ("synthetic_data_structure_patterns.json", self.generate_data_structure_patterns_dataset()),
            ("synthetic_design_patterns.json", self.generate_design_patterns_dataset()),
            ("synthetic_api_design_patterns.json", self.generate_api_design_patterns_dataset()),
            ("synthetic_testing_patterns.json", self.generate_testing_patterns_dataset()),
            ("synthetic_concurrency_patterns.json", self.generate_concurrency_patterns_dataset()),
        ]
        
        created_files = []
        total_samples = 0
        
        for filename, dataset in datasets:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            sample_count = len(dataset['samples'])
            total_samples += sample_count
            print(f"✅ Created: {filename} ({sample_count:,} samples)")
            created_files.append(filename)
        
        print()
        print("=" * 70)
        print(f"✅ Successfully created {len(created_files)} synthetic datasets")
        print(f"📊 Total training samples: {total_samples:,}")
        print(f"📁 Location: {self.output_dir}")
        print("=" * 70)
        
        return created_files, total_samples


def main():
    """Main entry point."""
    generator = SyntheticCodeGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
