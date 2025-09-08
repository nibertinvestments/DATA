#!/usr/bin/env python3
"""
ML Training Data Generator for Code Samples
Processes code samples into various ML training formats.
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CodeSample:
    """Represents a code sample with metadata."""
    language: str
    filename: str
    content: str
    file_path: str
    size_bytes: int
    line_count: int
    function_count: int
    class_count: int
    comment_lines: int
    complexity_score: float
    algorithms: List[str]
    data_structures: List[str]
    design_patterns: List[str]
    file_hash: str

@dataclass
class MLDataset:
    """Represents a processed ML dataset."""
    dataset_type: str
    language: str
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CodeAnalyzer:
    """Analyzes code samples to extract features and metadata."""
    
    # Language-specific patterns for analysis
    LANGUAGE_PATTERNS = {
        'python': {
            'function': r'def\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s\(:]+',
            'comment': r'#.*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
            'import': r'import\s+(\w+)|from\s+(\w+)\s+import',
        },
        'javascript': {
            'function': r'function\s+(\w+)\s*\(|(\w+)\s*=\s*function|\w+\s*=>\s*{|(\w+)\s*\([^)]*\)\s*=>',
            'class': r'class\s+(\w+)[\s{]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]|require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        },
        'typescript': {
            'function': r'function\s+(\w+)\s*\(|(\w+)\s*=\s*function|\w+\s*=>\s*{|(\w+)\s*\([^)]*\)\s*=>',
            'class': r'class\s+(\w+)[\s{<]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
        },
        'rust': {
            'function': r'fn\s+(\w+)\s*\(',
            'class': r'struct\s+(\w+)|enum\s+(\w+)',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'use\s+([^;]+);',
        },
        'go': {
            'function': r'func\s+(\w+)\s*\(',
            'class': r'type\s+(\w+)\s+struct',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+[\'"]([^\'"]+)[\'"]',
        },
        'csharp': {
            'function': r'(?:public|private|protected|internal)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'class': r'(?:public|private|protected|internal)?\s*class\s+(\w+)',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'using\s+([^;]+);',
        },
        'php': {
            'function': r'function\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s{]+',
            'comment': r'//.*|/\*[\s\S]*?\*/|#.*',
            'import': r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)|include\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        },
        'ruby': {
            'function': r'def\s+(\w+)[\s\(]+',
            'class': r'class\s+(\w+)[\s<]+',
            'comment': r'#.*',
            'import': r'require\s+[\'"]([^\'"]+)[\'"]',
        },
        'swift': {
            'function': r'func\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s{:<]+|struct\s+(\w+)[\s{:<]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+(\w+)',
        },
        'java': {
            'function': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'class': r'(?:public|private|protected)?\s*class\s+(\w+)',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'import\s+([^;]+);',
        },
        'cpp': {
            'function': r'\w+\s+(\w+)\s*\(',
            'class': r'class\s+(\w+)[\s{:]+',
            'comment': r'//.*|/\*[\s\S]*?\*/',
            'import': r'#include\s*[<"]([^>"]+)[>"]',
        }
    }
    
    # Algorithm keywords for detection
    ALGORITHM_KEYWORDS = {
        'sorting': ['sort', 'bubble', 'quick', 'merge', 'heap', 'insertion', 'selection'],
        'searching': ['search', 'binary', 'linear', 'find', 'lookup'],
        'graph': ['dfs', 'bfs', 'dijkstra', 'graph', 'node', 'edge', 'vertex'],
        'dynamic_programming': ['memo', 'dp', 'fibonacci', 'knapsack', 'lcs'],
        'recursion': ['recursive', 'recursion', 'factorial'],
        'hashing': ['hash', 'dictionary', 'map'],
        'string': ['string', 'substring', 'palindrome', 'anagram'],
        'math': ['gcd', 'lcm', 'prime', 'factorial', 'fibonacci'],
        'concurrency': ['thread', 'async', 'await', 'parallel', 'concurrent', 'mutex', 'lock']
    }
    
    # Data structure keywords
    DATA_STRUCTURE_KEYWORDS = {
        'array': ['array', 'list', 'vector'],
        'stack': ['stack', 'push', 'pop'],
        'queue': ['queue', 'enqueue', 'dequeue'],
        'tree': ['tree', 'binary', 'node', 'leaf', 'root'],
        'graph': ['graph', 'vertex', 'edge', 'adjacency'],
        'hash_table': ['hash', 'dictionary', 'map', 'bucket'],
        'linked_list': ['linked', 'next', 'node'],
        'heap': ['heap', 'priority', 'min', 'max']
    }
    
    # Design pattern keywords
    DESIGN_PATTERN_KEYWORDS = {
        'singleton': ['singleton', 'instance'],
        'factory': ['factory', 'create'],
        'observer': ['observer', 'notify', 'update'],
        'strategy': ['strategy', 'algorithm'],
        'decorator': ['decorator', 'wrapper'],
        'builder': ['builder', 'build'],
        'adapter': ['adapter', 'interface'],
        'facade': ['facade', 'wrapper']
    }
    
    def analyze_file(self, file_path: Path, language: str) -> CodeSample:
        """Analyze a single code file and extract features."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {file_path} as UTF-8, trying with errors='ignore'")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        # Basic metrics
        size_bytes = len(content.encode('utf-8'))
        lines = content.split('\n')
        line_count = len(lines)
        
        # Language-specific analysis
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        
        # Count functions and classes
        function_count = len(re.findall(patterns.get('function', ''), content, re.IGNORECASE)) if 'function' in patterns else 0
        class_count = len(re.findall(patterns.get('class', ''), content, re.IGNORECASE)) if 'class' in patterns else 0
        
        # Count comment lines
        comment_pattern = patterns.get('comment', '')
        comment_lines = 0
        if comment_pattern:
            comment_matches = re.findall(comment_pattern, content, re.MULTILINE)
            comment_lines = sum(match.count('\n') + 1 for match in comment_matches if match)
        
        # Calculate complexity score (simple heuristic)
        complexity_score = self._calculate_complexity(content, language)
        
        # Detect algorithms, data structures, and patterns
        algorithms = self._detect_keywords(content, self.ALGORITHM_KEYWORDS)
        data_structures = self._detect_keywords(content, self.DATA_STRUCTURE_KEYWORDS)
        design_patterns = self._detect_keywords(content, self.DESIGN_PATTERN_KEYWORDS)
        
        # Generate file hash
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        return CodeSample(
            language=language,
            filename=file_path.name,
            content=content,
            file_path=str(file_path),
            size_bytes=size_bytes,
            line_count=line_count,
            function_count=function_count,
            class_count=class_count,
            comment_lines=comment_lines,
            complexity_score=complexity_score,
            algorithms=algorithms,
            data_structures=data_structures,
            design_patterns=design_patterns,
            file_hash=file_hash
        )
    
    def _calculate_complexity(self, content: str, language: str) -> float:
        """Calculate a simple complexity score based on code features."""
        score = 0.0
        
        # Nested structures (loops, conditions)
        nested_patterns = [
            r'for\s*\([^)]*\)\s*{[^}]*for',  # Nested loops
            r'if\s*\([^)]*\)\s*{[^}]*if',    # Nested conditions
            r'while\s*\([^)]*\)\s*{[^}]*while',  # Nested while loops
        ]
        
        for pattern in nested_patterns:
            score += len(re.findall(pattern, content, re.IGNORECASE | re.DOTALL)) * 2
        
        # Function calls
        function_calls = len(re.findall(r'\w+\s*\(', content))
        score += function_calls * 0.1
        
        # Conditional statements
        conditionals = len(re.findall(r'\b(if|else|switch|case)\b', content, re.IGNORECASE))
        score += conditionals * 0.5
        
        # Loops
        loops = len(re.findall(r'\b(for|while|do)\b', content, re.IGNORECASE))
        score += loops * 0.5
        
        # Normalize by line count
        lines = len(content.split('\n'))
        if lines > 0:
            score = score / lines * 100
        
        return round(score, 2)
    
    def _detect_keywords(self, content: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
        """Detect keywords in content and return matching categories."""
        content_lower = content.lower()
        detected = []
        
        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword in content_lower:
                    detected.append(category)
                    break
        
        return detected

class MLDatasetGenerator:
    """Generates ML training datasets from code samples."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.analyzer = CodeAnalyzer()
    
    def process_code_samples(self, code_samples_dir: Path) -> List[CodeSample]:
        """Process all code samples in the directory."""
        samples = []
        
        for lang_dir in code_samples_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            
            language = lang_dir.name
            logger.info(f"Processing {language} files...")
            
            for code_file in lang_dir.iterdir():
                if code_file.is_file() and not code_file.name.startswith('.'):
                    try:
                        sample = self.analyzer.analyze_file(code_file, language)
                        samples.append(sample)
                        logger.debug(f"Processed {code_file}")
                    except Exception as e:
                        logger.error(f"Error processing {code_file}: {e}")
        
        logger.info(f"Processed {len(samples)} code samples across {len(set(s.language for s in samples))} languages")
        return samples
    
    def save_dataset(self, dataset: MLDataset, filename: str):
        """Save dataset to JSON file."""
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(dataset), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {dataset.dataset_type} dataset to {output_path}")
        logger.info(f"Dataset contains {len(dataset.samples)} samples")
    
    def save_raw_samples(self, samples: List[CodeSample], filename: str):
        """Save raw code samples to JSON file."""
        output_path = self.output_dir / f"{filename}.json"
        
        samples_data = [asdict(sample) for sample in samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(samples)} raw samples to {output_path}")

def main():
    """Main function to process code samples and generate ML datasets."""
    # Set up paths
    repo_root = Path(__file__).parent.parent.parent
    code_samples_dir = repo_root / "code_samples"
    datasets_dir = repo_root / "datasets"
    processed_dir = datasets_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting ML dataset generation...")
    logger.info(f"Code samples directory: {code_samples_dir}")
    logger.info(f"Output directory: {processed_dir}")
    
    # Initialize generator
    generator = MLDatasetGenerator(processed_dir)
    
    # Process code samples
    samples = generator.process_code_samples(code_samples_dir)
    
    if not samples:
        logger.error("No code samples found!")
        return
    
    # Save raw samples
    generator.save_raw_samples(samples, "raw_code_samples")
    
    # Generate summary statistics
    stats = {
        'total_samples': len(samples),
        'languages': list(set(s.language for s in samples)),
        'language_distribution': {lang: len([s for s in samples if s.language == lang]) 
                                 for lang in set(s.language for s in samples)},
        'total_lines_of_code': sum(s.line_count for s in samples),
        'total_functions': sum(s.function_count for s in samples),
        'total_classes': sum(s.class_count for s in samples),
        'algorithms_covered': list(set(algo for s in samples for algo in s.algorithms)),
        'data_structures_covered': list(set(ds for s in samples for ds in s.data_structures)),
        'design_patterns_covered': list(set(pattern for s in samples for pattern in s.design_patterns))
    }
    
    stats_path = processed_dir / "dataset_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Dataset generation completed!")
    logger.info(f"Generated datasets saved to: {processed_dir}")
    logger.info(f"Statistics saved to: {stats_path}")
    
    # Print summary
    print("\n=== ML Dataset Generation Summary ===")
    print(f"Original samples: {stats['total_samples']}")
    print(f"Languages: {', '.join(stats['languages'])}")
    print(f"Total lines of code: {stats['total_lines_of_code']:,}")
    print(f"Algorithms covered: {', '.join(stats['algorithms_covered'])}")
    print(f"Data structures covered: {', '.join(stats['data_structures_covered'])}")
    print(f"Design patterns covered: {', '.join(stats['design_patterns_covered'])}")

if __name__ == "__main__":
    main()