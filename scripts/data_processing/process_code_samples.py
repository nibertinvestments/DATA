"""
Data Processing Script for ML Training Datasets
Converts raw code samples into structured training data for AI coding agents
"""

import os
import json
import ast
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd

class CodeDataProcessor:
    """Processes code samples into ML-ready training datasets"""
    
    def __init__(self, base_path: str = "/home/runner/work/DATA/DATA"):
        self.base_path = Path(base_path)
        self.code_samples_path = self.base_path / "code_samples"
        self.datasets_path = self.base_path / "datasets"
        self.processed_path = self.datasets_path / "processed"
        
        # Ensure output directories exist
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def extract_code_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract features from code for ML training"""
        features = {
            'language': language,
            'length': len(code),
            'lines': len(code.split('\n')),
            'complexity_score': self._calculate_complexity(code, language),
            'has_classes': self._has_classes(code, language),
            'has_functions': self._has_functions(code, language),
            'has_imports': self._has_imports(code, language),
            'has_comments': self._has_comments(code, language),
            'has_error_handling': self._has_error_handling(code, language),
            'has_async': self._has_async_patterns(code, language),
            'keywords': self._extract_keywords(code, language),
            'patterns': self._identify_patterns(code, language)
        }
        return features
    
    def _calculate_complexity(self, code: str, language: str) -> int:
        """Calculate cyclomatic complexity approximation"""
        complexity_keywords = {
            'python': ['if', 'elif', 'while', 'for', 'try', 'except', 'with', 'and', 'or'],
            'javascript': ['if', 'else', 'while', 'for', 'try', 'catch', '&&', '||', 'switch'],
            'typescript': ['if', 'else', 'while', 'for', 'try', 'catch', '&&', '||', 'switch'],
            'java': ['if', 'else', 'while', 'for', 'try', 'catch', '&&', '||', 'switch'],
            'csharp': ['if', 'else', 'while', 'for', 'try', 'catch', '&&', '||', 'switch'],
            'cpp': ['if', 'else', 'while', 'for', 'try', 'catch', '&&', '||', 'switch'],
            'rust': ['if', 'else', 'while', 'for', 'match', 'loop', '&&', '||'],
            'go': ['if', 'else', 'while', 'for', 'switch', '&&', '||'],
            'php': ['if', 'else', 'while', 'for', 'try', 'catch', '&&', '||', 'switch']
        }
        
        keywords = complexity_keywords.get(language, [])
        complexity = 1  # Base complexity
        
        for keyword in keywords:
            complexity += len(re.findall(r'\b' + keyword + r'\b', code, re.IGNORECASE))
        
        return complexity
    
    def _has_classes(self, code: str, language: str) -> bool:
        """Check if code contains class definitions"""
        patterns = {
            'python': r'\bclass\s+\w+',
            'javascript': r'\bclass\s+\w+',
            'typescript': r'\bclass\s+\w+',
            'java': r'\bclass\s+\w+',
            'csharp': r'\bclass\s+\w+',
            'cpp': r'\bclass\s+\w+',
            'rust': r'\bstruct\s+\w+|\btrait\s+\w+|\bimpl\s+',
            'go': r'\btype\s+\w+\s+struct',
            'php': r'\bclass\s+\w+'
        }
        
        pattern = patterns.get(language, '')
        return bool(re.search(pattern, code, re.IGNORECASE))
    
    def _has_functions(self, code: str, language: str) -> bool:
        """Check if code contains function definitions"""
        patterns = {
            'python': r'\bdef\s+\w+',
            'javascript': r'\bfunction\s+\w+|\w+\s*=\s*\(',
            'typescript': r'\bfunction\s+\w+|\w+\s*=\s*\(',
            'java': r'\b(public|private|protected).*\w+\s*\(',
            'csharp': r'\b(public|private|protected).*\w+\s*\(',
            'cpp': r'\w+\s+\w+\s*\(',
            'rust': r'\bfn\s+\w+',
            'go': r'\bfunc\s+\w+',
            'php': r'\bfunction\s+\w+'
        }
        
        pattern = patterns.get(language, '')
        return bool(re.search(pattern, code, re.IGNORECASE))
    
    def _has_imports(self, code: str, language: str) -> bool:
        """Check if code contains imports/includes"""
        patterns = {
            'python': r'\bimport\s+\w+|\bfrom\s+\w+\s+import',
            'javascript': r'\bimport\s+.*\bfrom|\brequire\s*\(',
            'typescript': r'\bimport\s+.*\bfrom|\brequire\s*\(',
            'java': r'\bimport\s+\w+',
            'csharp': r'\busing\s+\w+',
            'cpp': r'#include\s*<.*>|#include\s*".*"',
            'rust': r'\buse\s+\w+',
            'go': r'\bimport\s+',
            'php': r'\buse\s+\w+|\brequire|\binclude'
        }
        
        pattern = patterns.get(language, '')
        return bool(re.search(pattern, code, re.IGNORECASE))
    
    def _has_comments(self, code: str, language: str) -> bool:
        """Check if code contains comments"""
        patterns = {
            'python': r'#.*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
            'javascript': r'//.*|/\*[\s\S]*?\*/',
            'typescript': r'//.*|/\*[\s\S]*?\*/',
            'java': r'//.*|/\*[\s\S]*?\*/',
            'csharp': r'//.*|/\*[\s\S]*?\*/',
            'cpp': r'//.*|/\*[\s\S]*?\*/',
            'rust': r'//.*|/\*[\s\S]*?\*/',
            'go': r'//.*|/\*[\s\S]*?\*/',
            'php': r'//.*|/\*[\s\S]*?\*/|#.*'
        }
        
        pattern = patterns.get(language, '')
        return bool(re.search(pattern, code, re.MULTILINE | re.DOTALL))
    
    def _has_error_handling(self, code: str, language: str) -> bool:
        """Check if code contains error handling"""
        patterns = {
            'python': r'\btry:|except\s+\w+:|finally:',
            'javascript': r'\btry\s*{|\bcatch\s*\(|\bfinally\s*{',
            'typescript': r'\btry\s*{|\bcatch\s*\(|\bfinally\s*{',
            'java': r'\btry\s*{|\bcatch\s*\(|\bfinally\s*{',
            'csharp': r'\btry\s*{|\bcatch\s*\(|\bfinally\s*{',
            'cpp': r'\btry\s*{|\bcatch\s*\(',
            'rust': r'\bResult<|\bOption<|\bmatch\s+.*{|\b\.unwrap\(',
            'go': r'\bif\s+err\s*!=\s*nil|\berror\s+interface',
            'php': r'\btry\s*{|\bcatch\s*\('
        }
        
        pattern = patterns.get(language, '')
        return bool(re.search(pattern, code, re.IGNORECASE))
    
    def _has_async_patterns(self, code: str, language: str) -> bool:
        """Check if code contains async/concurrent patterns"""
        patterns = {
            'python': r'\basync\s+def|\bawait\s+|\bthreading\.|multiprocessing\.',
            'javascript': r'\basync\s+function|\bawait\s+|\bPromise\.|\.then\(',
            'typescript': r'\basync\s+function|\bawait\s+|\bPromise\.|\.then\(',
            'java': r'\bCompletableFuture|\bExecutorService|\b@Async',
            'csharp': r'\basync\s+Task|\bawait\s+|\bTask\.|\.ConfigureAwait',
            'cpp': r'\bstd::thread|\bstd::async|\bstd::future',
            'rust': r'\basync\s+fn|\bawait|\btokio::|async_std::',
            'go': r'\bgo\s+func|\bgoroutine|\bchan\s+',
            'php': r'\bPsr\\Http\\Message|\bGuzzleHttp'
        }
        
        pattern = patterns.get(language, '')
        return bool(re.search(pattern, code, re.IGNORECASE))
    
    def _extract_keywords(self, code: str, language: str) -> List[str]:
        """Extract programming keywords and important identifiers"""
        # Common programming keywords
        keywords_map = {
            'python': ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 
                      'try', 'except', 'with', 'async', 'await', 'return', 'yield'],
            'javascript': ['function', 'class', 'import', 'export', 'if', 'else', 'for', 'while',
                          'try', 'catch', 'async', 'await', 'return', 'const', 'let', 'var'],
            'typescript': ['function', 'class', 'interface', 'type', 'import', 'export', 'if', 'else',
                          'for', 'while', 'try', 'catch', 'async', 'await', 'return'],
            'java': ['class', 'interface', 'import', 'package', 'if', 'else', 'for', 'while',
                    'try', 'catch', 'public', 'private', 'protected', 'static', 'final'],
            'csharp': ['class', 'interface', 'using', 'namespace', 'if', 'else', 'for', 'while',
                      'try', 'catch', 'public', 'private', 'protected', 'static', 'async'],
            'cpp': ['class', 'struct', 'namespace', 'include', 'if', 'else', 'for', 'while',
                   'try', 'catch', 'public', 'private', 'protected', 'template'],
            'rust': ['fn', 'struct', 'trait', 'impl', 'use', 'mod', 'if', 'else', 'for', 'while',
                    'match', 'pub', 'const', 'let', 'mut'],
            'go': ['func', 'type', 'struct', 'interface', 'import', 'package', 'if', 'else',
                  'for', 'switch', 'select', 'go', 'chan', 'var', 'const'],
            'php': ['class', 'interface', 'trait', 'use', 'namespace', 'if', 'else', 'for', 'while',
                   'try', 'catch', 'function', 'public', 'private', 'protected']
        }
        
        keywords = keywords_map.get(language, [])
        found_keywords = []
        
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', code, re.IGNORECASE):
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _identify_patterns(self, code: str, language: str) -> List[str]:
        """Identify common programming patterns and concepts"""
        patterns = []
        
        # Design patterns
        if re.search(r'\bsingleton\b|\bSingleton\b', code, re.IGNORECASE):
            patterns.append('singleton_pattern')
        if re.search(r'\bfactory\b|\bFactory\b', code, re.IGNORECASE):
            patterns.append('factory_pattern')
        if re.search(r'\bobserver\b|\bObserver\b|\bsubscribe\b', code, re.IGNORECASE):
            patterns.append('observer_pattern')
        if re.search(r'\bstrategy\b|\bStrategy\b', code, re.IGNORECASE):
            patterns.append('strategy_pattern')
        
        # Architectural patterns
        if re.search(r'\brepository\b|\bRepository\b', code, re.IGNORECASE):
            patterns.append('repository_pattern')
        if re.search(r'\bservice\b|\bService\b', code, re.IGNORECASE):
            patterns.append('service_layer')
        if re.search(r'\bmvc\b|\bMVC\b|\bcontroller\b', code, re.IGNORECASE):
            patterns.append('mvc_pattern')
        
        # Programming paradigms
        if re.search(r'\.map\(|\.filter\(|\.reduce\(|lambda\s+|=>', code):
            patterns.append('functional_programming')
        if re.search(r'\bclass\s+\w+.*:|\bclass\s+\w+.*{', code):
            patterns.append('object_oriented')
        if re.search(r'\bgeneric\b|\btemplate\b|<.*>', code):
            patterns.append('generic_programming')
        
        # Concurrency patterns
        if re.search(r'\bthread\b|\bTask\b|\basync\b|\bawait\b|\bgoroutine\b', code, re.IGNORECASE):
            patterns.append('concurrency')
        
        # Data structures
        if re.search(r'\blist\b|\barray\b|\bvector\b|\bList\b', code, re.IGNORECASE):
            patterns.append('data_structures')
        if re.search(r'\bmap\b|\bdict\b|\bhash\b|\bHashMap\b', code, re.IGNORECASE):
            patterns.append('hash_maps')
        
        # Testing patterns
        if re.search(r'\btest\b|\bTest\b|\bassert\b|\bmock\b', code, re.IGNORECASE):
            patterns.append('testing')
        
        return patterns
    
    def process_language_samples(self, language: str) -> List[Dict[str, Any]]:
        """Process all code samples for a specific language"""
        language_path = self.code_samples_path / language
        
        if not language_path.exists():
            print(f"No samples found for language: {language}")
            return []
        
        samples = []
        
        for file_path in language_path.glob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Extract features
                    features = self.extract_code_features(code, language)
                    
                    # Create sample record
                    sample = {
                        'id': hashlib.md5(f"{language}:{file_path.name}:{code}".encode()).hexdigest(),
                        'file_name': file_path.name,
                        'language': language,
                        'code': code,
                        'code_snippet': code[:500] + "..." if len(code) > 500 else code,
                        'features': features,
                        'metadata': {
                            'file_size': len(code),
                            'processed_at': datetime.utcnow().isoformat(),
                            'source_file': str(file_path.relative_to(self.base_path))
                        }
                    }
                    
                    samples.append(sample)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return samples
    
    def create_training_datasets(self) -> Dict[str, str]:
        """Create comprehensive training datasets from all code samples"""
        all_samples = []
        languages = ['python', 'javascript', 'typescript', 'java', 'csharp', 'cpp', 'rust', 'go', 'php']
        
        # Process each language
        for language in languages:
            print(f"Processing {language} samples...")
            samples = self.process_language_samples(language)
            all_samples.extend(samples)
            print(f"  Processed {len(samples)} {language} samples")
        
        # Create different dataset formats
        datasets_created = {}
        
        # 1. Complete dataset with all features
        complete_dataset = pd.DataFrame(all_samples)
        complete_path = self.processed_path / "complete_code_dataset.json"
        complete_dataset.to_json(complete_path, orient='records', indent=2)
        datasets_created['complete'] = str(complete_path)
        
        # 2. Feature matrix for ML training
        feature_matrix = []
        for sample in all_samples:
            row = {
                'id': sample['id'],
                'language': sample['language'],
                'file_name': sample['file_name'],
                **sample['features']
            }
            feature_matrix.append(row)
        
        feature_df = pd.DataFrame(feature_matrix)
        feature_path = self.processed_path / "code_features_dataset.csv"
        feature_df.to_csv(feature_path, index=False)
        datasets_created['features'] = str(feature_path)
        
        # 3. Code snippets for language modeling
        snippets_dataset = []
        for sample in all_samples:
            snippets_dataset.append({
                'id': sample['id'],
                'language': sample['language'],
                'text': sample['code_snippet'],
                'full_code': sample['code'],
                'complexity': sample['features']['complexity_score'],
                'patterns': sample['features']['patterns']
            })
        
        snippets_df = pd.DataFrame(snippets_dataset)
        snippets_path = self.processed_path / "code_snippets_dataset.json"
        snippets_df.to_json(snippets_path, orient='records', indent=2)
        datasets_created['snippets'] = str(snippets_path)
        
        # 4. Pattern-based classification dataset
        pattern_dataset = []
        for sample in all_samples:
            for pattern in sample['features']['patterns']:
                pattern_dataset.append({
                    'id': sample['id'],
                    'language': sample['language'],
                    'pattern': pattern,
                    'code_snippet': sample['code_snippet'],
                    'has_classes': sample['features']['has_classes'],
                    'has_functions': sample['features']['has_functions'],
                    'complexity': sample['features']['complexity_score']
                })
        
        if pattern_dataset:
            pattern_df = pd.DataFrame(pattern_dataset)
            pattern_path = self.processed_path / "pattern_classification_dataset.csv"
            pattern_df.to_csv(pattern_path, index=False)
            datasets_created['patterns'] = str(pattern_path)
        
        # 5. Language detection dataset
        lang_detection_dataset = []
        for sample in all_samples:
            # Create multiple snippets of different sizes for training
            code = sample['code']
            snippet_sizes = [100, 200, 500, 1000]
            
            for size in snippet_sizes:
                if len(code) > size:
                    snippet = code[:size]
                    lang_detection_dataset.append({
                        'text': snippet,
                        'language': sample['language'],
                        'snippet_size': size,
                        'features': {
                            'length': len(snippet),
                            'lines': len(snippet.split('\n')),
                            'has_brackets': '{' in snippet or '[' in snippet,
                            'has_semicolons': ';' in snippet,
                            'has_indentation': snippet.startswith('    ') or snippet.startswith('\t')
                        }
                    })
        
        if lang_detection_dataset:
            lang_df = pd.DataFrame(lang_detection_dataset)
            lang_path = self.processed_path / "language_detection_dataset.json"
            lang_df.to_json(lang_path, orient='records', indent=2)
            datasets_created['language_detection'] = str(lang_path)
        
        # 6. Code quality metrics dataset
        quality_dataset = []
        for sample in all_samples:
            quality_score = self._calculate_quality_score(sample)
            quality_dataset.append({
                'id': sample['id'],
                'language': sample['language'],
                'quality_score': quality_score,
                'complexity': sample['features']['complexity_score'],
                'has_comments': sample['features']['has_comments'],
                'has_error_handling': sample['features']['has_error_handling'],
                'code_length': sample['features']['length'],
                'patterns_count': len(sample['features']['patterns'])
            })
        
        quality_df = pd.DataFrame(quality_dataset)
        quality_path = self.processed_path / "code_quality_dataset.csv"
        quality_df.to_csv(quality_path, index=False)
        datasets_created['quality'] = str(quality_path)
        
        return datasets_created
    
    def _calculate_quality_score(self, sample: Dict[str, Any]) -> float:
        """Calculate a quality score for the code sample"""
        features = sample['features']
        score = 0.0
        
        # Base score
        score += 50.0
        
        # Comments boost quality
        if features['has_comments']:
            score += 15.0
        
        # Error handling boost
        if features['has_error_handling']:
            score += 15.0
        
        # Function organization
        if features['has_functions']:
            score += 10.0
        
        # OOP structure
        if features['has_classes']:
            score += 10.0
        
        # Reasonable complexity (not too simple, not too complex)
        complexity = features['complexity_score']
        if 5 <= complexity <= 20:
            score += 10.0
        elif complexity > 50:
            score -= 20.0
        
        # Length penalty for very long files
        if features['length'] > 10000:
            score -= 10.0
        
        # Pattern recognition bonus
        score += len(features['patterns']) * 2.0
        
        return min(100.0, max(0.0, score))
    
    def generate_summary_report(self, datasets_created: Dict[str, str]) -> str:
        """Generate a summary report of the processed datasets"""
        report_path = self.processed_path / "processing_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Code Datasets Processing Report\n\n")
            f.write(f"Generated on: {datetime.utcnow().isoformat()}\n\n")
            
            f.write("## Datasets Created\n\n")
            for dataset_name, dataset_path in datasets_created.items():
                f.write(f"- **{dataset_name.title()}**: `{dataset_path}`\n")
            
            f.write("\n## Dataset Descriptions\n\n")
            
            f.write("### Complete Code Dataset\n")
            f.write("Contains all code samples with full feature extraction and metadata.\n")
            f.write("Format: JSON with nested structures\n")
            f.write("Use case: Comprehensive analysis and feature exploration\n\n")
            
            f.write("### Code Features Dataset\n")
            f.write("Flattened feature matrix suitable for machine learning models.\n")
            f.write("Format: CSV with numeric and categorical features\n")
            f.write("Use case: Classification, clustering, and statistical analysis\n\n")
            
            f.write("### Code Snippets Dataset\n")
            f.write("Code snippets with truncated text for language modeling.\n")
            f.write("Format: JSON with text and metadata\n")
            f.write("Use case: Training language models and code completion\n\n")
            
            f.write("### Pattern Classification Dataset\n")
            f.write("Pattern-based samples for design pattern recognition.\n")
            f.write("Format: CSV with pattern labels\n")
            f.write("Use case: Training pattern detection models\n\n")
            
            f.write("### Language Detection Dataset\n")
            f.write("Code snippets of various sizes for language identification.\n")
            f.write("Format: JSON with text and language labels\n")
            f.write("Use case: Training programming language classifiers\n\n")
            
            f.write("### Code Quality Dataset\n")
            f.write("Quality metrics and scores for code assessment.\n")
            f.write("Format: CSV with quality scores and metrics\n")
            f.write("Use case: Training code quality evaluation models\n\n")
            
            f.write("## Statistics\n\n")
            
            # Load and analyze the complete dataset for statistics
            if 'complete' in datasets_created:
                try:
                    import pandas as pd
                    df = pd.read_json(datasets_created['complete'])
                    
                    f.write(f"- Total samples: {len(df)}\n")
                    f.write(f"- Languages covered: {df['language'].nunique()}\n")
                    f.write(f"- Average code length: {df['features'].apply(lambda x: x['length']).mean():.0f} characters\n")
                    f.write(f"- Samples with classes: {df['features'].apply(lambda x: x['has_classes']).sum()}\n")
                    f.write(f"- Samples with error handling: {df['features'].apply(lambda x: x['has_error_handling']).sum()}\n")
                    
                    f.write("\n### Language Distribution\n\n")
                    lang_counts = df['language'].value_counts()
                    for lang, count in lang_counts.items():
                        f.write(f"- {lang}: {count} samples\n")
                    
                except Exception as e:
                    f.write(f"Error generating statistics: {e}\n")
        
        return str(report_path)


def main():
    """Main processing function"""
    print("Starting code dataset processing...")
    
    processor = CodeDataProcessor()
    datasets_created = processor.create_training_datasets()
    
    print(f"\nDatasets created:")
    for name, path in datasets_created.items():
        print(f"- {name}: {path}")
    
    # Generate summary report
    report_path = processor.generate_summary_report(datasets_created)
    print(f"\nSummary report: {report_path}")
    
    print("\nCode dataset processing completed!")


if __name__ == "__main__":
    main()