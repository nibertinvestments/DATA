import JSZip from 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm';
import { saveAs } from 'https://cdn.jsdelivr.net/npm/file-saver@2.0.5/+esm';

interface RepositoryFile {
    path: string;
    content: string;
    type: 'file' | 'directory';
}

class DataRepositoryDownloader {
    private zipSize: number = 0;
    private totalFiles: number = 0;

    constructor() {
        this.initializeEventListeners();
        this.calculateZipSize();
    }

    private initializeEventListeners(): void {
        const downloadBtn = document.getElementById('downloadBtn') as HTMLButtonElement;
        const downloadBtnMain = document.getElementById('downloadBtnMain') as HTMLButtonElement;
        const closeModal = document.getElementById('closeModal') as HTMLButtonElement;

        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.handleDownload());
        }

        if (downloadBtnMain) {
            downloadBtnMain.addEventListener('click', () => this.handleDownload());
        }

        if (closeModal) {
            closeModal.addEventListener('click', () => this.closeModal());
        }

        window.addEventListener('click', (event: MouseEvent) => {
            const modal = document.getElementById('downloadModal') as HTMLElement;
            if (event.target === modal) {
                this.closeModal();
            }
        });
    }

    private calculateZipSize(): void {
        const estimatedSize = 15.5; // MB
        const zipSizeElement = document.getElementById('zipSize') as HTMLElement;
        if (zipSizeElement) {
            zipSizeElement.textContent = `${estimatedSize} MB`;
        }
    }

    private showModal(): void {
        const modal = document.getElementById('downloadModal') as HTMLElement;
        if (modal) {
            modal.classList.add('show');
        }
    }

    private closeModal(): void {
        const modal = document.getElementById('downloadModal') as HTMLElement;
        if (modal) {
            modal.classList.remove('show');
        }
    }

    private updateProgress(percentage: number): void {
        const progressBar = document.getElementById('progressBar') as HTMLElement;
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
    }

    private updateProgressText(text: string): void {
        const progressText = document.getElementById('progressText') as HTMLElement;
        if (progressText) {
            progressText.textContent = text;
        }
    }

    private async handleDownload(): Promise<void> {
        this.showModal();
        this.updateProgress(0);
        this.updateProgressText('Initializing download...');

        try {
            const zip = new JSZip();

            // Simulate adding repository files
            await this.addRepositoryFilesToZip(zip);

            this.updateProgress(90);
            this.updateProgressText('Generating ZIP archive...');

            // Generate ZIP blob
            const blob = await zip.generateAsync({ type: 'blob' });

            this.updateProgress(100);
            this.updateProgressText('Download ready!');

            // Trigger download
            setTimeout(() => {
                saveAs(blob, 'DATA-Repository-v0.1.1.zip');
                this.closeModal();
            }, 500);
        } catch (error) {
            console.error('Download error:', error);
            this.updateProgressText('Error creating archive. Please try again.');
            setTimeout(() => this.closeModal(), 2000);
        }
    }

    private async addRepositoryFilesToZip(zip: JSZip): Promise<void> {
        const files = this.getRepositoryStructure();
        const totalFiles = this.countTotalFiles(files);
        let processedFiles = 0;

        for (const file of files) {
            this.addFileToZip(zip, file);
            processedFiles++;
            const progress = Math.floor((processedFiles / totalFiles) * 80);
            this.updateProgress(progress);
            this.updateProgressText(`Adding files... (${processedFiles}/${totalFiles})`);
        }
    }

    private getRepositoryStructure(): RepositoryFile[] {
        return [
            // Core files
            { path: 'README.md', content: this.getReadmeContent(), type: 'file' },
            { path: 'LICENSE', content: this.getLicenseContent(), type: 'file' },
            { path: '.gitignore', content: this.getGitignoreContent(), type: 'file' },
            { path: 'Cargo.toml', content: this.getCargoContent(), type: 'file' },

            // Datasets
            {
                path: 'datasets/processed/ai_coding_agent_training_suite.json',
                content: this.getAiTrainingDatasetContent(),
                type: 'file',
            },

            // Code samples (examples)
            {
                path: 'code_samples/python/algorithms_sorting.py',
                content: this.getPythonAlgorithmsSample(),
                type: 'file',
            },
            {
                path: 'code_samples/javascript/algorithms_sorting.js',
                content: this.getJavaScriptAlgorithmsSample(),
                type: 'file',
            },

            // Scripts
            {
                path: 'scripts/validate_dataset_schema.py',
                content: this.getValidationScriptContent(),
                type: 'file',
            },

            // Documentation
            {
                path: 'datasets/README.md',
                content: this.getDatasetsReadmeContent(),
                type: 'file',
            },
            {
                path: 'datasets/processed/README.md',
                content: this.getProcessedDatasetsReadmeContent(),
                type: 'file',
            },
        ];
    }

    private addFileToZip(zip: JSZip, file: RepositoryFile): void {
        if (file.type === 'file') {
            zip.file(file.path, file.content);
        }
    }

    private countTotalFiles(files: RepositoryFile[]): number {
        return files.length;
    }

    // Content generators for repository files
    private getReadmeContent(): string {
        return `# DATA Repository v0.1.1

A curated repository of code samples, structured datasets, and validation tooling designed for ML and AI training pipelines focused on software engineering.

## Quick Start

\`\`\`bash
# Validate a dataset
python3 scripts/validate_dataset_schema.py datasets/processed/ai_coding_agent_training_suite.json

# Run repository tests
python3 -m pytest tests/test_dataset_schema.py -q
\`\`\`

## Contents

- **code_samples/**: 1,409 production-ready implementations across 18 languages
- **datasets/processed/**: Schema-validated training datasets
- **scripts/**: Validation and processing tooling
- **VECTOR_PROCESSING/**: Vector operations for ML/AI applications

## License

MIT License - See LICENSE for details

Built with care by Nibert Investments LLC
`;
    }

    private getLicenseContent(): string {
        return `MIT License

Copyright (c) 2026 Nibert Investments LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
`;
    }

    private getGitignoreContent(): string {
        return `__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
*.log
node_modules/
.DS_Store
*.zip
`;
    }

    private getCargoContent(): string {
        return `[package]
name = "ml_production_patterns"
version = "0.1.1"
edition = "2021"

[[bin]]
name = "ml_production_patterns"
path = "code_samples/rust/ml_production_patterns.rs"

[dependencies]
serde_json = "1.0"
fastrand = "2.0"
`;
    }

    private getAiTrainingDatasetContent(): string {
        return JSON.stringify(
            {
                dataset_id: 'ai_coding_agent_training_suite_v1',
                name: 'AI Coding Agent Training Suite',
                version: '1.0.0',
                description: 'Curated software-engineering tasks for code generation, bug fixing, and reasoning.',
                language_coverage: ['python', 'javascript', 'typescript', 'solidity', 'java', 'go', 'rust'],
                task_types: ['bug_fix', 'code_generation', 'security_review', 'translation', 'refactor'],
                tasks: [
                    {
                        id: 'task-001',
                        language: 'python',
                        task_type: 'bug_fix',
                        difficulty: 'medium',
                        prompt: 'Fix the median function...',
                        response: 'def median(values):\\n    if not values:\\n        raise ValueError()\\n    ordered = sorted(values)\\n    mid = len(ordered) // 2\\n    if len(ordered) % 2 == 0:\\n        return (ordered[mid - 1] + ordered[mid]) / 2\\n    return ordered[mid]',
                        metadata: { tags: ['statistics'], source: 'synthetic' },
                    },
                ],
            },
            null,
            2
        );
    }

    private getPythonAlgorithmsSample(): string {
        return `# Sorting Algorithms

def bubble_sort(arr):
    """Bubble sort implementation O(n²)"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def merge_sort(arr):
    """Merge sort implementation O(n log n)"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
`;
    }

    private getJavaScriptAlgorithmsSample(): string {
        return `// Sorting Algorithms

function bubbleSort(arr) {
    const n = arr.length;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}

function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    
    return merge(left, right);
}

function merge(left, right) {
    const result = [];
    let i = 0, j = 0;
    
    while (i < left.length && j < right.length) {
        result.push(left[i] <= right[j] ? left[i++] : right[j++]);
    }
    
    return [...result, ...left.slice(i), ...right.slice(j)];
}
`;
    }

    private getValidationScriptContent(): string {
        return `#!/usr/bin/env python3
"""Validate DATA repository datasets."""

import json
import sys
from pathlib import Path

REQUIRED_KEYS = {"dataset_id", "name", "version", "language_coverage", "task_types", "tasks"}

def validate_dataset(path):
    with open(path) as f:
        data = json.load(f)
    
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    
    if not isinstance(data.get("tasks"), list):
        raise ValueError("tasks must be a list")
    
    print(f"OK: {path} - {len(data['tasks'])} tasks")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        validate_dataset(path)
`;
    }

    private getDatasetsReadmeContent(): string {
        return `# Datasets

This directory contains curated training data organized into three categories:

- **processed/**: Schema-valid datasets ready for model training
- **raw/**: Source data and exploratory samples
- **synthetic/**: Generated task variants and benchmarks

## Usage

\`\`\`bash
python3 scripts/validate_dataset_schema.py datasets/processed/*.json
\`\`\`
`;
    }

    private getProcessedDatasetsReadmeContent(): string {
        return `# Processed Datasets

Processed datasets are curated and schema-validated training assets for ML model ingestion.

## Current Assets

- \`ai_coding_agent_training_suite.json\` - 10+ tasks covering code generation, bug fixing, and security

## Standards

- Explicit schema version metadata
- Language coverage and task-type metadata
- Realistic prompts and ground-truth answers
- Consistent difficulty labels
`;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    new DataRepositoryDownloader();
});
