# Sample Datasets for ML/AI Agent Training

## Overview

This directory contains comprehensive, structured sample datasets specifically designed for training AI coding agents and machine learning models. These datasets are **not production data** but carefully curated training examples that demonstrate various ML patterns, structures, and use cases.

## ⚠️ Important Notice

**These are sample training datasets for educational and AI training purposes only. Do not use this data for real-time production systems or make business decisions based on this sample data.**

## Dataset Categories

### 1. Code Analysis (`code_analysis/`)

**File**: `programming_patterns_dataset.json`

**Purpose**: Train AI agents to recognize, understand, and generate proper programming patterns while avoiding anti-patterns.

**Contents**:
- Design patterns (Singleton, Factory, Observer, etc.)
- Anti-patterns (God Object, Spaghetti Code, etc.)
- Refactoring examples (before/after code improvements)
- Optimization patterns (performance improvements)
- Security patterns (input validation, secure coding)
- Testing patterns (comprehensive test strategies)

**Use Cases**:
- Code pattern recognition and suggestion
- Automated code review and quality assessment
- Refactoring recommendation systems
- Best practice enforcement tools
- Anti-pattern detection and prevention

**Sample Structure**:
```json
{
  "id": "pattern_001",
  "pattern_type": "design_pattern|anti_pattern|refactoring|optimization|security|testing",
  "language": "python|javascript|java|cpp|typescript|go|rust",
  "complexity": "beginner|intermediate|advanced|expert",
  "code_before": "original_code_example",
  "code_after": "improved_code_example",
  "explanation": "detailed_explanation",
  "benefits": ["list_of_improvements"],
  "common_mistakes": ["typical_errors"]
}
```

---

### 2. Natural Language Processing (`nlp/`)

**File**: `nlp_training_dataset.json`

**Purpose**: Train AI models on various NLP tasks including sentiment analysis, entity recognition, text classification, and language generation.

**Contents**:
- Sentiment analysis (product reviews, customer feedback)
- Named Entity Recognition (person, location, organization extraction)
- Text classification (support tickets, content categorization)
- Language generation (email responses, content creation)
- Text summarization (document condensation)
- Question answering (context-based information retrieval)
- Intent classification (chatbot training)
- Multi-language analysis (language detection, translation)

**Use Cases**:
- Customer sentiment monitoring
- Content moderation systems
- Automated customer support
- Document processing and analysis
- Language translation services
- Information extraction systems

**Sample Structure**:
```json
{
  "id": "nlp_001",
  "task_type": "sentiment_analysis|ner|classification|generation|summarization",
  "text": "input_text_content",
  "labels": "ground_truth_annotations",
  "language": "target_language",
  "complexity": "task_difficulty_level",
  "domain": "application_area"
}
```

---

### 3. Computer Vision (`computer_vision/`)

**File**: `computer_vision_dataset.json`

**Purpose**: Train AI models on visual recognition, object detection, image segmentation, and visual analysis tasks.

**Contents**:
- Image classification (objects, scenes, activities)
- Object detection (bounding box identification)
- Semantic segmentation (pixel-level classification)
- Face recognition and analysis
- Optical Character Recognition (OCR)
- Visual Question Answering
- Image generation and style transfer

**Use Cases**:
- Autonomous vehicle perception
- Medical image analysis
- Security and surveillance systems
- Content moderation
- Manufacturing quality control
- Augmented reality applications

**Sample Structure**:
```json
{
  "id": "cv_001",
  "task_type": "classification|detection|segmentation|recognition|generation",
  "image_metadata": "image_properties_and_description",
  "annotations": "labels_bounding_boxes_masks",
  "complexity": "task_difficulty",
  "domain": "application_area"
}
```

---

### 4. Time Series Analysis (`time_series/`)

**File**: `time_series_dataset.json`

**Purpose**: Train AI models on temporal data analysis including forecasting, anomaly detection, and pattern recognition.

**Contents**:
- Forecasting (energy consumption, stock prices, demand prediction)
- Anomaly detection (system monitoring, fraud detection)
- Pattern classification (trend analysis, seasonal patterns)
- Seasonality detection (periodic behavior identification)
- Change point detection (regime changes)
- Correlation analysis (multi-variate relationships)

**Use Cases**:
- Business forecasting and planning
- System monitoring and alerting
- Financial market analysis
- IoT sensor data analysis
- Supply chain optimization
- Predictive maintenance

**Sample Structure**:
```json
{
  "id": "ts_001",
  "task_type": "forecasting|anomaly_detection|pattern_classification|seasonality",
  "time_series_data": "temporal_data_with_timestamps",
  "labels": "predictions_anomalies_patterns",
  "metadata": "data_characteristics",
  "complexity": "task_difficulty"
}
```

---

### 5. Recommendation Systems (`recommendation/`)

**File**: `recommendation_dataset.json`

**Purpose**: Train AI models on personalization and recommendation tasks including collaborative filtering and content-based approaches.

**Contents**:
- User-item rating prediction
- Top-K recommendation generation
- Session-based recommendations
- Cold start problem handling
- Cross-domain recommendations
- Explainable recommendations

**Use Cases**:
- E-commerce product recommendations
- Content streaming suggestions
- Social media feed curation
- Job and career recommendations
- Learning path suggestions
- Travel and accommodation recommendations

**Sample Structure**:
```json
{
  "id": "rec_001",
  "task_type": "rating_prediction|top_k|session_based|cold_start|explainable",
  "user_data": "user_profiles_and_behavior",
  "item_data": "item_features_and_metadata",
  "interaction_data": "user_item_interactions",
  "recommendations": "generated_suggestions_with_scores"
}
```

---

### 6. Anomaly Detection (`anomaly_detection/`)

**File**: `anomaly_detection_dataset.json`

**Purpose**: Train AI models to identify outliers, unusual patterns, and anomalous behaviors across various domains.

**Contents**:
- Fraud detection (financial transactions)
- Network intrusion detection (cybersecurity)
- Equipment failure prediction (industrial monitoring)
- Quality control (manufacturing)
- Behavioral anomaly detection (user analytics)
- Medical anomaly detection (healthcare)

**Use Cases**:
- Financial fraud prevention
- Cybersecurity threat detection
- Industrial maintenance planning
- Quality assurance systems
- Healthcare monitoring
- System reliability management

**Sample Structure**:
```json
{
  "id": "ad_001",
  "task_type": "fraud_detection|intrusion_detection|failure_prediction|quality_control",
  "normal_data": "baseline_behavior_patterns",
  "anomalous_data": "outliers_and_unusual_patterns",
  "detection_results": "anomaly_scores_and_classifications",
  "ground_truth": "actual_anomaly_labels"
}
```

---

### 7. Multi-Modal Learning (`multi_modal/`)

**File**: `multi_modal_dataset.json`

**Purpose**: Train AI models on tasks involving multiple data modalities (text, images, audio, structured data).

**Contents**:
- Vision-language understanding (image-text alignment)
- Audio-visual analysis (video understanding)
- Cross-modal retrieval (finding related content across modalities)
- Multi-modal sentiment analysis (combining text, images, ratings)
- Visual question answering (reasoning about images using text)

**Use Cases**:
- Social media content analysis
- Video understanding and summarization
- Accessibility applications
- Human-computer interaction
- Content creation and editing
- Educational technology

**Sample Structure**:
```json
{
  "id": "mm_001",
  "task_type": "vision_language|audio_visual|cross_modal_retrieval|multimodal_sentiment",
  "modalities_data": "data_from_multiple_modalities",
  "cross_modal_relationships": "connections_between_modalities",
  "unified_representation": "joint_embeddings_or_features"
}
```

## Dataset Statistics

| Dataset | Samples | Complexity Levels | Languages/Domains | Primary Use Cases |
|---------|---------|-------------------|-------------------|-------------------|
| Code Analysis | 150 | 4 (Basic-Expert) | 7 Languages | Code Quality, Refactoring |
| NLP | 200 | 4 (Basic-Expert) | 5 Languages | Text Analysis, Generation |
| Computer Vision | 180 | 4 (Basic-Expert) | 8 Domains | Visual Recognition, Analysis |
| Time Series | 160 | 4 (Basic-Expert) | 6 Domains | Forecasting, Monitoring |
| Recommendation | 140 | 4 (Basic-Expert) | 5 Domains | Personalization, Suggestions |
| Anomaly Detection | 120 | 4 (Basic-Expert) | 6 Domains | Outlier Detection, Security |
| Multi-Modal | 100 | 4 (Basic-Expert) | 5 Domains | Cross-Modal Understanding |

**Total**: 1,050+ training samples across 7 major ML domains

## Usage Guidelines

### For AI Training

1. **Model Development**: Use these datasets to train and validate ML models
2. **Pattern Learning**: Help AI agents understand common patterns and structures
3. **Evaluation**: Benchmark model performance across different complexity levels
4. **Transfer Learning**: Pre-train models on diverse, high-quality examples

### For Educational Purposes

1. **Learning**: Understand ML concepts through practical examples
2. **Prototyping**: Rapid development and testing of ML applications
3. **Research**: Academic research and experimentation
4. **Tutorials**: Teaching ML concepts with real-world-like data

### Best Practices

1. **Data Validation**: Always validate data quality before training
2. **Complexity Progression**: Start with basic examples, progress to expert level
3. **Cross-Domain Training**: Use multiple datasets for robust model development
4. **Evaluation Metrics**: Use provided evaluation metrics as training targets
5. **Documentation**: Refer to metadata for context and usage guidance

## Data Quality Assurance

- **Structured Format**: All data follows consistent JSON schemas
- **Comprehensive Metadata**: Detailed information about each sample
- **Quality Labels**: Ground truth annotations for supervised learning
- **Complexity Levels**: Graduated difficulty for progressive training
- **Real-World Relevance**: Examples based on actual use cases
- **Multiple Domains**: Coverage across diverse application areas

## Future Extensions

These sample datasets are designed to be:
- **Extensible**: Easy to add new samples and categories
- **Modular**: Each dataset can be used independently
- **Scalable**: Structured for easy expansion and modification
- **Standardized**: Consistent format across all datasets

## Support and Maintenance

For questions about dataset usage, structure, or content:
1. Review the metadata in each dataset file
2. Check the sample structure documentation above
3. Examine the complexity progression from basic to expert
4. Refer to the use case descriptions for context

---

*Last Updated: 2024-01-01*  
*Version: 1.0.0*  
*Total Training Samples: 1,050+*