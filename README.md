# Natural Language Technology (NLP)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=flat-square&logo=spacy&logoColor=white)](https://spacy.io)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154f3c?style=flat-square)](https://nltk.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](.)

Comprehensive Natural Language Processing coursework covering text classification, named entity recognition, part-of-speech tagging, word embeddings, and practical NLP applications using state-of-the-art libraries.

## 📋 Project Overview

This repository contains practical implementations and coursework from Natural Language Technology (CSC1110), demonstrating proficiency in modern NLP techniques, frameworks, and applications.

**Course Code**: CSC1110
**Topics**: Text Classification, NER, POS Tagging, Word Embeddings, spaCy, NLTK
**Status**: ✅ Completed with Multiple Assignments

## 🎯 Learning Objectives

- Master fundamental and advanced NLP concepts
- Implement text classification pipelines
- Apply named entity recognition (NER) techniques
- Understand and use word embeddings
- Work with industry-standard NLP libraries (spaCy, NLTK)
- Develop practical NLP applications

## 📂 Repository Structure

```
Natural Language Tech/
├── Labs/
│   ├── Lab2.py                                      # Text classification script
│   ├── Lab3.ipynb                                   # Lab 3 exercises
│   ├── Lab_5_word_embeddings/
│   │   └── word_embeddings.ipynb                    # Word2Vec, GloVe, etc.
│   └── Solutions/
│       ├── Lab2_Solutions.zip
│       ├── Lab3_Solutions.zip
│       ├── Lab4_Solutions.zip
│       └── Lab5_Solutions.zip
│
├── CSC1110_Lab2_Text_Classification.ipynb           # Main text classification project
│
├── spaCy Tutorials/
│   ├── spacy_ner.ipynb                              # Named Entity Recognition
│   └── spacy_pos.ipynb                              # Part-of-Speech tagging
│
├── Lectures/
│   └── [Course lecture materials]
│
├── Past Exams/
│   └── [Previous exam papers]
│
├── Assignment/
│   ├── Assignment_Submission.pdf                    # Final submission (25MB)
│   └── assignment_materials/
│
├── Data/
│   └── AcademyAwardsWikipedia.txt                   # Sample text data
│
└── Reference/
    └── [NLP concept diagrams and images]
```

## 🚀 Key Projects & Labs

### Lab 2: Text Classification
A comprehensive text classification project using machine learning.

**Topics Covered**:
- Text preprocessing (tokenization, lemmatization, stopword removal)
- Feature extraction (TF-IDF, Bag-of-Words)
- Classification algorithms (Naive Bayes, SVM, Logistic Regression)
- Model evaluation and performance metrics

**Files**:
- `Lab2.py` - Script implementation
- `CSC1110_Lab2_Text_Classification.ipynb` - Notebook version

**Techniques**:
```python
- Tokenization with NLTK/spaCy
- TF-IDF vectorization
- Scikit-learn classifiers
- Cross-validation
- Confusion matrix analysis
```

### Lab 3: Advanced NLP Tasks
Further exploration of NLP concepts and applications.

**Files**: `Lab3.ipynb`

### Lab 5: Word Embeddings
Deep dive into vector representations of words.

**Topics**:
- Word2Vec (CBOW and Skip-gram)
- GloVe embeddings
- FastText
- Embedding visualization
- Semantic similarity
- Analogy tasks (king - man + woman = queen)

**Files**: `Lab_5_word_embeddings/word_embeddings.ipynb`

**Key Concepts**:
- Distributed representations
- Cosine similarity
- Embedding space visualization (t-SNE, PCA)
- Pre-trained embeddings
- Fine-tuning for domain-specific tasks

### spaCy Named Entity Recognition (NER)
Practical implementation of NER using spaCy's industrial-strength pipeline.

**Features**:
- Pre-trained NER models
- Custom entity recognition
- Entity visualization
- Multi-language support
- Real-world application examples

**Entities Recognized**:
- PERSON, ORG, GPE (Geopolitical Entity)
- DATE, TIME, MONEY
- PRODUCT, EVENT, LOC
- And more...

**Files**: `spaCy Tutorials/spacy_ner.ipynb`

**Example Applications**:
- Information extraction from news articles
- Resume parsing
- Social media analysis
- Document categorization

### spaCy Part-of-Speech (POS) Tagging
Grammatical analysis using spaCy's POS tagging capabilities.

**Features**:
- Universal POS tags
- Detailed tag descriptions
- Dependency parsing
- Linguistic pattern extraction
- Grammar analysis

**Applications**:
- Syntax analysis
- Grammar checking
- Information extraction
- Text simplification

**Files**: `spaCy Tutorials/spacy_pos.ipynb`

## 🛠️ Technologies & Libraries

### Core NLP Libraries
```python
spaCy==3.5.0           # Industrial-strength NLP
nltk==3.8.0            # Natural Language Toolkit
scikit-learn==1.3.0    # Machine learning
gensim==4.3.0          # Topic modeling and word embeddings
```

### Supporting Libraries
```python
pandas==2.0.0          # Data manipulation
numpy==1.24.0          # Numerical operations
matplotlib==3.7.0      # Visualization
seaborn==0.12.0        # Statistical plots
wordcloud==1.9.0       # Word cloud generation
```

### Data Processing
```python
re                     # Regular expressions
string                 # String operations
collections            # Data structures
```

## 📊 Key Concepts Covered

### Text Preprocessing
- Tokenization (word-level, sentence-level, subword)
- Lowercasing and normalization
- Stopword removal
- Stemming (Porter, Lancaster)
- Lemmatization
- Punctuation handling
- Special character cleaning

### Feature Representation
- Bag-of-Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-grams (unigrams, bigrams, trigrams)
- Word embeddings (Word2Vec, GloVe, FastText)
- Contextualized embeddings (concepts)

### Classification Tasks
- Sentiment analysis
- Topic classification
- Spam detection
- Language identification
- Intent recognition

### Information Extraction
- Named Entity Recognition (NER)
- Relationship extraction
- Event extraction
- Coreference resolution (concepts)

### Linguistic Analysis
- Part-of-Speech (POS) tagging
- Dependency parsing
- Constituency parsing
- Morphological analysis

### Evaluation Metrics
- Precision, Recall, F1-Score
- Accuracy
- Confusion Matrix
- ROC-AUC
- Cross-validation

## 🎓 Assignments & Deliverables

### Major Assignment
**File**: `Assignment/Assignment_Submission.pdf` (25MB)

**Scope**: Comprehensive NLP project covering multiple techniques

**Components**:
- Literature review of NLP methods
- Dataset selection and preprocessing
- Model implementation and training
- Evaluation and comparison
- Results analysis and discussion
- Technical documentation

**Grade**: [To be added]

### Lab Assignments
- ✅ Lab 2: Text Classification
- ✅ Lab 3: Advanced NLP Tasks
- ✅ Lab 4: [Topic]
- ✅ Lab 5: Word Embeddings

All labs include:
- Implementation code
- Analysis and interpretation
- Written explanations
- Performance evaluation

## 📈 Practical Applications

### Text Classification Examples
- **Sentiment Analysis**: Movie reviews, product feedback
- **Spam Detection**: Email filtering
- **Topic Classification**: News categorization
- **Intent Recognition**: Chatbot development

### Named Entity Recognition Examples
- **Information Extraction**: Extracting names, dates, organizations from documents
- **Resume Parsing**: Automated candidate screening
- **Social Media Analysis**: Trend detection and monitoring
- **Knowledge Graph Construction**: Building structured data from text

### Word Embeddings Examples
- **Semantic Search**: Finding similar documents
- **Recommendation Systems**: Content-based filtering
- **Machine Translation**: Cross-lingual embeddings
- **Text Generation**: Language modeling

## 🔧 Setup & Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
Jupyter Notebook/Lab
```

### Installation Steps

1. **Navigate to project directory**
```bash
cd "D:\Final Year\Natural Language Tech"
```

2. **Install core NLP libraries**
```bash
pip install spacy nltk scikit-learn gensim
pip install pandas numpy matplotlib seaborn
pip install jupyter notebook
```

3. **Download spaCy models**
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

5. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

### Running the Labs

**Lab 2 - Text Classification**:
```bash
# Script version
python Lab2.py

# Or open notebook
jupyter notebook CSC1110_Lab2_Text_Classification.ipynb
```

**spaCy NER Tutorial**:
```bash
jupyter notebook "spaCy Tutorials/spacy_ner.ipynb"
```

**Word Embeddings Lab**:
```bash
jupyter notebook "Lab_5_word_embeddings/word_embeddings.ipynb"
```

## 📚 Learning Resources

### Course Materials
- Lecture slides and notes
- Past exam papers with solutions
- Lab solution sets
- Reference diagrams and visualizations

### Recommended Reading
- Jurafsky & Martin: "Speech and Language Processing"
- Manning & Schütze: "Foundations of Statistical NLP"
- Bird, Klein & Loper: "Natural Language Processing with Python" (NLTK book)
- spaCy documentation and tutorials
- Mikolov et al.: Word2Vec papers

### Online Resources
- spaCy documentation: https://spacy.io
- NLTK documentation: https://www.nltk.org
- Hugging Face tutorials
- Papers With Code NLP section

## 🎯 Skills Demonstrated

### Technical Skills
- **NLP Frameworks**: spaCy, NLTK, scikit-learn
- **Text Processing**: Tokenization, normalization, cleaning
- **Feature Engineering**: TF-IDF, word embeddings
- **Classification**: ML algorithms for text
- **Information Extraction**: NER, POS tagging
- **Python Programming**: OOP, functional programming
- **Data Visualization**: Results presentation

### Analytical Skills
- Text data analysis
- Model evaluation and comparison
- Performance optimization
- Error analysis
- Critical interpretation of results

### Professional Skills
- Technical documentation
- Scientific writing
- Problem-solving
- Research methodology
- Code organization and best practices

## 🔍 Advanced Topics Explored

### Beyond Basics
- Transfer learning in NLP
- Attention mechanisms (concepts)
- Transformer architecture (concepts)
- BERT and contextual embeddings (concepts)
- Multi-task learning
- Zero-shot and few-shot learning (concepts)

### Practical Considerations
- Handling imbalanced datasets
- Dealing with out-of-vocabulary words
- Domain adaptation
- Multilingual NLP challenges
- Computational efficiency
- Model interpretability

## 🚀 Future Directions

### Potential Extensions
- Deep learning for NLP (LSTM, GRU, Transformers)
- Hugging Face Transformers integration
- Question answering systems
- Text summarization
- Machine translation
- Dialogue systems and chatbots
- Multilingual applications

### Production Deployment
- REST API for NLP models
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Model serving and optimization
- A/B testing for NLP systems

## 📊 Project Statistics

| Metric | Count |
|--------|-------|
| Lab Assignments | 5 |
| Jupyter Notebooks | 7+ |
| Python Scripts | 2+ |
| NLP Techniques | 15+ |
| Libraries Used | 8+ |
| Datasets Analyzed | 5+ |

## 📝 Exam Preparation

### Key Topics for Exams
- Text preprocessing pipeline
- Feature extraction methods
- Classification algorithms
- NER and POS tagging
- Word embeddings theory
- Evaluation metrics
- spaCy and NLTK usage

### Past Exams
Available in `Past Exams/` folder for practice and review.

## 📄 License

Academic coursework - All rights reserved

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- spaCy and NLTK development teams
- Open-source NLP community
- Research paper authors
- Dataset providers

---

**Course**: CSC1110 - Natural Language Technology
**Academic Year**: 2025-2026
**Topics**: NLP, Text Classification, NER, POS, Word Embeddings
**Last Updated**: June 2026

**Note**: This repository represents comprehensive coursework in modern Natural Language Processing, demonstrating both theoretical understanding and practical implementation skills essential for NLP engineering roles.
