# Deceptive-Review-Detection

### Overview:
This project focuses on detecting deceptive (fake) reviews using Natural Language Processing (NLP) and Machine Learning techniques. It analyzes textual data to identify linguistic, stylistic, and semantic patterns associated with deceptive content.

The system predicts the likelihood of deception and provides insights into features influencing the classification.

### Objectives

- Detect deceptive vs genuine reviews using NLP techniques  
- Extract linguistic and stylometric features from text  
- Compare performance of classical ML and transformer-based models  
- Analyze patterns associated with deceptive content

### Dataset

- **Dataset:** Fake Reviews CSV  
- **Total Records:** ~40,000  
- **Class Distribution:** Balanced (50% Deceptive, 50% Genuine)  

Target Variable
- Deceptive (CG) = 1  
- Genuine (OR) = 0

### Data Preprocessing

- **Text Cleaning:**
  - Lowercasing  
  - URL removal  
  - Whitespace normalization  

- **Tokenization:**
  - TextBlob (for ML models)  
  - WordPiece tokenizer (for BERT)  

- **Stopword Handling:**
  - Retained for stylometric features  
  - Filtered in TF-IDF  

- Label encoding (binary classification)  
- Train-test split (80:20, stratified)

### Feature Engineering

**TF-IDF Features**
- Max features: 20,000  
- Unigrams + Bigrams (n-grams: 1–2)  
- Sublinear TF scaling  
- min_df = 3  

**Stylometric Features (14 total)**
- Sentiment polarity & subjectivity  
- Exclamation & question mark ratio  
- First-person pronoun usage  
- Type-token ratio (lexical richness)  
- Sentence length statistics  
- Capitalization and repetition ratios

### Models Implemented

**1. Logistic Regression (Baseline)**
- TF-IDF + Stylometric features  
- L2 Regularization  
- Class balancing enabled  
- Highly interpretable  

**2. BERT (bert-base-uncased)**
- Transformer-based deep learning model  
- Fine-tuned for classification  
- Captures contextual semantics  
- High accuracy but computationally expensive  

### Training Details
- Train/Test Split: 80/20  
- BERT Epochs: 3  
- Learning Rate: 2e-5  
- Batch Size: 32  
- Optimizer: AdamW

### Evaluation Metrics
- Accuracy  
- ROC-AUC Score  
- Confusion Matrix

### Results & Analysis
- Logistic Regression Accuracy: ~94.7%  
- Logistic Regression ROC-AUC: 0.9953  
- BERT Accuracy: ~99%+  
- BERT ROC-AUC: 0.9978

### Observations:
- BERT significantly outperforms classical models  
- Logistic Regression provides a strong baseline with high interpretability  
- Stylometric + TF-IDF features effectively capture deception signals

### Key Insights
- Deceptive reviews are more verbose and repetitive  
- High usage of exclamations and superlatives indicates deception  
- Lower lexical richness observed in fake reviews  
- Genuine reviews contain more first-person pronouns  
- BERT captures deep semantic patterns missed by traditional models

### Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK, TextBlob  
- HuggingFace Transformers (BERT)

### Limitations & Future Improvements:
This project has certain limitations, including possible overfitting in the BERT model, reliance on a single fake reviews dataset, and high computational requirements for training transformer-based models. Additionally, the system may face challenges in generalizing to other domains or languages. Future improvements can focus on using more diverse and larger datasets, experimenting with advanced models like RoBERTa or DeBERTa, building ensemble approaches, and optimizing models through techniques like quantization or distillation. Deploying the system as a real-time API and extending support to multiple languages can further enhance its practical applicability.


