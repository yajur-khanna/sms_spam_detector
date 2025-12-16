# Spam Detection with Classical Machine Learning

This project explores **spam vs ham text classification** using classical machine learning models, with an emphasis on **feature engineering, model comparison, and evaluation trade-offs** rather than deep learning.

The goal is not to build a production system, but to understand *why* simple models often perform competitively on structured NLP tasks like spam detection.

---

## Problem Definition

Given a labeled dataset of text messages, the task is to classify each message as:

- **Spam** — unsolicited or promotional content  
- **Ham** — legitimate, non-spam messages  

This is framed as a **binary supervised learning problem** with imbalanced classes.

---

## Design Choices & Rationale

### Why Classical ML (not deep learning)?
- Spam datasets are typically **small to medium-sized**
- Feature-based models (e.g. TF-IDF + Naive Bayes) are:
  - faster to train
  - easier to interpret
  - surprisingly strong baselines
- Deep models add complexity without guaranteed gains here

### Text Representation
- **Bag of Words** and **TF-IDF** are used
- TF-IDF reduces the dominance of frequent but uninformative tokens
- N-grams can capture short spam-specific phrases

### Models Considered
- Multinomial Naive Bayes
- Logistic Regression

These models were chosen for:
- interpretability
- well-understood behavior in text classification
- strong historical performance on spam datasets

---

## Pipeline Overview

1. **Data Cleaning**
   - Lowercasing
   - Punctuation removal
   - Stopword filtering
   - Tokenization

2. **Feature Extraction**
   - Count Vectorizer (BoW)
   - TF-IDF Vectorizer

3. **Model Training**
   - Train/test split
   - Classifier fitting

4. **Evaluation**
   - Accuracy (not relied on alone)
   - Precision, Recall, F1-score
   - Confusion Matrix

---

## Results & Observations

- TF-IDF consistently outperforms raw Bag-of-Words
- Naive Bayes performs strongly despite its simplicity
- Precision is more critical than raw accuracy in spam detection
- Most errors occur on borderline promotional messages

---

## Limitations

- No hyperparameter optimization
- Dataset-specific results (not cross-domain validated)
- No deployment or real-time inference

These are **intentional omissions**, not oversights.

---

## Possible Extensions

- Hyperparameter tuning (GridSearch / Optuna)
- Character-level features
- Ensemble models
- Transformer baselines for comparison
- API or web-based inference

---

## License

MIT License
