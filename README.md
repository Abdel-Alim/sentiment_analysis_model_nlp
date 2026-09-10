# 🎬 IMDB Sentiment Analysis

> **A Machine Learning NLP project for classifying movie reviews as Positive or Negative using TF-IDF feature extraction and classical Machine Learning algorithms.**

---

## 📌 Project Overview

This project builds an end-to-end **Natural Language Processing (NLP) sentiment analysis pipeline** using the **IMDB Movie Reviews dataset**.

The objective is to automatically determine whether a movie review expresses a **positive** or **negative** sentiment.

The project covers the complete workflow:

**Raw Text → Text Cleaning → NLP Preprocessing → TF-IDF Vectorization → Model Training → Model Comparison → Hyperparameter Tuning → Evaluation → Model Saving**

The final selected model is **Logistic Regression**, achieving an accuracy of **89.44%** on the test set.

---

## 🎯 Project Objectives

* Build a complete NLP sentiment classification pipeline.
* Clean and preprocess raw movie reviews.
* Convert text into numerical features using **TF-IDF**.
* Compare multiple Machine Learning algorithms.
* Optimize the best-performing model using **GridSearchCV**.
* Evaluate model performance using multiple classification metrics.
* Save the trained model and TF-IDF vectorizer for future inference.

---

## 📊 Dataset

The project uses the **IMDB Dataset**, containing:

| Property         |  Value |
| ---------------- | -----: |
| Total Reviews    | 50,000 |
| Positive Reviews | 25,000 |
| Negative Reviews | 25,000 |
| Positive Label   |    `1` |
| Negative Label   |    `0` |

The dataset is perfectly balanced, with an equal number of positive and negative reviews.

Each record contains:

* `review` — Original movie review text
* `sentiment` — Original sentiment category
* `label` — Numerical sentiment label

The notebook also analyzes review length before modeling.

---

## 🧠 NLP Preprocessing

The text preprocessing pipeline contains several steps designed to reduce noise and prepare the reviews for machine learning.

### 1. HTML Removal

HTML tags such as:

```text
<br />
```

are removed from the reviews.

### 2. URL Removal

URLs are removed from the text.

### 3. Character Cleaning

Non-alphabetic characters are removed.

### 4. Lowercasing

All text is converted to lowercase.

### 5. Whitespace Normalization

Extra whitespace is removed.

### 6. Stopword Removal

English stopwords are removed while preserving important negation words such as:

```text
not
no
never
nor
```

This is particularly useful for sentiment analysis because negation can significantly change the meaning of a sentence.

### 7. Lemmatization

Words are lemmatized using NLTK's `WordNetLemmatizer`.

The preprocessing implementation is contained in the notebook.

---

## 🔢 Feature Engineering

After preprocessing, the text is transformed into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The vectorizer configuration is:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    min_df=5,
    stop_words='english'
)
```

### Configuration

| Parameter                  | Value              |
| -------------------------- | ------------------ |
| Feature Extraction         | TF-IDF             |
| N-grams                    | Unigrams + Bigrams |
| Maximum Features           | 15,000             |
| Minimum Document Frequency | 5                  |
| Stopwords                  | English            |

Using both **unigrams and bigrams** allows the model to capture individual words as well as short phrases.

For example:

```text
excellent
```

and:

```text
not good
```

can provide different sentiment signals.

The dataset is split into:

* **80% Training**
* **20% Testing**

using stratification to preserve the class distribution.

---

# 🤖 Machine Learning Models

Three classical Machine Learning algorithms were trained and compared.

### 1. Multinomial Naive Bayes

```text
Accuracy: 86.40%
```

### 2. Logistic Regression

```text
Accuracy: 89.44%
```

### 3. Linear Support Vector Machine

```text
Accuracy: 89.13%
```

### Model Comparison

| Model                   |   Accuracy |
| ----------------------- | ---------: |
| Multinomial Naive Bayes |     86.40% |
| Logistic Regression     | **89.44%** |
| Linear SVM              |     89.13% |

Logistic Regression achieved the highest accuracy among the three tested models.

---

# 🏆 Model Optimization

After comparing the initial models, **Logistic Regression** was selected for further optimization.

GridSearchCV was used to search for the best value of the regularization parameter:

```python
C = [0.1, 1, 10]
```

The search used:

```text
5-fold Cross Validation
```

with:

```text
F1 Score
```

as the optimization metric.

### Best Parameters

```text
C = 1
```

The optimized model achieved:

```text
Final Accuracy = 89.44%
```

The optimized model was then saved for future use.

---

# 📈 Model Evaluation

The final Logistic Regression model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| Negative             |      0.90 |   0.88 |     0.89 |
| Positive             |      0.89 |   0.91 |     0.90 |
| **Overall Accuracy** |           |        | **0.89** |

The test set contains **10,000 reviews**, with 5,000 reviews from each class.

### Confusion Matrix

```text
[[4413  587]
 [ 469 4531]]
```

This means:

* **4,413** negative reviews were correctly classified.
* **587** negative reviews were incorrectly classified as positive.
* **469** positive reviews were incorrectly classified as negative.
* **4,531** positive reviews were correctly classified.

---

# 💾 Model Persistence

The trained model and TF-IDF vectorizer are saved using `joblib`.

### Saved Files

```text
sentiment_model.pkl
tfidf_vectorizer.pkl
```

These files allow the trained pipeline components to be reused without retraining the model from scratch.

---

# 🛠️ Technologies & Libraries

### Programming Language

* 🐍 Python

### NLP

* NLTK
* TF-IDF
* Stopword Removal
* Lemmatization
* Text Cleaning

### Machine Learning

* Scikit-learn
* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes
* GridSearchCV

### Data Processing

* Pandas

### Model Persistence

* Joblib

### Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# 📂 Project Structure

```text
IMDB-Sentiment-Analysis/
│
├── sentiment_model.ipynb
│
├── IMDB Dataset.csv
│
├── sentiment_model.pkl
│
├── tfidf_vectorizer.pkl
│
└── README.md
```

> The exact files included in the repository may vary depending on which generated artifacts are committed.

---

# 🔄 Machine Learning Pipeline

```text
                IMDB Dataset
                     │
                     ▼
              Raw Movie Reviews
                     │
                     ▼
             Text Cleaning
                     │
        ┌────────────┴────────────┐
        │                         │
     HTML/URLs              Special Characters
        │                         │
        └────────────┬────────────┘
                     ▼
                Lowercase
                     │
                     ▼
             Stopword Removal
                     │
                     ▼
               Lemmatization
                     │
                     ▼
             TF-IDF Vectorization
                     │
                     ▼
             Train / Test Split
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Naive Bayes  Logistic    Linear SVM
                  Regression
          │          │          │
          └──────────┼──────────┘
                     ▼
              Model Comparison
                     │
                     ▼
           Logistic Regression
                     │
                     ▼
             GridSearchCV
                     │
                     ▼
              Final Model
                     │
                     ▼
            Model Evaluation
                     │
                     ▼
              Model Saving
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/IMDB-Sentiment-Analysis.git
```

Navigate to the project:

```bash
cd IMDB-Sentiment-Analysis
```

Install the required dependencies:

```bash
pip install pandas nltk scikit-learn joblib
```

Download the required NLTK resources:

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
```

---

# ▶️ Running the Project

Open the notebook:

```bash
jupyter notebook sentiment_model.ipynb
```

Then execute the notebook cells sequentially.

The notebook performs:

1. Dataset loading
2. Exploratory analysis
3. Text cleaning
4. NLP preprocessing
5. TF-IDF feature extraction
6. Train/test splitting
7. Model training
8. Model comparison
9. Hyperparameter tuning
10. Model evaluation
11. Model persistence

---

# 💡 Example Use Case

After training, the saved model can be used to classify a new movie review.

Example:

```text
"This movie was absolutely amazing. The story was excellent and the acting was fantastic."
```

Expected prediction:

```text
Positive
```

Another example:

```text
"The movie was boring and disappointing. I would not recommend it."
```

Expected prediction:

```text
Negative
```

> The notebook provided for this project trains and saves the model/vectorizer; a separate inference script is not included in the provided notebook.

---

# 📊 Results Summary

| Metric          |              Result |
| --------------- | ------------------: |
| Dataset Size    |              50,000 |
| Training Data   |                 80% |
| Test Data       |                 20% |
| TF-IDF Features |              15,000 |
| Best Model      | Logistic Regression |
| Best `C`        |                   1 |
| Test Accuracy   |          **89.44%** |
| Negative F1     |                0.89 |
| Positive F1     |                0.90 |

---

# 🎓 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Natural Language Processing
* Text preprocessing
* Feature engineering
* TF-IDF
* N-gram representation
* Binary classification
* Logistic Regression
* Support Vector Machines
* Naive Bayes
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Confusion matrix analysis
* Model serialization

---

# 🔮 Future Improvements

Potential improvements for a future version include:

* [ ] Build a reusable prediction script
* [ ] Create a Streamlit web application
* [ ] Add an interactive sentiment prediction interface
* [ ] Compare against modern NLP approaches
* [ ] Experiment with Word2Vec or GloVe embeddings
* [ ] Experiment with Transformer-based models
* [ ] Add ROC-AUC and Precision-Recall curves
* [ ] Perform error analysis on misclassified reviews
* [ ] Package preprocessing and prediction into a reusable pipeline
* [ ] Deploy the model as an API

---

# 📌 Conclusion

This project demonstrates a complete **classical NLP sentiment analysis workflow**, starting from raw IMDB movie reviews and ending with a tuned Machine Learning model.

Among the tested algorithms, **Logistic Regression achieved the best performance with 89.44% accuracy**, slightly outperforming Linear SVM and clearly outperforming Multinomial Naive Bayes.

The project provides a strong foundation for expanding from traditional NLP techniques toward more advanced approaches such as **Word Embeddings, Deep Learning, and Transformer-based NLP models**.

---

## 👨‍💻 Author

**Abdel-Alim Wagih Fathy**

 Data Analysis | AI Engineer | Machine Learning | NLP | Data Science

---

⭐ **If you found this project useful, consider giving the repository a star!**
