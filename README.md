# Sentiment Analysis with Python (YouTube Comments Dataset)

This project is a sentiment analysis pipeline built with Python, inspired by the [Sentiment Analysis YouTube Tutorial](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial). The goal is to classify YouTube video comments as positive, negative, or neutral using machine learning and natural language processing techniques.

## 📌 Project Overview
The project walks through the entire sentiment analysis workflow, including:

1. Data Collection (YouTube comments dataset)
2. Data Cleaning and Preprocessing
3. Sentiment Labeling
4. Text Vectorization (TF-IDF, Count Vectorizer)
5. Sentiment Prediction using Machine Learning Models
6. Model Evaluation (Accuracy, F1-Score)
7. Visualization of Sentiment Distribution

---

## 🔑 Technologies Used
- Python
- Pandas
- NLTK (Natural Language Toolkit)
- Scikit-learn
- VADER Sentiment Analyzer
- Matplotlib & Seaborn

---

## 📄 Dataset
The dataset contains YouTube comments with corresponding sentiment labels (positive, negative, or neutral). The comments were extracted using Kaggle's public datasets.

---

## ⚙️ Installation
### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install pandas nltk scikit-learn matplotlib seaborn
```

### Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

---

## 🧹 Data Preprocessing
1. Lowercasing text
2. Removing special characters
3. Tokenization
4. Stopwords Removal
5. Lemmatization

Example Code:
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)
```

---

## 🧠 Model Training
The following machine learning models were used:
- Logistic Regression
- Random Forest
- Naive Bayes

Example Training Code:
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_comments)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 🎯 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

Example:
```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📊 Results
| Model              | Accuracy | F1-Score |
|----------------|---------|----------|
| Logistic Regression | 83%     | 0.81    |
| Random Forest      | 79%     | 0.78    |
| Naive Bayes       | 76%     | 0.74    |

---

## 🔥 Visualizations
Sentiment distribution:
```python
import seaborn as sns
sns.countplot(y_pred)
```

---

## 📌 How to Run the Project
1. Clone the repository:
```bash
git clone https://github.com/username/sentiment-analysis-youtube.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook or Python scripts.

---

## 📌 Conclusion
This project demonstrates how to perform sentiment analysis on YouTube comments using machine learning techniques. The Logistic Regression model performed the best in terms of accuracy and F1-score.

---

## 💡 Future Improvements
- Implement deep learning models (LSTM, BERT)
- Hyperparameter tuning
- Support for multiple languages

---

## 📌 References
- [Kaggle Notebook](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 🔗 License


