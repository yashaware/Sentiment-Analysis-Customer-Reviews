Sentiment Analysis of Customer Reviews

📌 Project Overview

This project performs sentiment analysis on customer reviews using Natural Language Processing (NLP) techniques. It classifies the sentiment as positive or negative based on the review text using TF-IDF vectorization and a Logistic Regression model.

🔧 Technologies Used

Python 3.9+
Jupyter Notebook (VS Code)
scikit-learn
pandas, numpy
matplotlib, seaborn
nltk (Natural Language Toolkit)
vaderSentiment (optional)

📁 Dataset

Dataset: Twitter Sentiment Analysis
Columns used: tweet, label (renamed to text, sentiment)
Sentiment Labels: 0 = Negative, 1 = Positive

🔄 Workflow

Text Preprocessing
Remove punctuation and non-alphabet characters
Convert to lowercase
Tokenize
Remove stopwords
Stemming
Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency)

Model Building

Train/Test Split

Logistic Regression model

Evaluation

Accuracy

Confusion Matrix

Classification Report

Prediction Function

Predicts custom input sentiment

📊 Results

Model Accuracy: ~85%

Classifier: Logistic Regression

Feature Extraction: TF-IDF

▶️ How to Run

Clone the repo or download the .ipynb file.

Open with Jupyter Notebook in VS Code.

Install dependencies:

%pip install pandas numpy matplotlib seaborn scikit-learn nltk vaderSentiment

Run all cells sequentially.

🚀 Future Scope

Integrate with Flask or Streamlit for live web app

Deploy on Render or Streamlit Cloud

Use BERT or LSTM for better accuracy

Add neutral sentiment class

🙌 Acknowledgements

Dataset by dD2405 on GitHub

scikit-learn & NLTK documentation

🔗 Connect

Made with ❤️ by Yash Aware

🏷️ Tags

#NLP #SentimentAnalysis #DataScience #MachineLearning #Python #JupyterNotebook

