# Restaurant Review Sentiment Analysis

This project implements a sentiment analysis pipeline for restaurant reviews using Natural Language Processing (NLP) techniques. The project covers text preprocessing, feature extraction via a Bag-of-Words model and sentiment classification using a Gaussian Naive Bayes classifier.

## Project Overview

The code performs the following steps:
- **Data Loading:** Reads the dataset from a tab-separated file (`Restaurant_Reviews.tsv`).
- **Data Preprocessing:** Cleans the reviews by removing non-alphabetic characters, converting text to lowercase, tokenizing, removing stopwords (while preserving the word "not"), and applying stemming.
- **Feature Extraction:** Transforms the cleaned reviews into a numerical Bag-of-Words representation (limited to the top 1500 words).
- **Model Training:** Splits the data into training and test sets and trains a Gaussian Naive Bayes classifier.
- **Evaluation:** Predicts the sentiment of test reviews and evaluates the classifier using a confusion matrix and accuracy score.

### Install Dependencies
pip install -r requirements.txt

### Usage
- Ensure that the `Restaurant_Reviews.tsv` file is in the same directory.
- Run the script:
 ```bash
python main.py
