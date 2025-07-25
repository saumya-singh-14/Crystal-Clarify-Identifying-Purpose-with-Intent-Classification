# Intentify

A deep learning project for classifying user intents (e.g., AddToPlaylist, GetWeather, etc.) from text input.
Uses Convolutional Neural Networks (CNN) + pre-trained GloVe embeddings to learn from short text sentences, and deploys as an interactive web app using Streamlit.

# Project Overview

This project demonstrates:

Data preprocessing from raw JSON to clean, trainable format

Building & training a CNN text classifier

Using pre-trained word embeddings (GloVe) for better performance

Label encoding & saving preprocessors for reuse

Real-time predictions via Streamlit UI

The system predicts intents from text input, supporting these 6 categories - AddToPlaylist, BookRestaurant, GetWeather, RateBook, SearchCreativeWork, SearchScreeningEvent

# Architecture & Steps

1. Data Preparation (prepare.py)
Reads raw JSON files from data/raw_json_data/
Cleans text (removes symbols, lowercases)
Saves:
.txt files (cleaned text)
.npy arrays (train_text.npy, train_label.npy, etc.)
Labels are kept as strings initially (e.g., "GetWeather")

2. Model Training (train.py)
Loads:
Pre-trained GloVe vectors (glove.6B.100d.txt)
Cleaned data (.npy)
Tokenizes text -> converts to padded sequences
Encodes labels with LabelEncoder → saves encoder (label_encoder.pkl)

CNN:
Embedding layer (with GloVe weights, frozen)
Multiple convolution filters ( andnel sizes: 2,3,5)
Max-pooling
Dropout (prevents overfitting)
Dense layer with softmax output
Compiles with categorical_crossentropy loss & adam optimizer
Splits into train & validation sets
Trains & plots accuracy / loss

Saves:
Trained model (model.h5)
Tokenizer (tokenizer.pkl)
Label encoder

3. Prediction & Testing (test.py)
Loads saved model & preprocessors
Defines functions to:
Tokenize & pad new input text
Predict intent (using softmax argmax)
Runs a Streamlit app:
User enters sentence → shows predicted intent in real time

# Performance

Training accuracy: ~99%

Validation accuracy: ~98%

Robust even on small dataset, thanks to GloVe embeddings

# Setup & Installation

1. Make sure you have Python 3.7+ installed.

2. Install required packages:
pip install numpy matplotlib scikit-learn  andas tensorflow joblib streamlit

3. How to Run

    i. Prepare data
    python prepare.py
    (Cleans text, creates .txt and .npy files)
    
    ii. Train model
    python train.py
    (Builds CNN model, loads GloVe embeddings, saves: model.h5 tokenizer.pkl label_encoder.pkl, plots accuracy & loss curves)
    
    iii. Predict intents (web app)
    streamlit run test.py
    (Opens browser, enter sentence -> predicts intent)
