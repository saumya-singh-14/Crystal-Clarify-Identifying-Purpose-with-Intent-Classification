import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
import joblib

# Load the saved model
model = load_model('model.h5')
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 50

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Function to preprocess the input text
def preprocess_text(text):
    # Tokenize the text
    tokenizer = joblib.load('tokenizer.pkl')
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences

# Function to predict intent
def predict_intent(sentence):
    # Preprocess the text
    preprocessed_text = preprocess_text(sentence)
    # Make prediction
    prediction = model.predict(preprocessed_text)
    # Get the predicted label
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Streamlit app
def main():
    st.title("Intent Classification")

    # Text input for user to enter a sentence
    user_input = st.text_input("Enter a sentence:")

    # Button to submit the input
    if st.button("Submit"):
        if user_input:
            # Predict the intent
            intent = predict_intent(user_input)
            st.write(f"Predicted intent: {intent}")
        else:
            st.write("Please enter a sentence.")

if __name__ == "__main__":
    main()
