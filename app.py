import streamlit as st
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Load the pretrained tokenizer and your fine-tuned model
MODEL_DIR = r"C:\Users\bryso\MORINGA_Practice\PHASE_5\Capstone_Project\Bitcoin-Sentiment-Analysis\final-model"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

@st.cache(allow_output_mutation=True)

def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    return probs.numpy()

st.title("Bitcoin Sentiment Analysis")

user_input = st.text_area("Enter a comment for sentiment analysis:")
if user_input:
    probabilities = predict([user_input])
    st.write(f"Model's prediction probabilities: {probabilities}")