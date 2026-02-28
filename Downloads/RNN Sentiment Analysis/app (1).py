import streamlit as st 
import numpy as np 
from tensorflow.keras.models import load_model 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence 

words_by_index = imdb.get_word_index()
reversed_words_index = {value:key for key,value in words_by_index.items()}

def encode(review): 
    review = review.lower().split()
    encode_review = [reversed_words_index.get(word,2)+3 for word in review]
    seq = sequence.pad_sequences([encode_review], maxlen=500, padding="post")
    return seq

model = load_model("my_model.h5")

def prediction(text): 
    preprocessed = encode(text)
    pred = model.predict(preprocessed)[0][0]
    return "positive" if pred > 0.5 else "negative"

st.title("Sentiment Analysis for Movies")
st.write("Write your comment for the movie to get the sentiment")

txt_area = st.text_area("Enter your review here")

if st.button("Predict Sentiment"):
    if txt_area.strip() != "":
        result = prediction(txt_area)
        st.success(f"The result is: {result}")
    else:
        st.warning("Please write a comment first.")
