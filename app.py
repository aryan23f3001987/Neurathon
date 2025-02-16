import streamlit as st
import os
import subprocess
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained Fake News Model & Tokenizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Function to predict news authenticity
def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding='post', maxlen=100)
    prediction = model.predict(padded)[0][0]  # Get the first prediction value
    return "🟢 True News" if prediction > 0.5 else "🔴 Fake News"

def main():
    st.title("📰 Fake News Detector with Image & Text Support")

    # Remove previous files on reload
    for file in ["uploaded_image.jpg", "language.txt", "text_data.txt", "translated.txt"]:
        if os.path.exists(file):
            os.remove(file)

    # Let user choose between Image or Text
    option = st.radio("📌 Choose Input Method:", ("Upload Image", "Enter Text"))

    # Language selection
    language = st.selectbox("🌍 Select Language", ["Hindi", "English", "Bengali", "Tamil", "Telugu"])

    translated_text = ""

    if option == "Upload Image":
        uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Save uploaded image
            with open("uploaded_image.jpg", "wb") as f:
                f.write(uploaded_file.read())
            st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
            
            # Save language selection
            with open("language.txt", "w", encoding="utf-8") as lang_file:
                lang_file.write(language)

            # Run ImageToText.py
            subprocess.run(["python", "ImageToText.py"])

            # Run translation_to_english.py
            subprocess.run(["python", "translation_to_english.py"])

    elif option == "Enter Text":
        text_input = st.text_area("🔍 Enter Text Manually (Max: 4999 characters):", max_chars=4999)
        if text_input:
            # Save language in a separate file
            with open("language.txt", "w", encoding="utf-8") as lang_file:
                lang_file.write(language)

            # Save text input in a separate file
            with open("text_data.txt", "w", encoding="utf-8") as text_file:
                text_file.write(text_input[:4999])  # Ensuring the text is within limit

            # Run translation_to_english.py
            subprocess.run(["python", "translation_to_english.py"])

    # Display translated text & predict fake news
    if os.path.exists("translated.txt"):
        with open("translated.txt", "r", encoding="utf-8") as trans_file:
            translated_text = trans_file.read()
        
        st.subheader("🌍 Translated Text:")
        st.write(translated_text)

        # Predict Fake or True News
        if translated_text:
            prediction = predict_news(translated_text)
            st.subheader("📰 **News Authenticity:**")
            st.write(prediction)

if __name__ == "__main__":
    main()