import streamlit as st
import os
import subprocess
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure deep-translator is installed
os.system("pip install deep-translator")

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
    return "\U0001F7E2 True News" if prediction > 0.5 else "\U0001F534 Fake News"

def main():
    st.title("\U0001F4F0 Fake News Detector with Image & Text Support")
    
    st.write("\U0001F4DD **Note:** Refresh after every use before new data input.")

    # Remove previous files on reload
    for file in ["uploaded_image.jpg", "language.txt", "text_data.txt", "translated.txt"]:
        if os.path.exists(file):
            os.remove(file)

    # Let user choose between Image or Text
    option = st.radio("\U0001F4CC Choose Input Method:", ("Upload Image", "Enter Text"))

    # Language selection
    language = st.selectbox("\U0001F30D Select Language", ["Hindi", "English", "Bengali", "Tamil", "Telugu"])

    translated_text = ""

    if option == "Upload Image":
        uploaded_file = st.file_uploader("\U0001F4F7 Upload an Image", type=["jpg", "png", "jpeg"])
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

            # Debugging: Check if text_data.txt exists
            if os.path.exists("text_data.txt"):
                with open("text_data.txt", "r", encoding="utf-8") as check_file:
                    extracted_text = check_file.read().strip()
                    if extracted_text:
                        st.write("✅ Extracted Text from Image!")
                    else:
                        st.write("❌ Image text extraction failed: `text_data.txt` is empty!")
            else:
                st.write("❌ `text_data.txt` was not created!")

            # Run translation_to_english.py
            subprocess.run(["python", "translation_to_english.py"])

    elif option == "Enter Text":
        text_input = st.text_area("\U0001F50D Enter Text Manually (Max: 4999 characters):", max_chars=4999)

        if st.button("\U0001F504 Process Text"):
            if text_input.strip():  # Ensure it's not empty
                # Save language selection
                with open("language.txt", "w", encoding="utf-8") as lang_file:
                    lang_file.write(language)

                # Save user input in text_data.txt
                with open("text_data.txt", "w", encoding="utf-8") as text_file:
                    text_file.write(text_input.strip())

                # Verify if text was saved
                if os.path.exists("text_data.txt"):
                    with open("text_data.txt", "r", encoding="utf-8") as check_file:
                        saved_text = check_file.read().strip()
                        if saved_text:
                            st.write("✅ Text successfully saved!")
                        else:
                            st.write("❌ Error: Text not saved correctly!")
                else:
                    st.write("❌ `text_data.txt` does not exist! Something went wrong.")

                # Run translation script
                subprocess.run(["python", "translation_to_english.py"])
            else:
                st.write("❌ Please enter some text before processing.")

    # Display translated text & predict fake news
    if os.path.exists("translated.txt"):
        with open("translated.txt", "r", encoding="utf-8") as trans_file:
            translated_text = trans_file.read().strip()

        if translated_text:
            st.subheader("\U0001F30D Translated Text:")
            st.write(translated_text)

            # Predict Fake or True News
            prediction = predict_news(translated_text)
            st.subheader("\U0001F4F0 **News Authenticity:**")
            st.write(prediction)
        else:
            st.write("❌ Translation failed: `translated.txt` is empty!")
    else:
        st.write("❌ Translation file `translated.txt` not found!")

if __name__ == "__main__":
    main()
