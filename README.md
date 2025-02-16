Fake News Detection with Streamlit

Overview :

This project is a Fake News Detection Model with Text and Image Support, built using Python, Streamlit, and machine learning techniques. 
The model predicts whether a news article is fake or real based on text extracted from an image or manually entered by the user.

Features :

1. Text Input Support: Users can manually enter text to analyze.
2. Image Upload Support: Users can upload images containing text for analysis.
3. Multi-language Support: Supports Hindi, English, Bengali, Tamil, and Telugu.
4. Translation to English: Automatically translates text before processing.
5. Machine Learning Model: Utilizes a trained deep learning model for fake news classification.
6. User-friendly Interface: Powered by Streamlit for easy interaction.

Functionality :

Choose Input Method:

1. Upload an image (JPG, PNG, JPEG) containing text.
   Enter text manually.

2. Select Language:
   Choose from Hindi, English, Bengali, Tamil, or Telugu.

3. Processing Steps:
   If an image is uploaded, the app extracts text using ImageToText.py.
   The extracted or manually entered text is saved in text_data.txt.
   The script translation_to_english.py translates non-English text to English.

4. Fake News Detection:
  The translated text is analyzed using the trained fake news detection model.
  The model predicts whether the news is True (🟢) or Fake (🔴).

File Structure :

1. app.py - Main Streamlit application.
2. fake_news_model.pkl - Pre-trained Fake News Detection model.
3. tokenizer.pkl - Tokenizer used for text preprocessing.
4. ImageToText.py - Extracts text from images.
5. translation_to_english.py - Translates text to English.
6. requirements.txt - Required dependencies.

Contributors :

1. Aryan
2. Bedanta Roy
3. Avanish


