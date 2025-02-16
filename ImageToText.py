import cv2
import pytesseract
import os

# Set Tesseract path (only for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Language codes for Tesseract OCR
LANG_CODES = {
    "english": "eng",
    "hindi": "hin",
    "bengali": "ben",
    "tamil": "tam",
    "telugu": "tel"
}

def read_language():
    """Reads the selected language from language.txt."""
    if os.path.exists("language.txt"):
        with open("language.txt", "r", encoding="utf-8") as lang_file:
            return lang_file.read().strip().lower()
    return None

def extract_text(image_path, language):
    """Extracts text from the given image using OCR."""
    if language not in LANG_CODES:
        raise ValueError("Unsupported language. Choose from: Hindi, Bengali, Tamil, Telugu, English.")
    
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale for better OCR performance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (optional: improves OCR accuracy)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform OCR
    text = pytesseract.image_to_string(gray, lang=LANG_CODES[language])
    
    return text.strip()

def main():
    """Main function to process image and save extracted text."""
    language = read_language()
    
    if not language:
        print("Error: language.txt not found or empty.")
        return

    image_path = "uploaded_image.jpg"  # This is the image saved by Streamlit

    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return

    extracted_text = extract_text(image_path, language)
    
    # Save extracted text in text_data.txt
    with open("text_data.txt", "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)
    
    print("Text extraction completed. Check text_data.txt.")

if __name__ == "__main__":
    main()