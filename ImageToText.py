import cv2
import pytesseract
import os
import platform

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
# Auto-detect Tesseract path (for local Windows users)
if platform.system() == "Windows":
    tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path
    else:
        print("⚠️ Warning: Tesseract not found! Ensure it's installed and in PATH.")

# Supported language codes for OCR
LANG_CODES = {
    "english": "eng",
    "hindi": "hin",
    "bengali": "ben",
    "tamil": "tam",
    "telugu": "tel"
}

def read_language():
    """Reads the selected language from language.txt."""
    lang_path = "language.txt"
    if os.path.exists(lang_path):
        with open(lang_path, "r", encoding="utf-8") as lang_file:
            return lang_file.read().strip().lower()
    return None

def extract_text(image_path, language):
    """Extracts text from an image using OCR."""
    if language not in LANG_CODES:
        raise ValueError("❌ Unsupported language! Choose from: Hindi, Bengali, Tamil, Telugu, English.")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError("❌ Image file not found!")

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Error loading image. Ensure it's a valid image file.")

    # Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding (improves OCR on noisy backgrounds)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)

    # Perform OCR
    text = pytesseract.image_to_string(processed, lang=LANG_CODES[language])

    return text.strip()

def main():
    """Main function to process the image and save extracted text."""
    language = read_language()

    if not language:
        print("❌ Error: language.txt not found or empty.")
        return

    image_path = "uploaded_image.jpg"  # Image saved by Streamlit

    try:
        extracted_text = extract_text(image_path, language)
        
        # Save extracted text to text_data.txt
        with open("text_data.txt", "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)
        
        print("✅ Text extraction completed. Check text_data.txt.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()