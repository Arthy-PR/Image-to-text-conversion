# OCR & Image Captioning Web App

A simple web application that allows users to **upload an image**, extract text using **OCR (pytesseract)**, and generate an **image description** using the **BLIP model** from Hugging Face Transformers. Built with **Streamlit** for an interactive user interface.

---

## Features

* Upload images in JPG, JPEG, or PNG format.
* Extract text from images using **pytesseract** OCR.
* Generate descriptive captions for images using **BLIP**.
* Simple and interactive **web interface** with Streamlit.
* Works offline once models are downloaded.

---

## Demo
<img width="241" height="180" alt="image" src="https://github.com/user-attachments/assets/ed25075d-e0eb-4d0f-8731-5d70e973b9e5" />

*extracted text*
* MAKE TEXT
STAND OUT FROM
BACKGROUNDS **

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ocr-image-captioning.git
cd ocr-image-captioning
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR engine**

* **Windows:** [Download installer](https://github.com/tesseract-ocr/tesseract) and add to PATH.
* **Linux:**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

* **Mac:**

```bash
brew install tesseract
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Upload an image.
2. View the extracted text in the **"Extracted Text"** section.
3. View the image caption in the **"Image Description"** section.

---

## Requirements

See `requirements.txt`:

```
streamlit==1.26.0
pillow==10.1.0
numpy==1.25.0
opencv-python==4.8.1.78
pytesseract==0.5.3
torch==2.1.0
torchvision==0.16.1
torchaudio==2.1.0
transformers==4.41.0
```

---

## Project Structure

```
ocr-image-captioning/
│
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── screenshot.png      # Optional: Screenshot of the app
```

