import pytesseract
import cv2
from PIL import Image
import numpy as np
from google.colab import files
from IPython.display import display

# Optional: for image captioning
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# -------------------------------
# 1. Upload Image
# -------------------------------
uploaded = files.upload()  # User uploads image

for filename in uploaded.keys():
    image = Image.open(filename)
    display(image)  # Display uploaded image

    # -------------------------------
    # 2. OCR: Extract Text
    # -------------------------------
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)

    text = pytesseract.image_to_string(gray)

    print("----- Extracted Text -----")
    print(text if text.strip() != "" else "No text found in the image.")

    # -------------------------------
    # 3. Optional: Image Captioning
    # -------------------------------
    print("\n----- Image Description -----")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    print(description)
