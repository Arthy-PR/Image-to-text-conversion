import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="OCR & Image Captioning", layout="centered")
st.title("📄 OCR & Image Captioning App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OCR using EasyOCR
    reader = easyocr.Reader(['en'])
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = reader.readtext(image_cv)
    text = " ".join([res[1] for res in result])

    st.subheader("Extracted Text")
    if text.strip():
        st.text(text)
    else:
        st.text("No text found.")

    # Image Captioning
    st.subheader("Image Description")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    st.text(description)
