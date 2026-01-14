import pytesseract

pytesseract.pytesseract.tesseract_cmd = "tesseract"

import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

st.set_page_config(page_title="OCR with Tesseract", layout="wide")

st.title("📄 Tesseract OCR Streamlit App")
st.write("Upload an image and extract text using Tesseract OCR")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("OCR Settings")

language_map = {"English": "eng", "Telugu": "tel", "English + Telugu": "eng+tel"}

language = st.sidebar.selectbox("Select Language", list(language_map.keys()))

oem = st.sidebar.selectbox(
    "OCR Engine Mode (OEM)",
    {
        "Legacy engine only (0)": 0,
        "Neural nets LSTM only (1)": 1,
        "Legacy + LSTM (2)": 2,
        "Default (3)": 3,
    },
)

psm = st.sidebar.selectbox(
    "Page Segmentation Mode (PSM)",
    {
        "Single block of text (6)": 6,
        "Auto (3)": 3,
        "Single line (7)": 7,
        "Sparse text (11)": 11,
    },
)

st.sidebar.subheader("Preprocessing Filters")
resize = st.sidebar.checkbox("Resize (2x)", True)
blur = st.sidebar.checkbox("Gaussian Blur", True)
threshold = st.sidebar.checkbox("OTSU Threshold", True)

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    processed_img = img.copy()

    # -------------------------------
    # Preprocessing
    # -------------------------------
    if resize:
        processed_img = cv2.resize(
            processed_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

    if blur:
        processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)

    if threshold:
        _, processed_img = cv2.threshold(
            processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # -------------------------------
    # Display Images
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_img, use_column_width=True)

    # -------------------------------
    # OCR
    # -------------------------------
    custom_config = f"-l {language_map[language]} --oem {oem} --psm {psm}"

    if st.button("🔍 Extract Text"):
        text = pytesseract.image_to_string(processed_img, config=custom_config)

        st.subheader("📄 Extracted Text")
        st.text_area("OCR Output", text, height=300)
