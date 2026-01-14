import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="Tesseract OCR App", layout="wide")
st.title("📄 Tesseract OCR with Streamlit")
st.write("Upload an image and extract text (supports Telugu + English)")

# --------------------------------
# Sidebar - Settings
# --------------------------------
st.sidebar.header("OCR Settings")

# Language options
language_map = {"English": "eng", "Telugu": "tel", "English + Telugu": "eng+tel"}
language = st.sidebar.selectbox("Select Language", list(language_map.keys()))

# OEM options (force LSTM for Telugu)
oem_map = {"LSTM only (1)": 1, "Default (3)": 3}
oem_label = st.sidebar.selectbox("OCR Engine Mode (OEM)", list(oem_map.keys()))
oem = oem_map[oem_label]

# PSM options (numeric values only)
psm_map = {
    "Auto (3)": 3,
    "Single block of text (6)": 6,
    "Single line (7)": 7,
    "Sparse text (11)": 11,
}
psm_label = st.sidebar.selectbox("Page Segmentation Mode (PSM)", list(psm_map.keys()))
psm = int(psm_map[psm_label])

# Preprocessing filters
st.sidebar.subheader("Preprocessing Filters")
resize = st.sidebar.checkbox("Resize (2x)", True)
blur = st.sidebar.checkbox("Gaussian Blur", True)
threshold = st.sidebar.checkbox("OTSU Threshold", True)

# --------------------------------
# Image upload
# --------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    processed_img = img.copy()

    # --------------------------------
    # Preprocessing
    # --------------------------------
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

    # --------------------------------
    # Display images
    # --------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed_img, use_column_width=True)

    # --------------------------------
    # OCR
    # --------------------------------
    # Ensure Tesseract uses correct executable
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

    selected_lang = language_map[language]

    # Force LSTM for Telugu (legacy engine doesn't support Telugu)
    if "tel" in selected_lang:
        oem = 3

    custom_config = f"-l {selected_lang} --oem {oem} --psm {psm}"

    if st.button("🔍 Extract Text"):
        try:
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            st.subheader("📄 Extracted Text")
            st.text_area("OCR Output", text, height=300)

            # Optional: download extracted text
            st.download_button("⬇ Download Text", text, file_name="ocr_output.txt")
        except pytesseract.TesseractError as e:
            st.error(f"Tesseract OCR failed: {e}")
