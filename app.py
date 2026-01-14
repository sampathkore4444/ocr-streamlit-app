import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import pyperclip
from io import BytesIO

# PDF support
try:
    from pdf2image import convert_from_bytes
except ImportError:
    st.error(
        "Install pdf2image (`pip install pdf2image`) and poppler-utils for PDF support."
    )

st.set_page_config(page_title="Smart OCR: Image & PDF", layout="wide")
st.title("📄 Smart OCR with Confidence (Image & PDF)")
st.write(
    "Upload an image or PDF, extract text line by line, and see word-level confidence"
)

# -------------------------------
# Sidebar - Settings
# -------------------------------
st.sidebar.header("OCR Settings")

language_map = {
    "English": "eng",
    "Telugu": "tel",
    "Hindi": "hin",
    "Khmer": "khm",
    "Thai": "tha",
    "Vietnamese": "vie",
    "English + Telugu": "eng+tel",
}
language = st.sidebar.selectbox("Select Language", list(language_map.keys()))

oem_map = {"LSTM only (1)": 1, "Default (3)": 3}
oem_label = st.sidebar.selectbox("OCR Engine Mode (OEM)", list(oem_map.keys()))
oem = oem_map[oem_label]

psm_map = {
    "Auto (3)": 3,
    "Single block of text (6)": 6,
    "Single line (7)": 7,
    "Sparse text (11)": 11,
}
psm_label = st.sidebar.selectbox("Page Segmentation Mode (PSM)", list(psm_map.keys()))
psm = int(psm_map[psm_label])

st.sidebar.subheader("Preprocessing Filters")
resize = st.sidebar.checkbox("Resize (2x)", True)
blur = st.sidebar.checkbox("Gaussian Blur", True)
threshold = st.sidebar.checkbox("OTSU Threshold", True)

# -------------------------------
# Upload file
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    pages = []
    if file_ext == "pdf":
        # Convert PDF pages to images
        pages = convert_from_bytes(uploaded_file.read())
    else:
        # Single image
        pages = [Image.open(uploaded_file)]

    all_text = ""
    for page_num, page in enumerate(pages, start=1):
        img = np.array(page)
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        processed_img = img_gray.copy()

        # Preprocessing
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

        # Display original vs processed (for first page only)
        if page_num == 1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image / PDF Page 1")
                st.image(img, use_column_width=True)
            with col2:
                st.subheader("Processed Image / PDF Page 1")
                st.image(processed_img, use_column_width=True)

        # OCR configuration
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
        selected_lang = language_map[language]

        if "tel" in selected_lang:
            oem = 3  # Force LSTM for Telugu

        custom_config = f"-l {selected_lang} --oem {oem} --psm {psm}"

        # OCR with data output
        data = pytesseract.image_to_data(
            processed_img, config=custom_config, output_type=pytesseract.Output.DICT
        )
        text_output = ""
        annotated_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        current_line = -1

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            line_num = data["line_num"][i]

            if word != "":
                if line_num != current_line:
                    text_output += "\n"
                    current_line = line_num
                text_output += word + " "

                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )

                # Confidence-based color
                if conf > 80:
                    color = (0, 255, 0)
                elif conf > 50:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated_img,
                    f"{conf}%",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        text_output = text_output.strip()
        all_text += f"\n\n--- Page {page_num} ---\n{text_output}"

        # Show annotated image for the first page only
        if page_num == 1:
            st.subheader("🔹 OCR Bounding Boxes (Confidence Color)")
            st.image(annotated_img, channels="BGR", use_column_width=True)

    # Show combined text output
    st.subheader("📄 Extracted Text (Preserved Lines & Pages)")
    st.text_area("OCR Output", all_text, height=400)

    # Copy button
    if st.button("📋 Copy Text"):
        pyperclip.copy(all_text)
        st.success("Text copied to clipboard!")

    # Download
    st.download_button("⬇ Download Text", all_text, file_name="ocr_output.txt")
