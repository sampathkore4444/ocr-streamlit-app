import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

st.set_page_config(page_title="OCR Confidence App", layout="wide")
st.title("📄 Smart OCR: Multi-Languages Text Extraction with Confidence Visualization")
st.write("Upload an image, extract text, and see confidence for each word")

# -------------------------------
# Sidebar - Settings
# -------------------------------
st.sidebar.header("OCR Settings")

language_map = {"English": "eng", "Telugu": "tel", "English + Telugu": "eng+tel"}
language = st.sidebar.selectbox("Select Language", list(language_map.keys()))

# OEM options
oem_map = {"LSTM only (1)": 1, "Default (3)": 3}
oem_label = st.sidebar.selectbox("OCR Engine Mode (OEM)", list(oem_map.keys()))
oem = oem_map[oem_label]

# PSM options
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

# -------------------------------
# Upload image
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to grayscale
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    processed_img = img_gray.copy()

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
    # Display original vs processed
    # -------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed_img, use_column_width=True)

    # -------------------------------
    # OCR with confidence
    # -------------------------------
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    selected_lang = language_map[language]

    # Force LSTM for Telugu
    if "tel" in selected_lang:
        oem = 3

    custom_config = f"-l {selected_lang} --oem {oem} --psm {psm}"

    if st.button("🔍 Extract Text with Confidence"):
        try:
            # OCR with word-level data
            data = pytesseract.image_to_data(
                processed_img, config=custom_config, output_type=pytesseract.Output.DICT
            )
            text_output = ""
            annotated_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

            current_line = -1  # track line changes

            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                conf = int(data["conf"][i])
                line_num = data["line_num"][i]

                if word != "":
                    # Add a newline if the line changes
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
                        color = (0, 255, 0)  # Green = high confidence
                    elif conf > 50:
                        color = (0, 165, 255)  # Orange = medium confidence
                    else:
                        color = (0, 0, 255)  # Red = low confidence

                    # Draw bounding box and confidence
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

            # Display text
            st.subheader("📄 Extracted Text (Preserved Lines)")
            st.text_area("OCR Output", text_output, height=300)

            # Display annotated image
            st.subheader("🔹 OCR Bounding Boxes (Color = Confidence)")
            st.image(annotated_img, channels="BGR", use_column_width=True)

            # Download text
            st.download_button(
                "⬇ Download Text", text_output, file_name="ocr_output.txt"
            )

        except pytesseract.TesseractError as e:
            st.error(f"Tesseract OCR failed: {e}")
