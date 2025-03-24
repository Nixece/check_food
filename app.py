import streamlit as st
import numpy as np
import cv2
from PIL import Image
from classify import classify_waste_image

st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("♻️ Waste Classifier (3 Types)")
st.write("อัปโหลดภาพขยะ แล้วระบบจะบอกว่าขยะประเภทใด")

uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # แปลงเป็น array สำหรับโมเดล
    img_array = np.array(image)

    label, score, category = classify_waste_image(img_array)
    st.markdown(f"**Label จาก ImageNet:** `{label}`")
    st.markdown(f"**ความมั่นใจ:** `{score:.2f}`")
    st.markdown(f"**ประเภทขยะ (ไทย):** 🗑️ `{category}`")
