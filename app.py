import streamlit as st
import numpy as np
from PIL import Image
import cv2

from classify import classify_image, map_label_to_waste_category

st.set_page_config(page_title="Waste Classifier (5 Types)", layout="centered")
st.title("♻️ Waste Classifier (MobileNetV2 - ImageNet)")
st.write("อัปโหลดภาพขยะ แล้วระบบจะบอกประเภทขยะตามมาตรฐานประเทศไทย (5 ประเภท)")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # แปลงเป็น numpy array
    img_array = np.array(image)

    # ทำนาย
    imagenet_id, label, confidence = classify_image(img_array)
    category = map_label_to_waste_category(label)

    st.markdown("---")
    st.subheader("🧠 ผลการจำแนก:")
    st.write(f"**Label (อังกฤษ):** `{label}`")
    st.write(f"**ความมั่นใจ:** `{confidence * 100:.2f}%`")
    st.write(f"**ประเภทขยะ (ไทย):** 🗑️ `{category}`")
