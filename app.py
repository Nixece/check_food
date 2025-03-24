import streamlit as st
import numpy as np
from PIL import Image
from classify import classify_waste_image

st.set_page_config(page_title="Waste Classifier (TrashNet)", layout="centered")
st.title("♻️ Waste Classifier (MobileNetV2 - TrashNet)")
st.write("อัปโหลดภาพขยะ แล้วระบบจะบอกประเภทขยะพร้อมความมั่นใจ")

uploaded_file = st.file_uploader("📷 อัปโหลดรูปภาพขยะ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # แปลงเป็น numpy array
    img_array = np.array(image)

    # จำแนกขยะ
    result = classify_waste_image(img_array)

    # แสดงผล
    st.markdown("---")
    st.subheader("🧠 ผลการจำแนก:")
    st.write(f"**Label (อังกฤษ):** `{result['label_en']}`")
    st.write(f"**Label (ไทย):** `{result['label_th']}`")
    st.write(f"**ประเภทขยะ (ไทย):** 🗑️ `{result['category_th']}`")
    st.write(f"**ความมั่นใจ:** `{result['confidence'] * 100:.2f}%`")
