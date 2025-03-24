import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# โหลดโมเดล MobileNetV2 ที่เทรนด้วย ImageNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Mapping label อังกฤษ → ประเภทขยะไทย
def map_label_to_waste(label):
    label = label.lower()

    recycle = ['bottle', 'can', 'carton', 'glass', 'paper', 'plastic', 'tin', 'box']
    organic = ['banana', 'apple', 'orange', 'vegetable', 'food', 'fruit', 'corn', 'meat']
    general = ['trash', 'diaper', 'tissue', 'sponge', 'wrapper']

    if any(word in label for word in recycle):
        return 'ขยะรีไซเคิล'
    elif any(word in label for word in organic):
        return 'ขยะเปียก'
    else:
        return 'ขยะทั่วไป'

# แปล label ภาษาอังกฤษ → ไทย (แบบง่าย)
def translate_label(label):
    translations = {
        'banana': 'กล้วย',
        'apple': 'แอปเปิ้ล',
        'plastic_bottle': 'ขวดพลาสติก',
        'carton': 'กล่องกระดาษ',
        'glass': 'แก้ว',
        'can': 'กระป๋อง',
        'bottle': 'ขวด',
        'trash': 'ขยะ',
        'tissue': 'ทิชชู่',
        'box': 'กล่อง',
        'metal_can': 'กระป๋องโลหะ'
    }
    return translations.get(label.lower(), 'ไม่ทราบ')

# ประมวลผลภาพและทำนาย
def classify_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
    return decoded  # (id, label, score)

# UI ด้วย Streamlit
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("♻️ Waste Classifier (MobileNetV2 - ImageNet)")
st.write("อัปโหลดภาพ แล้วระบบจะพยายามแยกประเภทขยะให้คุณ")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพขยะ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # ทำนาย
    imagenet_id, label, score = classify_image(img_np)
    label_th = translate_label(label)
    category = map_label_to_waste(label)

    # แสดงผลลัพธ์
    st.markdown("---")
    st.subheader("🧠 ผลการจำแนก:")
    st.write(f"**Label (อังกฤษ):** `{label}`")
    st.write(f"**Label (ไทย):** `{label_th}`")
    st.write(f"**ประเภทขยะ:** 🗑️ `{category}`")
    st.write(f"**ความมั่นใจ:** `{score * 100:.2f}%`")
