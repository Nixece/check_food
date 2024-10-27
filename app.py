import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ส่วนการอัปโหลดภาพ
st.title("ตรวจจับบรรจุภัณฑ์ด้วยการใช้สีพื้นหลังและตรวจจับรูปทรง")

uploaded_file = st.file_uploader("อัปโหลดภาพที่ต้องการตรวจสอบ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # โหลดภาพจากการอัปโหลดและแปลงเป็น RGB
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # แปลงภาพเป็น Grayscale และใช้ค่าเฉลี่ยสีพื้นหลัง
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background_color = np.mean(gray_image)

    # ขั้นตอนที่ 1: ตรวจจับบริเวณที่มีสีแตกต่างจากพื้นหลัง
    _, mask = cv2.threshold(gray_image, background_color - 20, 255, cv2.THRESH_BINARY_INV)

    # ค้นหา Contours บน mask ที่สร้างจากการตรวจจับสีพื้นหลัง
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ขั้นตอนที่ 2: ตรวจจับรูปทรงบน Contours ที่ผ่านการตรวจจับจากสีพื้นหลัง
    for contour in contours:
        # Approximate contour เพื่อหาโครงสร้างที่เป็นสี่เหลี่ยม
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # ตรวจสอบว่ามี 4 จุดที่ใกล้เคียงกับสี่เหลี่ยม
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # กรองเฉพาะที่มีอัตราส่วนใกล้เคียงกับสี่เหลี่ยมและขนาดใหญ่พอสมควร
            if 0.8 <= aspect_ratio <= 1.2 and w > 50 and h > 50:
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # แปลงกลับเป็น RGB เพื่อแสดงใน Streamlit
    image_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_result, caption="ผลลัพธ์หลังการตรวจจับบรรจุภัณฑ์ด้วยสีพื้นหลังและรูปทรง", use_column_width=True)
