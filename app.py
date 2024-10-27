import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ส่วนการอัปโหลดภาพ
st.title("ตรวจจับบรรจุภัณฑ์จากรูปทรงสี่เหลี่ยม")

uploaded_file = st.file_uploader("อัปโหลดภาพที่ต้องการตรวจสอบ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # โหลดภาพจากการอัปโหลดและแปลงเป็น RGB
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # แปลงภาพเป็น Grayscale และใช้ GaussianBlur เพื่อลด noise
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # ตรวจจับขอบในภาพด้วย Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # ค้นหา Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # วาดกรอบสี่เหลี่ยมรอบ Contours ที่มีลักษณะเป็นสี่เหลี่ยม
    for contour in contours:
        # Approximate contour เพื่อหาโครงสร้างที่เป็นสี่เหลี่ยม
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # ตรวจสอบจำนวนจุดและลักษณะเป็นสี่เหลี่ยม
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # กรองให้แสดงเฉพาะที่มีอัตราส่วนใกล้เคียงกับสี่เหลี่ยม
            if 0.8 <= aspect_ratio <= 1.2 and w > 50 and h > 50:
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # แปลงกลับเป็น RGB เพื่อแสดงใน Streamlit
    image_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_result, caption="ผลลัพธ์หลังการตรวจจับบรรจุภัณฑ์ตามรูปทรง", use_column_width=True)
